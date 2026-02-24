"""Evaluation metrics for orbit prediction models.

Computes MAE and RMSE in physical units (km) and generates
comparison tables across models and prediction horizons.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_pytorch_model(
    model: nn.Module,
    test_loader: DataLoader,
    denormalize_fn=None,
    device: str = None,
) -> dict:
    """Evaluate a PyTorch model on test data.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        denormalize_fn: Optional function to convert predictions back to km
        device: Device to run on

    Returns:
        Dict of metrics
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for batch in test_loader:
        inputs = batch[0].to(device)
        targets = batch[-1]

        preds = model(inputs).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Denormalize if function provided
    if denormalize_fn is not None:
        predictions = denormalize_fn(predictions)
        targets = denormalize_fn(targets)

    return compute_metrics(predictions, targets)


def evaluate_tf_model(model, test_data: tuple, denormalize_fn=None) -> dict:
    """Evaluate a Keras model on test data."""
    test_inputs, test_targets = test_data
    predictions = model.predict(test_inputs, verbose=0)

    if denormalize_fn is not None:
        predictions = denormalize_fn(predictions)
        test_targets = denormalize_fn(test_targets)

    return compute_metrics(predictions, test_targets)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute position error metrics.

    Args:
        predictions: (N, horizon, 3) predicted positions
        targets: (N, horizon, 3) ground truth positions

    Returns:
        Dict with mae_km, rmse_km, and per-horizon breakdowns
    """
    # Position error per timestep (Euclidean distance in km)
    errors = np.linalg.norm(predictions - targets, axis=-1)  # (N, horizon)

    # Overall metrics
    mae_km = float(np.mean(errors))
    rmse_km = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(errors))

    # Per-axis errors
    axis_mae = {}
    for i, axis in enumerate(["x", "y", "z"]):
        axis_mae[f"mae_{axis}_km"] = float(np.mean(np.abs(predictions[..., i] - targets[..., i])))

    # Horizon breakdown (error at specific time steps)
    horizon_steps = predictions.shape[1]
    horizon_breakdown = {}

    checkpoints = {
        "1h": 60,
        "3h": 180,
        "6h": 360,
        "12h": 720,
        "24h": 1440,
    }

    for label, step in checkpoints.items():
        if step <= horizon_steps:
            horizon_breakdown[f"mae_{label}"] = float(np.mean(errors[:, :step]))
            horizon_breakdown[f"rmse_{label}"] = float(np.sqrt(np.mean(errors[:, :step] ** 2)))

    return {
        "mae_km": mae_km,
        "rmse_km": rmse_km,
        "max_error_km": max_error,
        "median_error_km": float(np.median(errors)),
        **axis_mae,
        **horizon_breakdown,
        "error_over_time": np.mean(errors, axis=0).tolist(),
    }


def comparison_table(results: dict[str, dict]) -> str:
    """Generate a formatted comparison table.

    Args:
        results: Dict mapping model names to their metric dicts

    Returns:
        Formatted table string
    """
    headers = ["Model", "MAE (km)", "RMSE (km)", "Median (km)", "Max (km)"]
    horizon_headers = ["MAE 1h", "MAE 6h", "MAE 24h"]

    # Check which horizon columns are available
    available_horizons = []
    for h in horizon_headers:
        key = h.lower().replace(" ", "_")
        if any(key in r for r in results.values()):
            available_horizons.append(h)
            headers.append(h)

    rows = []
    for name, metrics in results.items():
        row = [
            name,
            f"{metrics.get('mae_km', 0):.2f}",
            f"{metrics.get('rmse_km', 0):.2f}",
            f"{metrics.get('median_error_km', 0):.2f}",
            f"{metrics.get('max_error_km', 0):.2f}",
        ]
        for h in available_horizons:
            key = h.lower().replace(" ", "_")
            val = metrics.get(key, None)
            row.append(f"{val:.2f}" if val is not None else "N/A")
        rows.append(row)

    # Format table
    col_widths = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)

    lines = [header_line, separator]
    for row in rows:
        lines.append(" | ".join(v.ljust(w) for v, w in zip(row, col_widths)))

    return "\n".join(lines)
