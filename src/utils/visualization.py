"""Visualization utilities for orbit prediction and solar wind analysis.

Provides 3D trajectory plots, error analysis, and solar wind overlays
using Matplotlib and Plotly.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_3d_orbit(
    positions: np.ndarray,
    predicted: np.ndarray = None,
    title: str = "Spacecraft Orbit",
    labels: tuple = ("Actual", "Predicted"),
    save_path: str = None,
):
    """Plot 3D orbit trajectory with optional predictions.

    Args:
        positions: (N, 3) actual positions [x, y, z] in km
        predicted: (N, 3) predicted positions (optional)
        title: Plot title
        labels: Legend labels for actual and predicted
        save_path: Path to save figure (optional)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            "b-", alpha=0.7, linewidth=1, label=labels[0])

    if predicted is not None:
        ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2],
                "r--", alpha=0.7, linewidth=1, label=labels[1])

    # Plot Earth at origin
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    earth_r = 6371  # km
    xe = earth_r * np.outer(np.cos(u), np.sin(v))
    ye = earth_r * np.outer(np.sin(u), np.sin(v))
    ze = earth_r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(xe, ye, ze, alpha=0.3, color="cyan")

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_3d_orbit_plotly(
    positions: np.ndarray,
    predicted: np.ndarray = None,
    title: str = "Spacecraft Orbit",
):
    """Interactive 3D orbit plot using Plotly.

    Args:
        positions: (N, 3) actual positions
        predicted: (N, 3) predicted positions (optional)
        title: Plot title

    Returns:
        Plotly Figure object
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="lines", name="Actual",
        line=dict(color="blue", width=2),
    ))

    if predicted is not None:
        fig.add_trace(go.Scatter3d(
            x=predicted[:, 0], y=predicted[:, 1], z=predicted[:, 2],
            mode="lines", name="Predicted",
            line=dict(color="red", width=2, dash="dash"),
        ))

    # Earth sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    r = 6371
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, "lightblue"], [1, "lightblue"]],
        showscale=False, opacity=0.3, name="Earth",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        width=900, height=700,
    )

    return fig


def plot_prediction_error(
    errors_over_time: np.ndarray,
    time_resolution_minutes: int = 1,
    title: str = "Prediction Error Over Horizon",
    save_path: str = None,
):
    """Plot how prediction error grows with horizon.

    Args:
        errors_over_time: (horizon_steps,) mean error at each step
        time_resolution_minutes: Minutes per step
        title: Plot title
        save_path: Path to save figure
    """
    hours = np.arange(len(errors_over_time)) * time_resolution_minutes / 60

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hours, errors_over_time, "b-", linewidth=2)
    ax.set_xlabel("Prediction Horizon (hours)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    results: dict[str, dict],
    metric: str = "error_over_time",
    time_resolution_minutes: int = 1,
    title: str = "Model Comparison",
    save_path: str = None,
):
    """Plot error curves for multiple models.

    Args:
        results: Dict mapping model names to metric dicts
        metric: Key for the time-series metric
        time_resolution_minutes: Minutes per step
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, metrics) in enumerate(results.items()):
        if metric in metrics:
            data = np.array(metrics[metric])
            hours = np.arange(len(data)) * time_resolution_minutes / 60
            color = colors[i % len(colors)]
            ax.plot(hours, data, linewidth=2, label=name, color=color)

    ax.set_xlabel("Prediction Horizon (hours)")
    ax.set_ylabel("Position Error (km)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_solar_wind_correlation(
    solar_params: np.ndarray,
    orbit_errors: np.ndarray,
    param_names: list[str],
    title: str = "Solar Wind vs Orbit Perturbation",
    save_path: str = None,
):
    """Scatter plots of solar wind parameters vs orbit prediction error.

    Args:
        solar_params: (N, n_params) solar wind parameter values
        orbit_errors: (N,) orbit prediction errors
        param_names: Names for each solar wind parameter
        title: Plot title
        save_path: Path to save figure
    """
    n_params = solar_params.shape[1]
    cols = min(n_params, 4)
    rows = (n_params + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        ax.scatter(solar_params[:, i], orbit_errors, alpha=0.3, s=5)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Position Error (km)")
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        mask = ~(np.isnan(solar_params[:, i]) | np.isnan(orbit_errors))
        if mask.sum() > 10:
            corr = np.corrcoef(solar_params[mask, i], orbit_errors[mask])[0, 1]
            ax.set_title(f"r = {corr:.3f}")

    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history: dict,
    title: str = "Training History",
    save_path: str = None,
):
    """Plot training and validation loss curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'lr' lists
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train")
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if "lr" in history:
        ax2.plot(epochs, history["lr"], "g-")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
