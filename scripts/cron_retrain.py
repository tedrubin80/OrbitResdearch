#!/usr/bin/env python3
"""Weekly cron: retrain ISS LSTM, push if improved + sane.

Cron:
    0 3 * * 0 cd /var/www/orbit && .venv/bin/python3 scripts/cron_retrain.py >> logs/retrain.log 2>&1
"""
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cron-retrain")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

HF_MODEL_REPO = "datamatters24/orbital-chaos-predictor"
EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
PATIENCE = 10


def load_current_best_loss():
    ckpt_path = CHECKPOINT_DIR / "lstm_iss_6h_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt.get("val_loss", float("inf"))
    return float("inf")


def detect_architecture():
    ckpt_path = CHECKPOINT_DIR / "lstm_iss_6h_best.pt"
    if not ckpt_path.exists():
        return 128, 3, 360
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]
    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
    layer_keys = [k for k in state if k.startswith("lstm.weight_ih_l")]
    num_layers = len(layer_keys) // 2
    fc_keys = sorted([k for k in state if k.startswith("fc.") and k.endswith(".weight")])
    horizon = state[fc_keys[-1]].shape[0] // 3
    return hidden_dim, num_layers, horizon


def sanity_check(model, test_input, stats):
    """Verify ISS predictions are at realistic altitude."""
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(test_input[:1]).float()).squeeze(0).numpy()
    pos_km = np.zeros(3)
    for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
        pos_km[i] = pred[0, i] * stats["iss"]["std"][col] + stats["iss"]["mean"][col]
    dist = np.linalg.norm(pos_km)
    ok = 6350 < dist < 6850
    log.info(f"Sanity {'passed' if ok else 'FAILED'}: {dist:.0f} km from center")
    return ok


def main():
    start_time = time.time()
    log.info("=== Weekly retrain start ===")

    orbit_path = DATA_DIR / "iss_2023-01-01_2025-12-31.parquet"
    if not orbit_path.exists():
        log.error(f"Data not found: {orbit_path}")
        return 1

    from src.data.preprocessing import OrbitPreprocessor
    proc = OrbitPreprocessor()
    df = pd.read_parquet(orbit_path)
    processed = proc.preprocess(df, "iss")
    stats = proc.stats

    stats_dir = RESULTS_DIR / "norm_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / "iss_norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    hidden_dim, num_layers, horizon = detect_architecture()
    log.info(f"Architecture: hidden={hidden_dim}, layers={num_layers}, horizon={horizon}")

    norm_feat_cols = sorted([c for c in processed.columns if c.endswith("_norm")])
    norm_tgt_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]
    input_steps = 1440
    stride_steps = 360

    inputs, targets = [], []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon:
            continue
        feats = seg[norm_feat_cols].values
        tgts = seg[norm_tgt_cols].values
        for i in range(0, len(seg) - input_steps - horizon, stride_steps):
            inputs.append(feats[i:i + input_steps])
            targets.append(tgts[i + input_steps:i + input_steps + horizon])

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)

    n = len(inputs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_x, train_y = inputs[:n_train], targets[:n_train]
    val_x, val_y = inputs[n_train:n_train + n_val], targets[n_train:n_train + n_val]
    test_x = inputs[n_train + n_val:]
    log.info(f"Data: {n} windows, train={n_train}, val={n_val}, test={len(test_x)}")

    from scripts.train_gpu import OrbitLSTMDirect
    model = OrbitLSTMDirect(
        input_dim=train_x.shape[-1], hidden_dim=hidden_dim,
        num_layers=num_layers, horizon=horizon,
        output_dim=train_y.shape[-1], dropout=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    train_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=BATCH_SIZE
    )

    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_losses = []
        for x, y in train_dl:
            optimizer.zero_grad()
            pred = model(x)
            ml = min(pred.shape[1], y.shape[1])
            loss = criterion(pred[:, :ml], y[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())
        scheduler.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x)
                ml = min(pred.shape[1], y.shape[1])
                v_losses.append(criterion(pred[:, :ml], y[:, :ml]).item())

        avg_v = np.mean(v_losses)
        log.info(f"Epoch {epoch:3d}/{EPOCHS} | train={np.mean(t_losses):.6f} | val={avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / "lstm_iss_6h_retrained.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                log.info(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start_time
    log.info(f"Done in {elapsed / 60:.1f} min, best val={best_val:.6f}")

    current_best = load_current_best_loss()
    log.info(f"Current best: {current_best:.6f}, new: {best_val:.6f}")

    retrained_path = CHECKPOINT_DIR / "lstm_iss_6h_retrained.pt"
    if best_val >= current_best:
        log.info("No improvement - keeping existing")
        retrained_path.unlink(missing_ok=True)
        return 0

    retrained_ckpt = torch.load(retrained_path, map_location="cpu", weights_only=False)
    model.load_state_dict(retrained_ckpt["model_state_dict"])

    if not sanity_check(model, test_x, stats):
        log.error("Sanity failed - keeping existing")
        retrained_path.unlink(missing_ok=True)
        return 1

    shutil.copy2(retrained_path, CHECKPOINT_DIR / "lstm_iss_6h_best.pt")
    retrained_path.unlink(missing_ok=True)
    log.info("Promoted retrained checkpoint")

    try:
        from huggingface_hub import HfApi
        HfApi().upload_file(
            path_or_fileobj=str(CHECKPOINT_DIR / "lstm_iss_6h_best.pt"),
            path_in_repo="checkpoints/lstm_iss_6h_best.pt",
            repo_id=HF_MODEL_REPO, repo_type="model",
        )
        log.info(f"Pushed to {HF_MODEL_REPO}")
    except Exception as e:
        log.warning(f"HF push failed (non-fatal): {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
