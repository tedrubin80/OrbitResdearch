#!/usr/bin/env python3
"""Hourly cron: fetch ISS positions, run LSTM, write predictions.json.

Cron:
    0 * * * * cd /var/www/orbit && .venv/bin/python3 scripts/cron_predict.py >> logs/predict.log 2>&1
"""
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cron-predict")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "lstm_iss_6h_best.pt"
STATS_FILE = PROJECT_ROOT / "results" / "norm_stats" / "iss_norm_stats.json"
OUTPUT_FILE = PROJECT_ROOT / "public" / "predictions.json"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def fetch_recent_iss(hours=25):
    """Fetch last N hours of ISS positions from NASA SSC API."""
    from src.data.ssc_client import SSCClient
    client = SSCClient()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=hours)
    df = client.fetch_positions(
        spacecraft_id="iss",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        coord_systems=["Gse"],
    )
    if df is None or len(df) == 0:
        raise RuntimeError("No ISS data from NASA SSC API")
    log.info(f"Fetched {len(df)} ISS positions")
    return df


def preprocess_with_training_stats(df, training_stats):
    """Derive velocity, normalize with saved training stats."""
    from src.data.preprocessing import OrbitPreprocessor
    proc = OrbitPreprocessor()
    processed = proc.preprocess(df, "iss")

    # Re-normalize with TRAINING stats (not freshly computed stats)
    for col in ["x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse"]:
        mean = training_stats["iss"]["mean"][col]
        std = training_stats["iss"]["std"][col]
        processed[f"{col}_norm"] = (processed[col] - mean) / std if std > 0 else 0.0

    return processed


def load_and_run_lstm(input_data):
    """Detect architecture from checkpoint, load model, run inference."""
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    layer_keys = [k for k in state if k.startswith("lstm.weight_ih_l")]
    num_layers = len(layer_keys) // 2
    fc_keys = sorted([k for k in state if k.startswith("fc.") and k.endswith(".weight")])
    horizon = state[fc_keys[-1]].shape[0] // 3

    log.info(f"Model: hidden={hidden_dim}, layers={num_layers}, horizon={horizon}")

    from scripts.train_gpu import OrbitLSTMDirect
    model = OrbitLSTMDirect(
        input_dim=input_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, horizon=horizon,
        output_dim=3, dropout=0.0
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(input_data).float().unsqueeze(0)
        pred = model(x).squeeze(0).numpy()
    return pred, horizon


def to_lat_lon_alt(pred_norm, training_stats):
    """Denormalize GSE, convert to lat/lon/alt via astropy."""
    from src.utils.coords import gse_to_geo

    pred_km = np.zeros_like(pred_norm)
    for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
        pred_km[:, i] = (pred_norm[:, i] * training_stats["iss"]["std"][col]
                         + training_stats["iss"]["mean"][col])

    now = datetime.now(timezone.utc)
    times = np.array([np.datetime64(now + timedelta(minutes=t), "ns") for t in range(len(pred_km))])

    try:
        x_geo, y_geo, z_geo = gse_to_geo(pred_km[:, 0], pred_km[:, 1], pred_km[:, 2], times)
    except Exception as e:
        log.error(f"GSE-to-GEO failed: {e}")
        return []

    r_earth = 6371.0
    r = np.sqrt(x_geo**2 + y_geo**2 + z_geo**2)
    lat = np.degrees(np.arcsin(np.clip(z_geo / r, -1, 1)))
    lon = np.degrees(np.arctan2(y_geo, x_geo))
    alt = r - r_earth

    step = max(1, len(pred_km) // 30)
    path = []
    for t in range(0, len(pred_km), step):
        if not (np.isfinite(lat[t]) and np.isfinite(lon[t]) and np.isfinite(alt[t])):
            continue
        if alt[t] < 300 or alt[t] > 500:
            continue
        path.append({
            "lat": round(float(lat[t]), 2),
            "lng": round(float(lon[t]), 2),
            "alt": round(float(alt[t]), 0),
            "minutes_ahead": int(t),
        })
    return path


def main():
    log.info("=== Hourly prediction start ===")

    if not STATS_FILE.exists():
        log.error(f"Stats not found: {STATS_FILE}")
        return 1
    if not CHECKPOINT.exists():
        log.error(f"Checkpoint not found: {CHECKPOINT}")
        return 1

    with open(STATS_FILE) as f:
        training_stats = json.load(f)

    try:
        df = fetch_recent_iss()
    except Exception as e:
        log.error(f"Fetch failed: {e}")
        return 1

    try:
        processed = preprocess_with_training_stats(df, training_stats)
    except Exception as e:
        log.error(f"Preprocess failed: {e}")
        return 1

    norm_cols = sorted([c for c in processed.columns if c.endswith("_norm")])
    data = processed[norm_cols].values
    if len(data) < 1440:
        log.error(f"Insufficient data: {len(data)} rows, need 1440")
        return 1

    try:
        pred_norm, horizon = load_and_run_lstm(data[-1440:])
        log.info(f"Inference: {pred_norm.shape}")
    except Exception as e:
        log.error(f"Inference failed: {e}")
        return 1

    path = to_lat_lon_alt(pred_norm, training_stats)
    if len(path) < 5:
        log.error(f"Too few points: {len(path)}")
        return 1

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "lstm_iss_6h",
        "points": len(path),
        "path": path,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Wrote {len(path)} points to {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
