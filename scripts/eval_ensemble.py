#!/usr/bin/env python3
"""Ensemble: average LSTM + Multi-modal predictions.

Usage:
    python scripts/eval_ensemble.py --spacecraft iss
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ensemble")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacecraft", default="iss")
    args = parser.parse_args()
    sc = args.spacecraft

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    from scripts.eval_storm_conditioned import load_model, assign_kp
    from src.data.preprocessing import OrbitPreprocessor

    proc = OrbitPreprocessor()
    df = pd.read_parquet(f"data/raw/{sc}_2023-01-01_2025-12-31.parquet")
    processed = proc.preprocess(df, sc)
    stats = proc.stats

    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (config["model"]["input_hours"] * 60) // time_res
    horizon_steps = (6 * 60) // time_res
    stride_steps = horizon_steps

    norm_feat_cols = sorted([c for c in processed.columns if c.endswith("_norm")])
    norm_tgt_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]

    inputs, targets, window_times = [], [], []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[norm_feat_cols].values
        tgts = seg[norm_tgt_cols].values
        times = seg["time"].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            inputs.append(feats[i:i + input_steps])
            targets.append(tgts[i + input_steps:i + input_steps + horizon_steps])
            window_times.append(times[i + input_steps])

    inputs = np.array(inputs, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    window_times = np.array(window_times)

    test_start = int(0.85 * len(inputs))
    test_inputs = inputs[test_start:]
    test_targets = targets[test_start:]
    test_times = window_times[test_start:]

    sw_path = Path("data/raw/solar_wind_2023-01-01_2025-12-31.parquet")
    solar_df = pd.read_parquet(sw_path)
    test_kp = assign_kp(test_times, solar_df)

    # Load both models
    lstm = load_model("lstm", CHECKPOINT_DIR / f"lstm_{sc}_6h_best.pt", input_dim=inputs.shape[-1])
    mm = load_model("multimodal", CHECKPOINT_DIR / f"multimodal_{sc}_6h_best.pt", input_dim=inputs.shape[-1])

    # Run predictions in batches
    batch_size = 64
    lstm_preds_list, mm_preds_list = [], []

    for b in range(0, len(test_inputs), batch_size):
        with torch.no_grad():
            x = torch.from_numpy(test_inputs[b:b+batch_size]).float().to(DEVICE)
            lstm_preds_list.append(lstm(x).cpu().numpy())
            sw_zeros = torch.zeros(x.shape[0], input_steps, 8).to(DEVICE)
            mm_preds_list.append(mm(x, sw_zeros).cpu().numpy())

    lstm_preds = np.concatenate(lstm_preds_list)
    mm_preds = np.concatenate(mm_preds_list)
    ensemble_preds = (lstm_preds + mm_preds) / 2.0

    conditions = {
        "all": np.ones(len(test_kp), dtype=bool),
        "quiet": test_kp <= 3,
        "active": (test_kp >= 4) & (test_kp <= 5),
        "storm": test_kp >= 6,
    }

    ensemble_results = {}
    for cond_name, mask in conditions.items():
        n = mask.sum()
        if n == 0:
            ensemble_results[cond_name] = None
            continue

        preds_km = np.zeros_like(ensemble_preds[mask])
        tgts_km = np.zeros_like(test_targets[mask])
        for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
            std = stats[sc]["std"][col]
            mean = stats[sc]["mean"][col]
            preds_km[..., i] = ensemble_preds[mask][..., i] * std + mean
            tgts_km[..., i] = test_targets[mask][..., i] * std + mean

        distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
        mae = round(float(np.mean(distances)), 1)
        ensemble_results[cond_name] = mae
        log.info(f"Ensemble {cond_name} (n={n}): {mae:.1f} km")

    # Merge into storm results
    storm_path = RESULTS_DIR / "storm_conditioned_mae.json"
    if storm_path.exists():
        with open(storm_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if "ensemble" not in all_results:
        all_results["ensemble"] = {}
    all_results["ensemble"][sc] = ensemble_results

    with open(storm_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {storm_path}")


if __name__ == "__main__":
    main()
