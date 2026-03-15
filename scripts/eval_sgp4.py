#!/usr/bin/env python3
"""Evaluate Keplerian two-body baseline on test sets.

Usage:
    python scripts/eval_sgp4.py --spacecraft iss
    python scripts/eval_sgp4.py --spacecraft all
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.baseline_sgp4 import SGP4Baseline
from src.data.preprocessing import OrbitPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sgp4-baseline")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def compute_kepler_mae(spacecraft_id, config):
    """Compute Keplerian two-body baseline MAE on test set."""
    log.info(f"Computing Keplerian baseline for {spacecraft_id}")

    proc = OrbitPreprocessor()
    raw_path = Path(f"data/raw/{spacecraft_id}_2023-01-01_2025-12-31.parquet")
    if not raw_path.exists():
        log.error(f"Data not found: {raw_path}")
        return float("nan")

    df = pd.read_parquet(raw_path)
    processed = proc.preprocess(df, spacecraft_id)
    stats = proc.stats

    # Save stats
    stats_dir = RESULTS_DIR / "norm_stats"
    stats_dir.mkdir(exist_ok=True)
    with open(stats_dir / f"{spacecraft_id}_norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (config["model"]["input_hours"] * 60) // time_res
    horizon_steps = (6 * 60) // time_res
    stride_steps = horizon_steps

    pos_cols = ["x_gse", "y_gse", "z_gse"]
    vel_cols = ["vx_gse", "vy_gse", "vz_gse"]
    all_cols = pos_cols + vel_cols

    inputs_raw, targets_raw = [], []
    for _, seg in processed.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[all_cols].values
        tgts = seg[pos_cols].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
            inputs_raw.append(feats[i:i + input_steps])
            targets_raw.append(tgts[i + input_steps:i + input_steps + horizon_steps])

    inputs_raw = np.array(inputs_raw, dtype=np.float64)
    targets_raw = np.array(targets_raw, dtype=np.float64)

    # Test split (last 15%)
    test_start = int(0.85 * len(inputs_raw))
    test_inputs = inputs_raw[test_start:]
    test_targets = targets_raw[test_start:]
    log.info(f"Test set: {len(test_inputs)} windows")

    dt_seconds = time_res * 60.0
    all_distances = []

    for i in range(len(test_inputs)):
        last_pos = test_inputs[i, -1, :3]
        last_vel = test_inputs[i, -1, 3:6]
        pred = SGP4Baseline.simple_kepler_propagate(last_pos, last_vel, dt_seconds, horizon_steps)
        distances = np.sqrt(np.sum((pred - test_targets[i])**2, axis=-1))
        all_distances.append(distances)

    if not all_distances:
        log.warning(f"{spacecraft_id}: no valid test windows (data may be too sparse)")
        return float("nan")

    all_distances = np.concatenate(all_distances)
    mae = float(np.mean(all_distances))
    rmse = float(np.sqrt(np.mean(all_distances**2)))
    log.info(f"{spacecraft_id} Keplerian: MAE={mae:.1f} km, RMSE={rmse:.1f} km")
    return mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacecraft", default="all")
    args = parser.parse_args()

    config = load_config()
    spacecraft_list = (
        list(config["spacecraft"].keys()) if args.spacecraft == "all"
        else [args.spacecraft]
    )

    results = {}
    results["keplerian"] = {}
    for sc in spacecraft_list:
        results["keplerian"][sc] = compute_kepler_mae(sc, config)

    out_path = RESULTS_DIR / "sgp4_baselines.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
