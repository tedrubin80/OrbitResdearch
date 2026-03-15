#!/usr/bin/env python3
"""Assess models under different geomagnetic conditions.

Usage:
    python scripts/eval_storm_conditioned.py --spacecraft iss
    python scripts/eval_storm_conditioned.py --spacecraft all
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
from src.data.preprocessing import OrbitPreprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("storm-assessment")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def load_model(model_type, checkpoint_path, input_dim=6, solar_dim=8):
    """Load model, auto-detecting architecture from checkpoint weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"]

    from scripts.train_gpu import OrbitLSTMDirect, OrbitTransformerDirect, SolarWindOrbitModel

    if model_type == "lstm":
        hidden_dim = state["lstm.weight_ih_l0"].shape[0] // 4
        layer_keys = [k for k in state if k.startswith("lstm.weight_ih_l")]
        num_layers = len(layer_keys) // 2
        fc_keys = sorted([k for k in state if k.startswith("fc.") and k.endswith(".weight")])
        horizon = state[fc_keys[-1]].shape[0] // 3
        model = OrbitLSTMDirect(input_dim=input_dim, hidden_dim=hidden_dim,
                                num_layers=num_layers, horizon=horizon, output_dim=3, dropout=0.0)

    elif model_type == "transformer":
        hidden_dim = state["input_proj.weight"].shape[0]
        layer_keys = [k for k in state if "layers." in k and "self_attn.in_proj_weight" in k]
        num_layers = len(layer_keys)
        # nhead must divide hidden_dim; train_gpu.py uses 8
        nhead = 8 if hidden_dim % 8 == 0 else 4
        fc_keys = sorted([k for k in state if k.startswith("head.") and k.endswith(".weight")])
        horizon = state[fc_keys[-1]].shape[0] // 3
        ff_dim = state["encoder.layers.0.linear1.weight"].shape[0]
        model = OrbitTransformerDirect(input_dim=input_dim, d_model=hidden_dim, nhead=nhead,
                                       num_layers=num_layers, dim_feedforward=ff_dim,
                                       horizon=horizon, output_dim=3, dropout=0.0)

    elif model_type == "multimodal":
        hidden_dim = state["orbit_enc.weight_ih_l0"].shape[0] // 4
        layer_keys = [k for k in state if k.startswith("orbit_enc.weight_ih_l")]
        num_layers = len(layer_keys) // 2
        fc_keys = sorted([k for k in state if k.startswith("base_head.") and k.endswith(".weight")])
        horizon = state[fc_keys[-1]].shape[0] // 3
        model = SolarWindOrbitModel(orbit_input_dim=input_dim, solar_input_dim=solar_dim,
                                     hidden_dim=hidden_dim, num_layers=num_layers, nhead=8,
                                     horizon=horizon, output_dim=3, dropout=0.0)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def assign_kp(window_times, solar_df):
    """Assign preceding Kp to each window start time."""
    kp_df = solar_df[["time", "kp"]].dropna(subset=["kp"]).copy()
    kp_df["time"] = pd.to_datetime(kp_df["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    kp_df = kp_df.sort_values("time").drop_duplicates("time")
    windows_df = pd.DataFrame({"time": pd.to_datetime(window_times).astype("datetime64[ns]")})
    windows_df["time"] = windows_df["time"].dt.tz_localize(None)
    windows_df = windows_df.sort_values("time")
    merged = pd.merge_asof(windows_df, kp_df, on="time", direction="backward")
    kp_vals = merged["kp"].values
    # OMNI stores Kp*10 (0-90 scale). Convert to standard 0-9 scale.
    if np.nanmax(kp_vals) > 9:
        kp_vals = kp_vals / 10.0
    return kp_vals


def run_spacecraft(spacecraft_id, config):
    log.info(f"=== {spacecraft_id} ===")

    proc = OrbitPreprocessor()
    df = pd.read_parquet(f"data/raw/{spacecraft_id}_2023-01-01_2025-12-31.parquet")
    processed = proc.preprocess(df, spacecraft_id)
    stats = proc.stats

    stats_dir = RESULTS_DIR / "norm_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    with open(stats_dir / f"{spacecraft_id}_norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

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

    log.info(f"Test: {len(test_inputs)} windows | quiet={np.sum(test_kp <= 3)}, "
             f"active={np.sum((test_kp >= 4) & (test_kp <= 5))}, storm={np.sum(test_kp >= 6)}")

    # Solar wind windows for multimodal
    test_solar = None
    try:
        from src.data.preprocessing import SolarWindPreprocessor
        sw_proc = SolarWindPreprocessor()
        solar_processed = sw_proc.preprocess(solar_df)
        aligned = sw_proc.align_with_positions(solar_processed, processed)
        solar_norm_cols = sorted([c for c in aligned.columns
                                  if c.endswith("_norm") and c.split("_norm")[0] in
                                  ["bx_gse", "by_gse", "bz_gse", "flow_speed",
                                   "proton_density", "kp", "dst", "ae"]])
        if solar_norm_cols:
            solar_inputs = []
            for _, seg in aligned.groupby("segment_id"):
                if len(seg) < input_steps + horizon_steps:
                    continue
                sw_feats = seg[solar_norm_cols].values
                for i in range(0, len(seg) - input_steps - horizon_steps, stride_steps):
                    solar_inputs.append(sw_feats[i:i + input_steps])
            if len(solar_inputs) == len(inputs):
                test_solar = np.array(solar_inputs, dtype=np.float32)[test_start:]
            else:
                log.warning(f"Solar mismatch: {len(solar_inputs)} vs {len(inputs)}")
    except Exception as e:
        log.warning(f"Solar prep failed: {e}")

    conditions = {
        "all": np.ones(len(test_kp), dtype=bool),
        "quiet": test_kp <= 3,
        "active": (test_kp >= 4) & (test_kp <= 5),
        "storm": test_kp >= 6,
    }

    model_results = {}
    for model_type in ["lstm", "transformer", "multimodal"]:
        ckpt_path = CHECKPOINT_DIR / f"{model_type}_{spacecraft_id}_6h_best.pt"
        if not ckpt_path.exists():
            log.warning(f"No checkpoint: {ckpt_path}")
            continue

        log.info(f"  {model_type}")
        try:
            model = load_model(model_type, ckpt_path, input_dim=inputs.shape[-1])
        except Exception as e:
            log.error(f"  Load failed: {e}")
            model_results[model_type] = {c: None for c in conditions}
            continue

        cond_results = {}
        for cond_name, mask in conditions.items():
            n = mask.sum()
            if n == 0:
                cond_results[cond_name] = None
                continue

            batch_size = 64
            all_preds = []
            masked_inputs = test_inputs[mask]
            masked_solar = test_solar[mask] if test_solar is not None else None

            for b in range(0, n, batch_size):
                with torch.no_grad():
                    x = torch.from_numpy(masked_inputs[b:b+batch_size]).float().to(DEVICE)
                    if model_type == "multimodal":
                        if masked_solar is not None:
                            sw = torch.from_numpy(masked_solar[b:b+batch_size]).float().to(DEVICE)
                        else:
                            sw = torch.zeros(x.shape[0], input_steps, 8).to(DEVICE)
                        p = model(x, sw)
                    else:
                        p = model(x)
                    all_preds.append(p.cpu().numpy())

            preds = np.concatenate(all_preds, axis=0)
            tgts = test_targets[mask]

            preds_km = np.zeros_like(preds)
            tgts_km = np.zeros_like(tgts)
            for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
                std = stats[spacecraft_id]["std"][col]
                mean = stats[spacecraft_id]["mean"][col]
                preds_km[..., i] = preds[..., i] * std + mean
                tgts_km[..., i] = tgts[..., i] * std + mean

            distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
            valid = np.isfinite(distances)
            if valid.any():
                mae = round(float(np.nanmean(distances[valid])), 1)
            else:
                mae = None
            cond_results[cond_name] = mae
            log.info(f"    {cond_name} (n={n}): {mae:.1f} km")

        model_results[model_type] = cond_results
    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacecraft", default="all")
    args = parser.parse_args()

    config = load_config()
    spacecraft_list = (
        list(config["spacecraft"].keys()) if args.spacecraft == "all"
        else [args.spacecraft]
    )

    all_results = {}
    for sc in spacecraft_list:
        sc_results = run_spacecraft(sc, config)
        for mt, cr in sc_results.items():
            if mt not in all_results:
                all_results[mt] = {}
            all_results[mt][sc] = cr

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "storm_conditioned_mae.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
