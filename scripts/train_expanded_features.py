#!/usr/bin/env python3
"""Train multi-modal ISS model with expanded solar wind features + ablation.

Tasks:
1. Train full 13-feature model
2. Leave-one-out ablation (12 runs, dropping one feature group each)
3. Re-run storm-conditioned at Kp >= 4, 5, 6

Usage:
    python scripts/train_expanded_features.py
    python scripts/train_expanded_features.py --skip-ablation

Outputs:
    checkpoints/multimodal_iss_6h_13feat_best.pt
    results/ablation_results.csv
    results/storm_eval_expanded.csv
    results/expanded_feature_summary.md
"""
import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("expanded-features")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42

# Feature groups for ablation
# Derived features ablated as single units even if decomposed
FEATURE_GROUPS = {
    "bx_gse": ["bx_gse_norm"],
    "by_gse": ["by_gse_norm"],
    "bz_gse": ["bz_gse_norm"],
    "flow_speed": ["flow_speed_norm"],
    "proton_density": ["proton_density_norm"],
    "kp": ["kp_norm"],
    "dst": ["dst_norm"],
    "ae": ["ae_norm"],
    "al": ["al_norm"],
    "au": ["au_norm"],
    "clock_angle": ["clock_angle_sin_norm", "clock_angle_cos_norm"],
    "dynamic_pressure": ["dynamic_pressure_norm"],
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data():
    """Load orbit + expanded solar wind, create multimodal windows."""
    from scripts.train_gpu import (
        load_spacecraft_data, load_solar_wind_data,
        preprocess_orbit, preprocess_solar_wind,
        create_multimodal_windows,
    )

    log.info("Loading orbit data...")
    orbit_df = load_spacecraft_data("iss")
    orbit_processed, orbit_stats = preprocess_orbit(orbit_df, "iss")
    log.info(f"Orbit: {len(orbit_processed)} rows")

    log.info("Loading solar wind data...")
    sw_df = load_solar_wind_data()
    sw_processed, sw_stats = preprocess_solar_wind(sw_df)
    log.info(f"Solar wind: {len(sw_processed)} rows")

    sw_norm_cols = sorted([c for c in sw_processed.columns if c.endswith("_norm")])
    log.info(f"Solar wind features ({len(sw_norm_cols)}): {sw_norm_cols}")

    log.info("Creating multimodal windows...")
    o_wins, s_wins, t_wins = create_multimodal_windows(
        orbit_processed, sw_processed, 1440, 360, 360, subsample=1
    )
    log.info(f"Windows: {len(o_wins)} | orbit: {o_wins.shape} | sw: {s_wins.shape}")

    return o_wins, s_wins, t_wins, orbit_stats, sw_stats, sw_norm_cols, sw_df


def split_data(o_wins, s_wins, t_wins):
    n = len(o_wins)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    return {
        "train": (o_wins[:n_train], s_wins[:n_train], t_wins[:n_train]),
        "val": (o_wins[n_train:n_train+n_val], s_wins[n_train:n_train+n_val], t_wins[n_train:n_train+n_val]),
        "test": (o_wins[n_train+n_val:], s_wins[n_train+n_val:], t_wins[n_train+n_val:]),
    }


def train_multimodal_model(orbit_train, sw_train, target_train,
                           orbit_val, sw_val, target_val,
                           solar_input_dim, name, epochs=100, patience=15):
    """Two-phase training matching paper Section 4.4."""
    from scripts.train_gpu import SolarWindOrbitModel

    model = SolarWindOrbitModel(
        orbit_input_dim=orbit_train.shape[-1],
        solar_input_dim=solar_input_dim,
        hidden_dim=128, num_layers=3, nhead=8,
        horizon=target_train.shape[1], output_dim=target_train.shape[-1], dropout=0.1,
    ).to(DEVICE)

    log.info(f"  {name}: {sum(p.numel() for p in model.parameters()):,} params, sw_dim={solar_input_dim}")

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(orbit_train), torch.from_numpy(sw_train), torch.from_numpy(target_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(orbit_val), torch.from_numpy(sw_val), torch.from_numpy(target_val))
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64)

    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None

    def run_epoch(dl, training=True):
        losses = []
        for o, s, t in dl:
            o, s, t = o.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
            if training:
                optimizer.zero_grad()
            pred = model(o, s)
            ml = min(pred.shape[1], t.shape[1])
            loss = criterion(pred[:, :ml], t[:, :ml])
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        return np.mean(losses)

    # Phase 1: freeze solar branch (20 epochs, LR=1e-3)
    model.freeze_solar_branch()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    for epoch in range(1, 21):
        model.train()
        run_epoch(train_dl, training=True)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            avg_v = run_epoch(val_dl, training=False)
        if avg_v < best_val:
            best_val = avg_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            log.info(f"    P1 {epoch}/20 val={avg_v:.6f}")

    if best_state:
        model.load_state_dict(best_state)

    # Phase 2: unfreeze all (80 epochs, LR=1e-4)
    model.unfreeze_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 20)
    patience_ctr = 0

    for epoch in range(1, epochs - 20 + 1):
        model.train()
        run_epoch(train_dl, training=True)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            avg_v = run_epoch(val_dl, training=False)
        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"    Early stop P2 epoch {epoch}")
                break
        if epoch % 20 == 0:
            log.info(f"    P2 {epoch}/{epochs-20} val={avg_v:.6f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def compute_mae(model, orbit_test, sw_test, target_test, orbit_stats):
    from scripts.train_gpu import denormalize
    model.eval()
    all_preds = []
    for b in range(0, len(orbit_test), 64):
        with torch.no_grad():
            o = torch.from_numpy(orbit_test[b:b+64]).float().to(DEVICE)
            s = torch.from_numpy(sw_test[b:b+64]).float().to(DEVICE)
            all_preds.append(model(o, s).cpu().numpy())
    preds = np.concatenate(all_preds)
    preds_km = denormalize(preds, orbit_stats)
    tgts_km = denormalize(target_test, orbit_stats)
    return float(np.mean(np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    log.info("=" * 60)
    log.info("EXPANDED FEATURE TRAINING + ABLATION")
    log.info(f"Device: {DEVICE}")
    log.info("=" * 60)

    o_wins, s_wins, t_wins, orbit_stats, sw_stats, sw_norm_cols, sw_df = load_and_prepare_data()
    splits = split_data(o_wins, s_wins, t_wins)
    full_feat_count = s_wins.shape[-1]

    # ── Train full model ──
    log.info(f"\nTRAINING FULL MODEL ({full_feat_count} features)")
    set_seed(SEED)
    model, best_val = train_multimodal_model(
        *splits["train"], *splits["val"][:2], splits["val"][2],
        solar_input_dim=full_feat_count,
        name=f"multimodal_iss_6h_{full_feat_count}feat",
    )
    ckpt_name = f"multimodal_iss_6h_{full_feat_count}feat_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "val_loss": best_val,
        "feature_count": full_feat_count, "feature_columns": sw_norm_cols,
    }, CHECKPOINT_DIR / ckpt_name)

    baseline_mae = compute_mae(model, *splits["test"], orbit_stats)
    log.info(f"Full model MAE: {baseline_mae:.1f} km")

    # ── Leave-one-out ablation ──
    ablation_results = []
    noise_features = []

    if not args.skip_ablation:
        log.info(f"\nLEAVE-ONE-OUT ABLATION ({len(FEATURE_GROUPS)} runs)")
        for group_name, group_cols in FEATURE_GROUPS.items():
            missing = [c for c in group_cols if c not in sw_norm_cols]
            if missing:
                log.warning(f"  Skip {group_name}: missing {missing}")
                continue

            drop_idx = [sw_norm_cols.index(c) for c in group_cols]
            keep_idx = [i for i in range(len(sw_norm_cols)) if i not in drop_idx]
            reduced_dim = len(keep_idx)

            log.info(f"\n  Without {group_name} ({reduced_dim} features)")
            set_seed(SEED)
            try:
                abl_model, _ = train_multimodal_model(
                    splits["train"][0], splits["train"][1][:, :, keep_idx], splits["train"][2],
                    splits["val"][0], splits["val"][1][:, :, keep_idx], splits["val"][2],
                    solar_input_dim=reduced_dim, name=f"without_{group_name}",
                )
                abl_mae = compute_mae(
                    abl_model, splits["test"][0], splits["test"][1][:, :, keep_idx],
                    splits["test"][2], orbit_stats
                )
                delta = abl_mae - baseline_mae
                keep = delta >= 0  # removal worsened MAE = feature helps
                if not keep:
                    noise_features.append(group_name)
                log.info(f"    MAE: {abl_mae:.1f} km (delta: {delta:+.1f}) {'KEEP' if keep else 'NOISE'}")
                ablation_results.append({
                    "feature": group_name, "mae_without": round(abl_mae, 1),
                    "delta_vs_baseline": round(delta, 1), "keep": keep,
                })
            except Exception as e:
                log.error(f"    FAILED: {e}")
                ablation_results.append({
                    "feature": group_name, "mae_without": None,
                    "delta_vs_baseline": None, "keep": None,
                })

        with open(RESULTS_DIR / "ablation_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["feature", "mae_without", "delta_vs_baseline", "keep"])
            w.writeheader()
            w.writerows(ablation_results)
        log.info(f"\nAblation saved to results/ablation_results.csv")
        if noise_features:
            log.info(f"Noise features: {noise_features}")

    # ── Storm evaluation ──
    log.info(f"\nSTORM EVALUATION (Kp >= 4, 5, 6)")
    from scripts.eval_storm_conditioned import assign_kp
    from scripts.train_gpu import denormalize, load_spacecraft_data, preprocess_orbit

    # Get test window timestamps
    orbit_df = load_spacecraft_data("iss")
    orbit_processed, _ = preprocess_orbit(orbit_df, "iss")
    time_diffs = orbit_processed["time"].diff().dt.total_seconds()
    orbit_processed["segment_id"] = (time_diffs > 600).cumsum()

    window_starts = []
    for _, seg in orbit_processed.groupby("segment_id"):
        if len(seg) < 1800:
            continue
        times = seg["time"].values
        for i in range(0, len(seg) - 1800, 360):
            window_starts.append(times[i + 1440])

    test_start = int(0.85 * len(window_starts))
    test_times = np.array(window_starts[test_start:])
    test_len = min(len(test_times), len(splits["test"][0]))
    test_times = test_times[:test_len]

    test_kp = assign_kp(test_times, sw_df)

    # Compute predictions
    model.eval()
    all_preds = []
    for b in range(0, test_len, 64):
        with torch.no_grad():
            o = torch.from_numpy(splits["test"][0][b:b+64]).float().to(DEVICE)
            s = torch.from_numpy(splits["test"][1][b:b+64]).float().to(DEVICE)
            all_preds.append(model(o, s).cpu().numpy())
    mm_preds = np.concatenate(all_preds)[:test_len]
    mm_preds_km = denormalize(mm_preds, orbit_stats)
    tgts_km = denormalize(splits["test"][2][:test_len], orbit_stats)
    mm_dist = np.sqrt(np.sum((mm_preds_km - tgts_km)**2, axis=-1))

    # Also LSTM
    lstm_dist = None
    lstm_ckpt = CHECKPOINT_DIR / "lstm_iss_6h_best.pt"
    if lstm_ckpt.exists():
        from scripts.eval_storm_conditioned import load_model
        lstm = load_model("lstm", lstm_ckpt, input_dim=o_wins.shape[-1])
        lstm.eval()
        lstm_preds_list = []
        for b in range(0, test_len, 64):
            with torch.no_grad():
                o = torch.from_numpy(splits["test"][0][b:b+64]).float().to(DEVICE)
                lstm_preds_list.append(lstm(o).cpu().numpy())
        lstm_preds = np.concatenate(lstm_preds_list)[:test_len]
        lstm_km = denormalize(lstm_preds, orbit_stats)
        lstm_dist = np.sqrt(np.sum((lstm_km - tgts_km)**2, axis=-1))

    storm_results = []
    for model_name, distances in [("multimodal_13feat", mm_dist), ("lstm", lstm_dist)]:
        if distances is None:
            continue
        for threshold in ["all", "quiet", 4, 5, 6]:
            if threshold == "all":
                mask = np.ones(len(test_kp), dtype=bool)
            elif threshold == "quiet":
                mask = test_kp <= 3
            else:
                mask = test_kp >= threshold
            n = int(mask.sum())
            mae = round(float(np.mean(distances[mask])), 1) if n > 0 else None
            underpowered = n < 20 if isinstance(threshold, int) else False
            flag = " [UNDERPOWERED]" if underpowered else ""
            log.info(f"  {model_name} Kp>={threshold}: n={n}, MAE={mae}{flag}")
            storm_results.append({
                "model": model_name, "kp_threshold": str(threshold),
                "n_samples": n, "mae_km": mae, "underpowered": underpowered,
            })

    with open(RESULTS_DIR / "storm_eval_expanded.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "kp_threshold", "n_samples", "mae_km", "underpowered"])
        w.writeheader()
        w.writerows(storm_results)

    # ── Summary ──
    elapsed = time.time() - start_time
    summary = f"""# Expanded Feature Summary

## Feature Set
- **Total features:** {full_feat_count}
- **F10.7 solar flux:** EXCLUDED (daily cadence only, leakage risk for live pipeline)
- **Features:** {', '.join(c.replace('_norm','') for c in sw_norm_cols)}

## Baseline MAE
- Full model ({full_feat_count} features): **{baseline_mae:.1f} km**

## Ablation
{"See results/ablation_results.csv" if args.skip_ablation else ""}
{("Noise features: " + ", ".join(noise_features)) if noise_features else "No features flagged as noise."}

## Storm Window Counts
See results/storm_eval_expanded.csv

## Runtime
{elapsed/60:.1f} minutes on {DEVICE}
"""
    with open(RESULTS_DIR / "expanded_feature_summary.md", "w") as f:
        f.write(summary)

    log.info(f"\nDone in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
