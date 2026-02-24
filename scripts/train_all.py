#!/usr/bin/env python3
"""Run full training pipeline: LSTM, Transformer, Multi-modal on ISS data.

Designed to run unattended in tmux. Logs to logs/training_*.log.

Usage:
    python scripts/train_all.py                  # Full pipeline (10-min res for CPU)
    python scripts/train_all.py --model lstm     # Single model
    python scripts/train_all.py --model transformer
    python scripts/train_all.py --model multimodal
    python scripts/train_all.py --subsample 1    # Full 1-min resolution (GPU only)
"""

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import yaml

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orbit-train")

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_config():
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def prepare_data(spacecraft="iss", horizon_hours=6, subsample=10):
    """Load and preprocess data with subsampling for CPU training."""
    from src.data.preprocessing import OrbitPreprocessor

    config = load_config()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sc_config = config["spacecraft"][spacecraft]
    parquet_file = raw_dir / f"{spacecraft}_{sc_config['start_date']}_{sc_config['end_date']}.parquet"

    if not parquet_file.exists():
        log.error(f"Data file not found: {parquet_file}")
        return None

    log.info(f"Loading {parquet_file}")
    df = pd.read_parquet(parquet_file)
    log.info(f"  Raw: {len(df)} rows")

    preprocessor = OrbitPreprocessor()
    df_processed = preprocessor.preprocess(df, spacecraft)
    log.info(f"  Processed: {len(df_processed)} rows")

    inputs, targets, timestamps = preprocessor.create_sliding_windows(
        df_processed,
        input_hours=24,
        horizon_hours=horizon_hours,
        stride_hours=6,
        subsample=subsample,
    )
    log.info(f"  Windows: {len(inputs)} | input: {inputs.shape} | target: {targets.shape} | subsample: {subsample}x ({subsample}-min res)")

    if len(inputs) < 10:
        log.error("Not enough windows")
        return None

    splits = preprocessor.temporal_split(inputs, targets, timestamps)
    log.info(f"  Train: {len(splits['train'][0])} | Val: {len(splits['val'][0])} | Test: {len(splits['test'][0])}")

    return splits, preprocessor, config, spacecraft


def train_loop(model, train_loader, val_loader, name, epochs=50, patience=10):
    """Training loop with early stopping and checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training {name} on {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            min_len = min(pred.shape[1], y.shape[1])
            min_feat = min(pred.shape[2], y.shape[2])
            loss = criterion(pred[:, :min_len, :min_feat], y[:, :min_len, :min_feat])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                min_len = min(pred.shape[1], y.shape[1])
                min_feat = min(pred.shape[2], y.shape[2])
                val_losses.append(criterion(pred[:, :min_len, :min_feat], y[:, :min_len, :min_feat]).item())

        avg_val = np.mean(val_losses)
        history["val_loss"].append(avg_val)
        log.info(f"  Epoch {epoch:3d}/{epochs} | train={avg_train:.6f} | val={avg_val:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "history": history,
            }, CHECKPOINT_DIR / f"{name}_best.pt")
            log.info(f"    -> Best checkpoint saved (val={best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    return model, history, best_val_loss


def evaluate(model, test_loader, preprocessor, spacecraft, name):
    """Evaluate on test set with denormalization to km."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device))
            min_len = min(pred.shape[1], y.shape[1])
            min_feat = min(pred.shape[2], y.shape[2])
            all_preds.append(pred[:, :min_len, :min_feat].cpu().numpy())
            all_targets.append(y[:, :min_len, :min_feat].numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Denormalize to physical units
    preds_km = preprocessor.denormalize(preds, spacecraft)
    targets_km = preprocessor.denormalize(targets, spacecraft)

    distances = np.sqrt(np.sum((preds_km - targets_km)**2, axis=-1))
    mae = np.mean(distances)
    rmse = np.sqrt(np.mean(distances**2))

    log.info(f"\n{'='*60}")
    log.info(f"EVAL: {name}")
    log.info(f"{'='*60}")
    log.info(f"  MAE:  {mae:.2f} km | RMSE: {rmse:.2f} km | N={len(preds)}")

    n_steps = distances.shape[1]
    for frac, label in [(0.25, "1.5h"), (0.5, "3h"), (1.0, "6h")]:
        idx = min(int(frac * n_steps) - 1, n_steps - 1)
        log.info(f"  @{label}: MAE={np.mean(distances[:, idx]):.2f} km, RMSE={np.sqrt(np.mean(distances[:, idx]**2)):.2f} km")

    return {"mae": mae, "rmse": rmse, "n_samples": len(preds)}


def run_model(model_type, splits, preprocessor, config, spacecraft):
    """Train and evaluate a single model."""
    from src.data.dataset import OrbitDataset

    train_in, train_tgt = splits["train"]
    val_in, val_tgt = splits["val"]
    test_in, test_tgt = splits["test"]

    input_dim = train_in.shape[-1]
    output_dim = train_tgt.shape[-1]
    horizon = train_tgt.shape[1]

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(OrbitDataset(train_in, train_tgt), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(OrbitDataset(val_in, val_tgt), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(OrbitDataset(test_in, test_tgt), batch_size=batch_size)

    if model_type == "lstm":
        from src.models.lstm import OrbitLSTMDirect
        model = OrbitLSTMDirect(input_dim=input_dim, hidden_dim=64, num_layers=2,
                                horizon=horizon, output_dim=output_dim, dropout=0.1)
    elif model_type == "transformer":
        from src.models.transformer import OrbitTransformerDirect
        model = OrbitTransformerDirect(input_dim=input_dim, d_model=64, nhead=4,
                                       num_layers=2, dim_feedforward=128,
                                       horizon=horizon, output_dim=output_dim, dropout=0.1)

    log.info(f"{model_type.upper()}: input_dim={input_dim}, output_dim={output_dim}, horizon={horizon}")

    model, history, best_val = train_loop(model, train_loader, val_loader,
                                          name=f"{model_type}_{spacecraft}_6h",
                                          epochs=50, patience=10)
    return evaluate(model, test_loader, preprocessor, spacecraft, model_type.upper())


def run_multimodal(spacecraft="iss", horizon_hours=6, subsample=10):
    """Train multi-modal (orbit + solar wind) model."""
    from src.data.preprocessing import OrbitPreprocessor, SolarWindPreprocessor
    from src.models.multimodal import SolarWindOrbitModel

    config = load_config()
    raw_dir = PROJECT_ROOT / "data" / "raw"
    sc_config = config["spacecraft"][spacecraft]

    orbit_file = raw_dir / f"{spacecraft}_{sc_config['start_date']}_{sc_config['end_date']}.parquet"
    sw_file = raw_dir / f"solar_wind_{sc_config['start_date']}_{sc_config['end_date']}.parquet"

    if not orbit_file.exists() or not sw_file.exists():
        log.error("Missing data files")
        return None

    orbit_df = pd.read_parquet(orbit_file)
    sw_df = pd.read_parquet(sw_file)
    log.info(f"Orbit: {len(orbit_df)} rows | Solar wind: {len(sw_df)} rows")

    # Preprocess
    orbit_prep = OrbitPreprocessor()
    orbit_processed = orbit_prep.preprocess(orbit_df, spacecraft)

    sw_prep = SolarWindPreprocessor()
    sw_processed = sw_prep.preprocess(sw_df)

    # Align solar wind to orbit
    merged = sw_prep.align_with_positions(sw_processed, orbit_processed)
    log.info(f"Merged: {len(merged)} rows")

    # Feature columns
    orbit_norm_cols = sorted([c for c in merged.columns if c.endswith("_norm")
                              and any(c.startswith(p) for p in ["x_gse", "y_gse", "z_gse", "vx", "vy", "vz"])])
    sw_norm_cols = sorted([c for c in merged.columns if c.endswith("_norm")
                           and not any(c.startswith(p) for p in ["x_gse", "y_gse", "z_gse", "vx", "vy", "vz"])])
    target_cols = ["x_gse_norm", "y_gse_norm", "z_gse_norm"]
    target_cols = [c for c in target_cols if c in merged.columns]

    log.info(f"Orbit features ({len(orbit_norm_cols)}): {orbit_norm_cols}")
    log.info(f"SW features ({len(sw_norm_cols)}): {sw_norm_cols}")

    if not sw_norm_cols or not target_cols:
        log.error("Missing columns after merge")
        return None

    # Clean NaN rows
    all_cols = orbit_norm_cols + sw_norm_cols + target_cols
    clean = merged.dropna(subset=all_cols).reset_index(drop=True)
    log.info(f"Clean: {len(clean)} rows (dropped {len(merged) - len(clean)})")

    # Build windows with subsampling
    time_res = config["model"]["time_resolution_minutes"]
    input_steps = (24 * 60) // time_res
    horizon_steps = (horizon_hours * 60) // time_res
    stride = (6 * 60) // time_res

    orbit_vals = clean[orbit_norm_cols].values
    sw_vals = clean[sw_norm_cols].values
    tgt_vals = clean[target_cols].values

    orbit_wins, sw_wins, tgt_wins = [], [], []
    total = input_steps + horizon_steps

    for i in range(0, len(orbit_vals) - total, stride):
        o_in = orbit_vals[i:i + input_steps:subsample]
        s_in = sw_vals[i:i + input_steps:subsample]
        t_out = tgt_vals[i + input_steps:i + total:subsample]
        orbit_wins.append(o_in)
        sw_wins.append(s_in)
        tgt_wins.append(t_out)

    if len(orbit_wins) < 10:
        log.error(f"Only {len(orbit_wins)} windows")
        return None

    orbit_wins = np.array(orbit_wins, dtype=np.float32)
    sw_wins = np.array(sw_wins, dtype=np.float32)
    tgt_wins = np.array(tgt_wins, dtype=np.float32)

    log.info(f"Windows: {len(orbit_wins)} | orbit: {orbit_wins.shape} | sw: {sw_wins.shape} | tgt: {tgt_wins.shape}")

    # Split
    n = len(orbit_wins)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    class MM(torch.utils.data.Dataset):
        def __init__(self, o, s, t):
            self.o, self.s, self.t = torch.from_numpy(o), torch.from_numpy(s), torch.from_numpy(t)
        def __len__(self): return len(self.o)
        def __getitem__(self, i): return self.o[i], self.s[i], self.t[i]

    bs = 32
    train_dl = torch.utils.data.DataLoader(MM(orbit_wins[:n_train], sw_wins[:n_train], tgt_wins[:n_train]), batch_size=bs, shuffle=True)
    val_dl = torch.utils.data.DataLoader(MM(orbit_wins[n_train:n_train+n_val], sw_wins[n_train:n_train+n_val], tgt_wins[n_train:n_train+n_val]), batch_size=bs)
    test_dl = torch.utils.data.DataLoader(MM(orbit_wins[n_train+n_val:], sw_wins[n_train+n_val:], tgt_wins[n_train+n_val:]), batch_size=bs)

    log.info(f"Split: train={n_train} | val={n_val} | test={n - n_train - n_val}")

    model = SolarWindOrbitModel(
        orbit_input_dim=orbit_wins.shape[-1],
        solar_input_dim=sw_wins.shape[-1],
        hidden_dim=64, num_layers=2, nhead=4,
        horizon=tgt_wins.shape[1],
        output_dim=tgt_wins.shape[-1],
        dropout=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Multi-modal on {device} | params: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = torch.nn.MSELoss()
    best_val = float("inf")
    patience_ctr = 0

    for epoch in range(1, 51):
        model.train()
        t_losses = []
        for o, s, t in train_dl:
            o, s, t = o.to(device), s.to(device), t.to(device)
            optimizer.zero_grad()
            pred = model(o, s)
            ml = min(pred.shape[1], t.shape[1])
            loss = criterion(pred[:, :ml], t[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        scheduler.step()

        model.eval()
        v_losses = []
        with torch.no_grad():
            for o, s, t in val_dl:
                pred = model(o.to(device), s.to(device))
                ml = min(pred.shape[1], t.shape[1])
                v_losses.append(criterion(pred[:, :ml], t.to(device)[:, :ml]).item())

        avg_t, avg_v = np.mean(t_losses), np.mean(v_losses)
        log.info(f"  Epoch {epoch:3d}/50 | train={avg_t:.6f} | val={avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / f"multimodal_{spacecraft}_6h_best.pt")
            log.info(f"    -> Best (val={best_val:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= 10:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    # Evaluate
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for o, s, t in test_dl:
            pred = model(o.to(device), s.to(device))
            ml = min(pred.shape[1], t.shape[1])
            all_p.append(pred[:, :ml].cpu().numpy())
            all_t.append(t[:, :ml].numpy())

    preds, tgts = np.concatenate(all_p), np.concatenate(all_t)

    # Denormalize
    preds_km = orbit_prep.denormalize(preds, spacecraft)
    tgts_km = orbit_prep.denormalize(tgts, spacecraft)
    distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
    mae, rmse = np.mean(distances), np.sqrt(np.mean(distances**2))

    log.info(f"\n{'='*60}")
    log.info(f"EVAL: Multi-Modal (Orbit + Solar Wind)")
    log.info(f"{'='*60}")
    log.info(f"  MAE: {mae:.2f} km | RMSE: {rmse:.2f} km | N={len(preds)}")

    return {"mae": mae, "rmse": rmse, "n_samples": len(preds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "multimodal"])
    parser.add_argument("--spacecraft", default="iss")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--subsample", type=int, default=10, help="Subsample factor (10=10-min res, 1=full)")
    args = parser.parse_args()

    start_time = time.time()
    log.info("=" * 60)
    log.info("ORBITAL CHAOS - TRAINING PIPELINE")
    log.info(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"Subsample: {args.subsample}x ({args.subsample}-min resolution)")
    log.info(f"Log: {LOG_FILE}")
    log.info("=" * 60)

    results = {}
    models = [args.model] if args.model else ["lstm", "transformer", "multimodal"]

    splits = None
    if "lstm" in models or "transformer" in models:
        log.info("\n--- Preparing data ---")
        data = prepare_data(args.spacecraft, args.horizon, args.subsample)
        if data is not None:
            splits, preprocessor, config, spacecraft = data

    if "lstm" in models and splits:
        log.info(f"\n{'='*60}\nTRAINING: LSTM\n{'='*60}")
        try:
            results["lstm"] = run_model("lstm", splits, preprocessor, config, spacecraft)
        except Exception as e:
            log.error(f"LSTM failed: {e}\n{traceback.format_exc()}")

    if "transformer" in models and splits:
        log.info(f"\n{'='*60}\nTRAINING: TRANSFORMER\n{'='*60}")
        try:
            results["transformer"] = run_model("transformer", splits, preprocessor, config, spacecraft)
        except Exception as e:
            log.error(f"Transformer failed: {e}\n{traceback.format_exc()}")

    if "multimodal" in models:
        log.info(f"\n{'='*60}\nTRAINING: MULTI-MODAL\n{'='*60}")
        try:
            results["multimodal"] = run_multimodal(args.spacecraft, args.horizon, args.subsample)
        except Exception as e:
            log.error(f"Multi-modal failed: {e}\n{traceback.format_exc()}")

    elapsed = time.time() - start_time
    log.info(f"\n{'='*60}")
    log.info(f"DONE in {elapsed/60:.1f} min")
    log.info("=" * 60)
    for name, r in results.items():
        if r:
            log.info(f"  {name:15s} | MAE={r['mae']:.2f} km | RMSE={r['rmse']:.2f} km")
    log.info(f"Checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
