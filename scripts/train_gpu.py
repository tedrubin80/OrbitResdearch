#!/usr/bin/env python3
"""GPU training script for RunPod. Self-contained — loads data from HF.

Trains LSTM, Transformer, and Multi-modal on all 3 spacecraft at full 1-min resolution.
Pushes checkpoints to HF model repo when done.

Usage:
    python train_gpu.py                     # Full pipeline
    python train_gpu.py --model lstm        # Single model
    python train_gpu.py --spacecraft iss    # Single spacecraft
"""

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orbit-gpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

HF_DATASET = "datamatters24/orbital-chaos-nasa-ssc"
HF_MODEL_REPO = "datamatters24/orbital-chaos-predictor"

# ── Models ──────────────────────────────────────────────────────────────────

class OrbitLSTMDirect(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=3, horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h).view(-1, self.horizon, self.output_dim)


class OrbitTransformerDirect(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=8, num_layers=4, dim_feedforward=512,
                 horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon * output_dim),
        )

    def forward(self, x):
        src = self.input_proj(x)
        encoded = self.encoder(src)
        pooled = encoded.mean(dim=1)
        return self.head(pooled).view(-1, self.horizon, self.output_dim)


class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, context):
        attended, _ = self.attn(query, context, context)
        return self.norm(query + attended)


class SolarWindOrbitModel(nn.Module):
    """Residual gated multi-modal model: output = base_prediction + gate * perturbation.

    The orbit encoder produces a base prediction identical to the standalone LSTM.
    The solar wind branch produces a learned perturbation gated by a sigmoid,
    so the model can never be worse than LSTM (gate can learn ~0).

    Two-phase training:
        Phase 1: Freeze solar/perturbation/gate, train orbit encoder + base_head only.
        Phase 2: Unfreeze everything, fine-tune with lower LR.
    """

    def __init__(self, orbit_input_dim=6, solar_input_dim=8, hidden_dim=128, num_layers=3,
                 nhead=8, horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim

        # --- Orbit encoder (same as standalone LSTM) ---
        self.orbit_proj = nn.Linear(orbit_input_dim, hidden_dim)
        self.orbit_enc = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.orbit_norm = nn.LayerNorm(hidden_dim * 2)
        # Base prediction head: final hidden states -> trajectory (LSTM-equivalent)
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

        # --- Solar wind encoder ---
        self.solar_proj = nn.Linear(solar_input_dim, hidden_dim)
        self.solar_enc = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.solar_norm = nn.LayerNorm(hidden_dim * 2)

        # --- Cross-attention: orbit attends to solar wind ---
        self.cross_attn = CrossModalAttention(hidden_dim * 2, nhead, dropout)

        # --- Attention-weighted summary (learned, not mean pool) ---
        self.attn_weight = nn.Linear(hidden_dim * 2, 1)

        # --- Perturbation head: deeper MLP producing correction signal ---
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim),
        )

        # --- Gate: sigmoid controlling perturbation strength ---
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_dim),
            nn.Sigmoid(),
        )

    def forward(self, orbit_input, solar_input):
        # Encode orbit (single pass — get both sequence output and final hidden states)
        orbit_emb = self.orbit_proj(orbit_input)
        orbit_out, (h, _) = self.orbit_enc(orbit_emb)
        o = self.orbit_norm(orbit_out)

        # Base prediction from final hidden states (like standalone LSTM)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (batch, hidden*2)
        base = self.base_head(h_cat).view(-1, self.horizon, self.output_dim)

        # Encode solar wind
        s = self.solar_norm(self.solar_enc(self.solar_proj(solar_input))[0])

        # Cross-attention: orbit features attend to solar wind
        attended = self.cross_attn(o, s)  # (batch, seq, hidden*2)

        # Attention-weighted summary (not mean pool)
        attn_scores = torch.softmax(self.attn_weight(attended), dim=1)  # (batch, seq, 1)
        summary = (attended * attn_scores).sum(dim=1)  # (batch, hidden*2)

        # Perturbation: learned correction from solar wind context
        perturbation = self.perturbation_head(summary).view(-1, self.horizon, self.output_dim)

        # Gate: per-element sigmoid controlling correction strength
        gate = self.gate_net(h_cat).view(-1, self.horizon, self.output_dim)

        # Residual output: base + gated perturbation
        return base + gate * perturbation

    def freeze_solar_branch(self):
        """Phase 1: freeze solar wind encoder, cross-attention, perturbation, and gate."""
        for module in [self.solar_proj, self.solar_enc, self.solar_norm,
                       self.cross_attn, self.attn_weight, self.perturbation_head, self.gate_net]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        """Phase 2: unfreeze everything for fine-tuning."""
        for p in self.parameters():
            p.requires_grad = True


# ── Data ────────────────────────────────────────────────────────────────────

class OrbitDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.from_numpy(inputs)
        self.targets = torch.from_numpy(targets)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]


class MultiModalDataset(Dataset):
    def __init__(self, orbit, solar, targets):
        self.orbit = torch.from_numpy(orbit)
        self.solar = torch.from_numpy(solar)
        self.targets = torch.from_numpy(targets)
    def __len__(self): return len(self.orbit)
    def __getitem__(self, idx): return self.orbit[idx], self.solar[idx], self.targets[idx]


def load_spacecraft_data(spacecraft):
    """Load parquet from HF dataset. Tries root then data/ prefix."""
    from huggingface_hub import hf_hub_download
    start, end = "2023-01-01", "2025-12-31"
    fname = f"{spacecraft}_{start}_{end}.parquet"
    last_err = None
    for prefix in ["", "data/"]:
        try:
            path = hf_hub_download(repo_id=HF_DATASET, filename=f"{prefix}{fname}", repo_type="dataset")
            return pd.read_parquet(path)
        except Exception as e:
            log.warning(f"  Failed to load {prefix}{fname}: {e}")
            last_err = e
            continue
    raise FileNotFoundError(f"Could not find {fname} in HF dataset. Last error: {last_err}")


def load_solar_wind_data():
    """Load solar wind data, preferring local CDAWeb-fetched file (has expanded features).

    Priority:
    1. Local data/raw/ (may have AL, AU, clock_angle, dynamic_pressure from fresh CDAWeb fetch)
    2. HF dataset (has original 8 columns only)

    After loading, derives clock_angle_sin/cos and dynamic_pressure if missing.
    """
    fname = "solar_wind_2023-01-01_2025-12-31.parquet"

    # Try local first (may have expanded features from CDAWeb)
    local_path = Path(f"data/raw/{fname}")
    if local_path.exists():
        log.info(f"  Loading local solar wind: {local_path}")
        df = pd.read_parquet(local_path)
        df = _ensure_derived_features(df)
        log.info(f"  Solar wind columns: {sorted(df.columns.tolist())}")
        return df

    # Fall back to HF
    from huggingface_hub import hf_hub_download
    last_err = None
    for prefix in ["", "data/"]:
        try:
            path = hf_hub_download(repo_id=HF_DATASET, filename=f"{prefix}{fname}", repo_type="dataset")
            df = pd.read_parquet(path)
            df = _ensure_derived_features(df)
            log.info(f"  Solar wind columns: {sorted(df.columns.tolist())}")
            return df
        except Exception as e:
            log.warning(f"  Failed to load {prefix}{fname}: {e}")
            last_err = e
            continue
    raise FileNotFoundError(f"Could not find {fname}. Last error: {last_err}")


def _ensure_derived_features(df):
    """Add derived features if not already present in the DataFrame."""
    # Clock angle sin/cos (from IMF By, Bz)
    if "by_gse" in df.columns and "bz_gse" in df.columns:
        if "clock_angle_sin" not in df.columns:
            clock_angle = np.arctan2(df["by_gse"], df["bz_gse"])
            df["clock_angle_sin"] = np.sin(clock_angle)
            df["clock_angle_cos"] = np.cos(clock_angle)

    # Dynamic pressure (from density, speed)
    if "proton_density" in df.columns and "flow_speed" in df.columns:
        if "dynamic_pressure" not in df.columns:
            df["dynamic_pressure"] = 1.6726e-6 * df["proton_density"] * df["flow_speed"]**2

    return df


def preprocess_orbit(df, spacecraft_id):
    """Preprocess orbit data: derive velocity, normalize."""
    df = df.copy().sort_values("time").reset_index(drop=True)
    for col in ["x_gse", "y_gse", "z_gse"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dt = df["time"].diff().dt.total_seconds()
    for axis in ["x_gse", "y_gse", "z_gse"]:
        vel = axis.replace("x_", "vx_").replace("y_", "vy_").replace("z_", "vz_")
        df[vel] = df[axis].diff() / dt

    df = df.iloc[1:].dropna(subset=[c for c in df.columns if c != "time"]).reset_index(drop=True)
    # Gap threshold: 3x median resolution (handles DSCOVR 12-min, MMS 1-min, ISS 1-min)
    med_dt = df["time"].diff().dt.total_seconds().dropna().median()
    gap_threshold = max(med_dt * 3, 600)
    df["segment_id"] = (df["time"].diff().dt.total_seconds() > gap_threshold).cumsum()

    feature_cols = [c for c in df.columns
                    if c.startswith(("x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse"))
                    and not c.endswith("_norm")]

    stats = {"mean": df[feature_cols].mean().to_dict(), "std": df[feature_cols].std().to_dict()}

    for col in feature_cols:
        std = stats["std"][col]
        df[f"{col}_norm"] = (df[col] - stats["mean"][col]) / std if std > 0 else 0.0

    return df, stats


def preprocess_solar_wind(df):
    """Normalize solar wind parameters.

    Forward-fill strategy by variable type:
    - Native 1-min (bx_gse, by_gse, bz_gse, flow_speed, proton_density,
      clock_angle_sin, clock_angle_cos, dynamic_pressure): ffill limit=30 (30 min gaps)
    - Hourly/3-hourly indices (kp, dst, ae, al, au): ffill limit=180 (3h gaps)
      These are geophysical indices reported at coarser cadence — forward-fill
      is physically correct (NOT linear interpolation, which would imply a
      smooth ramp between e.g. Kp=2 and Kp=7 that doesn't exist).
    """
    df = df.copy().sort_values("time").reset_index(drop=True)
    param_cols = [c for c in df.columns if c != "time"]

    # Index variables: forward-fill with larger tolerance (up to 3h)
    index_cols = [c for c in ["kp", "dst", "ae", "al", "au"] if c in param_cols]
    # Native 1-min + derived columns: forward-fill with smaller tolerance
    native_cols = [c for c in param_cols if c not in index_cols]

    if native_cols:
        df[native_cols] = df[native_cols].ffill(limit=30)
    if index_cols:
        df[index_cols] = df[index_cols].ffill(limit=180)

    stats = {"mean": df[param_cols].mean().to_dict(), "std": df[param_cols].std().to_dict()}
    for col in param_cols:
        std = stats["std"].get(col, 0)
        mean = stats["mean"].get(col, 0)
        df[f"{col}_norm"] = (df[col] - mean) / std if std and std > 0 else 0.0
    return df, stats


def create_windows(df, input_steps=1440, horizon_steps=360, stride=360, subsample=1):
    """Create sliding windows."""
    feature_cols = sorted([c for c in df.columns if c.endswith("_norm")])
    target_cols = [c for c in ["x_gse_norm", "y_gse_norm", "z_gse_norm"] if c in df.columns]

    inputs, targets = [], []
    for _, seg in df.groupby("segment_id"):
        if len(seg) < input_steps + horizon_steps:
            continue
        feats = seg[feature_cols].values
        tgts = seg[target_cols].values
        for i in range(0, len(seg) - input_steps - horizon_steps, stride):
            inp = feats[i:i+input_steps:subsample]
            tgt = tgts[i+input_steps:i+input_steps+horizon_steps:subsample]
            inputs.append(inp)
            targets.append(tgt)

    return np.array(inputs, dtype=np.float32), np.array(targets, dtype=np.float32)


def denormalize(predictions, stats):
    """Convert normalized predictions back to km."""
    result = np.zeros_like(predictions)
    for i, col in enumerate(["x_gse", "y_gse", "z_gse"]):
        if col in stats["mean"]:
            result[..., i] = predictions[..., i] * stats["std"][col] + stats["mean"][col]
    return result


def create_multimodal_windows(orbit_df, sw_df, input_steps=1440, horizon_steps=360, stride=360, subsample=1):
    """Create paired orbit + solar wind windows."""
    # Align solar wind to orbit with L1 delay
    sw = sw_df.copy()
    sw["time"] = sw["time"] + pd.Timedelta(minutes=45)

    orbit_sorted = orbit_df.sort_values("time").copy()
    sw_sorted = sw.sort_values("time").copy()
    orbit_sorted["time"] = pd.to_datetime(orbit_sorted["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    sw_sorted["time"] = pd.to_datetime(sw_sorted["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")

    merged = pd.merge_asof(orbit_sorted, sw_sorted, on="time", tolerance=pd.Timedelta(minutes=5), direction="nearest")

    orbit_norm_cols = sorted([c for c in merged.columns if c.endswith("_norm")
                              and any(c.startswith(p) for p in ["x_gse", "y_gse", "z_gse", "vx", "vy", "vz"])])
    sw_norm_cols = sorted([c for c in merged.columns if c.endswith("_norm")
                           and not any(c.startswith(p) for p in ["x_gse", "y_gse", "z_gse", "vx", "vy", "vz"])])
    target_cols = [c for c in ["x_gse_norm", "y_gse_norm", "z_gse_norm"] if c in merged.columns]

    all_cols = orbit_norm_cols + sw_norm_cols + target_cols
    clean = merged.dropna(subset=all_cols).reset_index(drop=True)
    log.info(f"  Multimodal merged: {len(clean)} clean rows, orbit_feats={len(orbit_norm_cols)}, sw_feats={len(sw_norm_cols)}")

    # Detect resolution and adjust window sizes
    time_diffs = clean["time"].diff().dt.total_seconds().dropna()
    median_res_min = max(int(np.median(time_diffs) / 60), 1)
    input_steps = (24 * 60) // median_res_min
    # Recalculate horizon based on detected resolution (6h default)
    horizon_steps = (6 * 60) // median_res_min
    stride = max(horizon_steps, 1)
    log.info(f"  Multimodal resolution: {median_res_min}-min, input={input_steps}, horizon={horizon_steps}")

    med_dt = clean["time"].diff().dt.total_seconds().dropna().median()
    gap_threshold = max(med_dt * 3, 600)
    clean["segment_id"] = (clean["time"].diff().dt.total_seconds() > gap_threshold).cumsum()

    o_wins, s_wins, t_wins = [], [], []
    total = input_steps + horizon_steps

    for _, seg in clean.groupby("segment_id"):
        if len(seg) < total:
            continue
        o_data = seg[orbit_norm_cols].values
        s_data = seg[sw_norm_cols].values
        t_data = seg[target_cols].values
        for i in range(0, len(seg) - total, stride):
            o_wins.append(o_data[i:i+input_steps:subsample])
            s_wins.append(s_data[i:i+input_steps:subsample])
            t_wins.append(t_data[i+input_steps:i+total:subsample])

    return (np.array(o_wins, dtype=np.float32),
            np.array(s_wins, dtype=np.float32),
            np.array(t_wins, dtype=np.float32))


# ── Training ────────────────────────────────────────────────────────────────

def train_single(model, train_loader, val_loader, name, epochs=100, patience=15):
    """Train a single-input model."""
    log.info(f"Training {name} | params: {sum(p.numel() for p in model.parameters()):,} | device: {DEVICE}")
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val, patience_ctr = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = []
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            ml = min(pred.shape[1], y.shape[1])
            loss = criterion(pred[:, :ml], y[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss.append(loss.item())

        scheduler.step()
        model.eval()
        v_loss = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                ml = min(pred.shape[1], y.shape[1])
                v_loss.append(criterion(pred[:, :ml], y[:, :ml]).item())

        avg_t, avg_v = np.mean(t_loss), np.mean(v_loss)
        log.info(f"  Epoch {epoch:3d}/{epochs} | train={avg_t:.6f} | val={avg_v:.6f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / f"{name}_best.pt")
            log.info(f"    -> Best ({best_val:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    return model, best_val


def train_multimodal(model, train_loader, val_loader, name, epochs=100, patience=15):
    """Train a multi-modal model with two-phase training.

    Phase 1 (20 epochs): Freeze solar/perturbation/gate, train orbit encoder + base_head.
    Phase 2 (remaining epochs): Unfreeze all, fine-tune with lower LR.
    """
    log.info(f"Training {name} | params: {sum(p.numel() for p in model.parameters()):,} | device: {DEVICE}")
    model = model.to(DEVICE)
    criterion = nn.MSELoss()

    phase1_epochs = 20
    phase2_epochs = epochs - phase1_epochs

    # ── Phase 1: Train orbit encoder only (LSTM-equivalent baseline) ──
    log.info(f"  Phase 1: Training orbit encoder only ({phase1_epochs} epochs)")
    model.freeze_solar_branch()

    trainable = [p for p in model.parameters() if p.requires_grad]
    log.info(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    best_val, patience_ctr = float("inf"), 0

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        t_loss = []
        for o, s, t in train_loader:
            o, s, t = o.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            pred = model(o, s)
            ml = min(pred.shape[1], t.shape[1])
            loss = criterion(pred[:, :ml], t[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss.append(loss.item())

        scheduler.step()
        model.eval()
        v_loss = []
        with torch.no_grad():
            for o, s, t in val_loader:
                o, s, t = o.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
                pred = model(o, s)
                ml = min(pred.shape[1], t.shape[1])
                v_loss.append(criterion(pred[:, :ml], t[:, :ml]).item())

        avg_t, avg_v = np.mean(t_loss), np.mean(v_loss)
        log.info(f"  P1 Epoch {epoch:3d}/{phase1_epochs} | train={avg_t:.6f} | val={avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": epoch, "phase": 1, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / f"{name}_best.pt")
            log.info(f"    -> Best ({best_val:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"  Phase 1 early stopping at epoch {epoch}")
                break

    log.info(f"  Phase 1 done. Best val: {best_val:.6f}")

    # ── Phase 2: Unfreeze everything, fine-tune with lower LR ──
    log.info(f"  Phase 2: Fine-tuning all parameters ({phase2_epochs} epochs)")
    model.unfreeze_all()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs)
    patience_ctr = 0

    for epoch in range(1, phase2_epochs + 1):
        model.train()
        t_loss = []
        for o, s, t in train_loader:
            o, s, t = o.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            pred = model(o, s)
            ml = min(pred.shape[1], t.shape[1])
            loss = criterion(pred[:, :ml], t[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss.append(loss.item())

        scheduler.step()
        model.eval()
        v_loss = []
        with torch.no_grad():
            for o, s, t in val_loader:
                o, s, t = o.to(DEVICE), s.to(DEVICE), t.to(DEVICE)
                pred = model(o, s)
                ml = min(pred.shape[1], t.shape[1])
                v_loss.append(criterion(pred[:, :ml], t[:, :ml]).item())

        avg_t, avg_v = np.mean(t_loss), np.mean(v_loss)
        log.info(f"  P2 Epoch {epoch:3d}/{phase2_epochs} | train={avg_t:.6f} | val={avg_v:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": phase1_epochs + epoch, "phase": 2, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / f"{name}_best.pt")
            log.info(f"    -> Best ({best_val:.6f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"  Phase 2 early stopping at epoch {epoch}")
                break

    return model, best_val


def evaluate(model, test_loader, stats, name, multimodal=False):
    """Evaluate on test set with denormalization to km."""
    model = model.to(DEVICE).eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            if multimodal:
                o, s, t = batch
                pred = model(o.to(DEVICE), s.to(DEVICE))
            else:
                x, t = batch
                pred = model(x.to(DEVICE))
            ml = min(pred.shape[1], t.shape[1])
            all_p.append(pred[:, :ml].cpu().numpy())
            all_t.append(t[:, :ml].numpy())

    preds, tgts = np.concatenate(all_p), np.concatenate(all_t)
    preds_km = denormalize(preds, stats)
    tgts_km = denormalize(tgts, stats)
    distances = np.sqrt(np.sum((preds_km - tgts_km)**2, axis=-1))
    mae, rmse = np.mean(distances), np.sqrt(np.mean(distances**2))

    log.info(f"\n{'='*60}")
    log.info(f"EVAL: {name}")
    log.info(f"  MAE: {mae:.2f} km | RMSE: {rmse:.2f} km | N={len(preds)}")
    n = distances.shape[1]
    for frac, label in [(0.25, "1.5h"), (0.5, "3h"), (1.0, "6h")]:
        idx = min(int(frac * n) - 1, n - 1)
        log.info(f"  @{label}: MAE={np.mean(distances[:, idx]):.2f} km, RMSE={np.sqrt(np.mean(distances[:, idx]**2)):.2f} km")
    log.info("=" * 60)

    return {"mae": mae, "rmse": rmse}


# ── Main ────────────────────────────────────────────────────────────────────

def run_for_spacecraft(spacecraft, models_to_run, subsample=1, horizon_hours=6):
    """Run full training pipeline for one spacecraft."""
    log.info(f"\n{'#'*60}")
    log.info(f"SPACECRAFT: {spacecraft.upper()}")
    log.info(f"{'#'*60}")

    results = {}

    # Load and preprocess orbit data
    log.info("Loading orbit data from HF...")
    orbit_df = load_spacecraft_data(spacecraft)
    orbit_processed, orbit_stats = preprocess_orbit(orbit_df, spacecraft)
    log.info(f"Orbit: {len(orbit_processed)} rows")

    # Detect time resolution from data
    time_diffs = orbit_processed["time"].diff().dt.total_seconds().dropna()
    median_res_min = int(np.median(time_diffs) / 60)
    if median_res_min < 1:
        median_res_min = 1
    log.info(f"Detected resolution: {median_res_min}-min")

    # Create windows adapted to data resolution
    input_steps = (24 * 60) // median_res_min   # 24h worth of steps
    horizon_steps = (horizon_hours * 60) // median_res_min  # 6h worth of steps
    stride = max(horizon_steps, 1)  # non-overlapping

    inputs, targets = create_windows(orbit_processed, input_steps, horizon_steps, stride, subsample)
    log.info(f"Windows: {len(inputs)} | input: {inputs.shape} | target: {targets.shape}")

    # Split 70/15/15
    n = len(inputs)
    n_train, n_val = int(0.7 * n), int(0.15 * n)

    bs = 64
    train_dl = DataLoader(OrbitDataset(inputs[:n_train], targets[:n_train]), batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
    val_dl = DataLoader(OrbitDataset(inputs[n_train:n_train+n_val], targets[n_train:n_train+n_val]), batch_size=bs, pin_memory=True, num_workers=4)
    test_dl = DataLoader(OrbitDataset(inputs[n_train+n_val:], targets[n_train+n_val:]), batch_size=bs, pin_memory=True, num_workers=4)

    input_dim = inputs.shape[-1]
    output_dim = targets.shape[-1]
    horizon = targets.shape[1]

    log.info(f"Split: train={n_train} | val={n_val} | test={n - n_train - n_val}")
    log.info(f"Dims: input={input_dim}, output={output_dim}, horizon={horizon}")

    # LSTM
    if "lstm" in models_to_run:
        log.info(f"\n--- LSTM ({spacecraft}) ---")
        model = OrbitLSTMDirect(input_dim, hidden_dim=128, num_layers=3, horizon=horizon, output_dim=output_dim)
        ckpt_name = f"lstm_{spacecraft}_{horizon_hours}h"
        model, _ = train_single(model, train_dl, val_dl, ckpt_name, epochs=100, patience=15)
        results["lstm"] = evaluate(model, test_dl, orbit_stats, f"LSTM ({spacecraft} {horizon_hours}h)")

    # Transformer
    if "transformer" in models_to_run:
        log.info(f"\n--- Transformer ({spacecraft}) ---")
        model = OrbitTransformerDirect(input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512,
                                       horizon=horizon, output_dim=output_dim)
        ckpt_name = f"transformer_{spacecraft}_{horizon_hours}h"
        model, _ = train_single(model, train_dl, val_dl, ckpt_name, epochs=100, patience=15)
        results["transformer"] = evaluate(model, test_dl, orbit_stats, f"Transformer ({spacecraft} {horizon_hours}h)")

    # Multi-modal
    if "multimodal" in models_to_run:
        log.info(f"\n--- Multi-Modal ({spacecraft}) ---")
        log.info("Loading solar wind data...")
        sw_df = load_solar_wind_data()
        sw_processed, sw_stats = preprocess_solar_wind(sw_df)

        o_wins, s_wins, t_wins = create_multimodal_windows(
            orbit_processed, sw_processed, input_steps, horizon_steps, stride, subsample
        )
        log.info(f"Multimodal windows: {len(o_wins)} | orbit: {o_wins.shape} | sw: {s_wins.shape} | target: {t_wins.shape}")

        nm = len(o_wins)
        nm_train, nm_val = int(0.7 * nm), int(0.15 * nm)

        mm_train = DataLoader(MultiModalDataset(o_wins[:nm_train], s_wins[:nm_train], t_wins[:nm_train]),
                              batch_size=bs, shuffle=True, pin_memory=True, num_workers=4)
        mm_val = DataLoader(MultiModalDataset(o_wins[nm_train:nm_train+nm_val], s_wins[nm_train:nm_train+nm_val], t_wins[nm_train:nm_train+nm_val]),
                            batch_size=bs, pin_memory=True, num_workers=4)
        mm_test = DataLoader(MultiModalDataset(o_wins[nm_train+nm_val:], s_wins[nm_train+nm_val:], t_wins[nm_train+nm_val:]),
                             batch_size=bs, pin_memory=True, num_workers=4)

        model = SolarWindOrbitModel(
            orbit_input_dim=o_wins.shape[-1], solar_input_dim=s_wins.shape[-1],
            hidden_dim=128, num_layers=3, nhead=8,
            horizon=t_wins.shape[1], output_dim=t_wins.shape[-1],
        )
        ckpt_name = f"multimodal_{spacecraft}_{horizon_hours}h"
        model, _ = train_multimodal(model, mm_train, mm_val, ckpt_name, epochs=100, patience=15)
        results["multimodal"] = evaluate(model, mm_test, orbit_stats, f"Multi-Modal ({spacecraft} {horizon_hours}h)", multimodal=True)

    return results


def push_checkpoints(hf_token=None):
    """Push all checkpoints to HF model repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token or os.environ.get("HF_TOKEN"))
        for ckpt in CHECKPOINT_DIR.glob("*_best.pt"):
            api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=f"checkpoints/{ckpt.name}",
                repo_id=HF_MODEL_REPO,
                repo_type="model",
            )
            log.info(f"Pushed {ckpt.name} to {HF_MODEL_REPO}")
    except Exception as e:
        log.error(f"Push failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer", "multimodal"])
    parser.add_argument("--spacecraft", type=str, default=None, help="iss, dscovr, mms1, or all")
    parser.add_argument("--subsample", type=int, default=1, help="1=full 1-min, 5=5-min, 10=10-min")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None, help="HF API token for pushing checkpoints")
    parser.add_argument("--horizon-hours", type=int, default=6, help="Prediction horizon in hours (1, 3, or 6)")
    args = parser.parse_args()

    start = time.time()
    log.info("=" * 60)
    log.info("ORBITAL CHAOS — GPU TRAINING")
    log.info(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.0f} GB)")
    log.info(f"PyTorch: {torch.__version__}")
    log.info(f"Subsample: {args.subsample}x")
    log.info(f"Horizon: {args.horizon_hours}h")
    log.info("=" * 60)

    models = [args.model] if args.model else ["lstm", "transformer", "multimodal"]
    spacecraft_list = [args.spacecraft] if args.spacecraft else ["iss", "dscovr", "mms1"]

    all_results = {}
    for sc in spacecraft_list:
        try:
            all_results[sc] = run_for_spacecraft(sc, models, args.subsample, args.horizon_hours)
        except Exception as e:
            log.error(f"{sc} failed: {e}\n{traceback.format_exc()}")

    # Push checkpoints to HF
    if not args.no_push:
        log.info("\nPushing checkpoints to HF...")
        push_checkpoints(hf_token=args.hf_token)

    # Summary
    elapsed = time.time() - start
    log.info(f"\n{'='*60}")
    log.info(f"ALL DONE in {elapsed/60:.1f} min")
    log.info("=" * 60)
    for sc, results in all_results.items():
        for model_name, r in results.items():
            log.info(f"  {sc:8s} {model_name:15s} | MAE={r['mae']:.2f} km | RMSE={r['rmse']:.2f} km")


if __name__ == "__main__":
    main()

    # Auto-stop RunPod pod when training is done
    import subprocess
    log.info("Training complete — stopping RunPod pod in 60 seconds...")
    log.info("(Cancel with: tmux send-keys -t train C-c)")
    import time as _t
    _t.sleep(60)
    try:
        subprocess.run(["runpodctl", "stop", "pod"], capture_output=True, timeout=10)
    except Exception:
        pass
    # Fallback: just halt
    os.system("shutdown -h now 2>/dev/null || true")
