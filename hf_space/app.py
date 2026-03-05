"""Orbital Chaos — GPU Training Space on Hugging Face.

Loads dataset from datamatters24/orbital-chaos-nasa-ssc,
trains LSTM/Transformer/Multi-modal models on T4 GPU,
and pushes results to datamatters24/orbital-chaos-predictor.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import spaces
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("orbit-gpu")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_DATASET = "datamatters24/orbital-chaos-nasa-ssc"
HF_MODEL_REPO = "datamatters24/orbital-chaos-predictor"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Models ──────────────────────────────────────────────────────────────────

class OrbitLSTMDirect(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(hidden_dim, horizon * output_dim))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.fc(h).view(-1, self.horizon, self.output_dim)


class OrbitTransformerDirect(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=3, dim_feedforward=256,
                 horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim_feedforward, horizon * output_dim))

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
    """Residual gated multi-modal: output = base_prediction + gate * perturbation."""
    def __init__(self, orbit_input_dim=6, solar_input_dim=7, hidden_dim=128, num_layers=2,
                 nhead=4, horizon=360, output_dim=3, dropout=0.1):
        super().__init__()
        self.horizon, self.output_dim = horizon, output_dim
        # Orbit encoder
        self.orbit_proj = nn.Linear(orbit_input_dim, hidden_dim)
        self.orbit_enc = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.orbit_norm = nn.LayerNorm(hidden_dim * 2)
        # Base prediction head (LSTM-equivalent)
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim))
        # Solar wind encoder
        self.solar_proj = nn.Linear(solar_input_dim, hidden_dim)
        self.solar_enc = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True,
                                 dropout=dropout if num_layers > 1 else 0)
        self.solar_norm = nn.LayerNorm(hidden_dim * 2)
        # Cross-attention
        self.cross_attn = CrossModalAttention(hidden_dim * 2, nhead, dropout)
        # Attention-weighted summary
        self.attn_weight = nn.Linear(hidden_dim * 2, 1)
        # Perturbation head (3-layer MLP)
        self.perturbation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * output_dim))
        # Gate
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, horizon * output_dim), nn.Sigmoid())

    def forward(self, orbit_input, solar_input):
        # Encode orbit
        orbit_emb = self.orbit_proj(orbit_input)
        orbit_encoded, (h, _) = self.orbit_enc(orbit_emb)
        orbit_encoded = self.orbit_norm(orbit_encoded)
        # Base prediction from final hidden states
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        base = self.base_head(h_cat).view(-1, self.horizon, self.output_dim)
        # Encode solar wind
        s = self.solar_norm(self.solar_enc(self.solar_proj(solar_input))[0])
        # Cross-attention
        attended = self.cross_attn(orbit_encoded, s)
        # Attention-weighted summary
        attn_scores = torch.softmax(self.attn_weight(attended), dim=1)
        summary = (attended * attn_scores).sum(dim=1)
        # Perturbation + gate
        perturbation = self.perturbation_head(summary).view(-1, self.horizon, self.output_dim)
        gate = self.gate_net(h_cat).view(-1, self.horizon, self.output_dim)
        return base + gate * perturbation

    def freeze_solar_branch(self):
        for module in [self.solar_proj, self.solar_enc, self.solar_norm,
                       self.cross_attn, self.attn_weight, self.perturbation_head, self.gate_net]:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


# ── Data Loading ────────────────────────────────────────────────────────────

def load_hf_dataset():
    """Load parquet files from HF dataset."""
    from datasets import load_dataset
    log.info(f"Loading dataset from {HF_DATASET}...")
    ds = load_dataset(HF_DATASET)
    return ds


def load_parquet_from_hf(filename):
    """Download a specific parquet file from HF dataset repo."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_DATASET, filename=filename, repo_type="dataset")
    return pd.read_parquet(path)


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

    time_diff = df["time"].diff().dt.total_seconds()
    df["segment_id"] = (time_diff > 600).cumsum()

    feature_cols = [c for c in df.columns
                    if c.startswith(("x_gse", "y_gse", "z_gse", "vx_gse", "vy_gse", "vz_gse"))
                    and not c.endswith("_norm")]

    stats = {"mean": df[feature_cols].mean().to_dict(), "std": df[feature_cols].std().to_dict()}

    for col in feature_cols:
        std = stats["std"][col]
        df[f"{col}_norm"] = (df[col] - stats["mean"][col]) / std if std > 0 else 0.0

    return df, stats


def create_windows(df, input_steps=1440, horizon_steps=360, stride=360, subsample=1):
    """Create sliding windows from preprocessed data."""
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


# ── Training ────────────────────────────────────────────────────────────────

def train_model(model, train_x, train_y, val_x, val_y, name, epochs=100, patience=15, batch_size=64, multimodal_sw=None, val_sw=None):
    """Train a model with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Training {name} | params: {n_params:,} | device: {DEVICE}")

    best_val = float("inf")
    patience_ctr = 0
    log_lines = []

    train_ds = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = torch.utils.data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    if multimodal_sw is not None:
        train_ds = torch.utils.data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(multimodal_sw), torch.from_numpy(train_y))
        val_ds = torch.utils.data.TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_sw), torch.from_numpy(val_y))

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = []
        for batch in train_dl:
            optimizer.zero_grad()
            if multimodal_sw is not None:
                x, sw, y = [b.to(DEVICE) for b in batch]
                pred = model(x, sw)
            else:
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                pred = model(x)
            ml = min(pred.shape[1], y.shape[1])
            loss = criterion(pred[:, :ml], y[:, :ml])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss.append(loss.item())

        scheduler.step()
        avg_t = np.mean(t_loss)

        model.eval()
        v_loss = []
        with torch.no_grad():
            for batch in val_dl:
                if multimodal_sw is not None:
                    x, sw, y = [b.to(DEVICE) for b in batch]
                    pred = model(x, sw)
                else:
                    x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
                    pred = model(x)
                ml = min(pred.shape[1], y.shape[1])
                v_loss.append(criterion(pred[:, :ml], y[:, :ml]).item())

        avg_v = np.mean(v_loss)
        line = f"Epoch {epoch:3d}/{epochs} | train={avg_t:.6f} | val={avg_v:.6f}"
        log.info(line)
        log_lines.append(line)

        if avg_v < best_val:
            best_val = avg_v
            patience_ctr = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val},
                       CHECKPOINT_DIR / f"{name}_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

    return model, best_val, log_lines


def push_to_hub(name):
    """Push checkpoint to HF model repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        ckpt = CHECKPOINT_DIR / f"{name}_best.pt"
        if ckpt.exists():
            api.upload_file(path_or_fileobj=str(ckpt), path_in_repo=f"checkpoints/{name}_best.pt",
                            repo_id=HF_MODEL_REPO, repo_type="model")
            log.info(f"Pushed {name} checkpoint to {HF_MODEL_REPO}")
            return True
    except Exception as e:
        log.error(f"Push failed: {e}")
    return False


# ── Gradio App ──────────────────────────────────────────────────────────────

def run_training(model_choice, spacecraft, subsample, epochs, progress=gr.Progress()):
    """Main training function called by Gradio."""
    start = time.time()
    output = []
    output.append(f"Device: {DEVICE}")
    output.append(f"Model: {model_choice} | Spacecraft: {spacecraft} | Subsample: {subsample}x | Epochs: {epochs}")
    output.append("")

    # Load data
    progress(0.05, desc="Loading data from HF...")
    try:
        orbit_df = load_parquet_from_hf(f"{spacecraft}_2023-01-01_2025-12-31.parquet")
        output.append(f"Loaded orbit data: {len(orbit_df)} rows")
    except Exception as e:
        return f"Failed to load orbit data: {e}"

    # Preprocess
    progress(0.1, desc="Preprocessing...")
    orbit_processed, stats = preprocess_orbit(orbit_df, spacecraft)
    output.append(f"Preprocessed: {len(orbit_processed)} rows")

    # Windows
    subsample = int(subsample)
    input_steps = 1440  # 24h at 1-min
    horizon_steps = 360  # 6h at 1-min
    stride = 360  # 6h stride

    inputs, targets = create_windows(orbit_processed, input_steps, horizon_steps, stride, subsample)
    output.append(f"Windows: {len(inputs)} | Input: {inputs.shape} | Target: {targets.shape}")

    # Split
    n = len(inputs)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    train_x, train_y = inputs[:n_train], targets[:n_train]
    val_x, val_y = inputs[n_train:n_train+n_val], targets[n_train:n_train+n_val]
    test_x, test_y = inputs[n_train+n_val:], targets[n_train+n_val:]
    output.append(f"Split: train={len(train_x)} | val={len(val_x)} | test={len(test_x)}")

    input_dim = inputs.shape[-1]
    output_dim = targets.shape[-1]
    horizon = targets.shape[1]

    # Build model
    progress(0.15, desc="Building model...")
    if model_choice == "LSTM":
        model = OrbitLSTMDirect(input_dim, hidden_dim=128, num_layers=2, horizon=horizon, output_dim=output_dim)
    elif model_choice == "Transformer":
        model = OrbitTransformerDirect(input_dim, d_model=128, nhead=4, num_layers=3, horizon=horizon, output_dim=output_dim)
    else:  # Multi-modal
        # Load solar wind too
        try:
            sw_df = load_parquet_from_hf("solar_wind_2023-01-01_2025-12-31.parquet")
            output.append(f"Solar wind data: {len(sw_df)} rows")
        except Exception as e:
            return f"Failed to load solar wind: {e}"
        output.append("Multi-modal requires additional processing... (full implementation in train_all.py)")
        # For now, train orbit-only LSTM as fallback
        model = OrbitLSTMDirect(input_dim, hidden_dim=128, num_layers=2, horizon=horizon, output_dim=output_dim)
        model_choice = "LSTM (fallback)"

    n_params = sum(p.numel() for p in model.parameters())
    output.append(f"Model params: {n_params:,}")
    output.append("")

    # Train
    progress(0.2, desc="Training...")
    model, best_val, log_lines = train_model(model, train_x, train_y, val_x, val_y,
                                              name=f"{model_choice.lower()}_{spacecraft}_6h",
                                              epochs=int(epochs), patience=15, batch_size=64)
    output.extend(log_lines)

    # Evaluate
    progress(0.9, desc="Evaluating...")
    model.eval()
    with torch.no_grad():
        test_t = torch.from_numpy(test_x).to(DEVICE)
        preds = model(test_t).cpu().numpy()
    ml = min(preds.shape[1], test_y.shape[1])
    distances = np.sqrt(np.sum((preds[:, :ml] - test_y[:, :ml])**2, axis=-1))
    mae, rmse = np.mean(distances), np.sqrt(np.mean(distances**2))
    output.append(f"\nTest MAE: {mae:.6f} (normalized) | RMSE: {rmse:.6f}")

    # Push
    progress(0.95, desc="Pushing to HF Hub...")
    name = f"{model_choice.lower().replace(' ', '_')}_{spacecraft}_6h"
    pushed = push_to_hub(name)
    if pushed:
        output.append(f"Checkpoint pushed to {HF_MODEL_REPO}")

    elapsed = time.time() - start
    output.append(f"\nDone in {elapsed/60:.1f} min")

    return "\n".join(output)


# ── UI ──────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Orbital Chaos Training", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Orbital Chaos — GPU Training")
    gr.Markdown("Train orbit prediction models on NASA spacecraft data using a T4 GPU.")

    with gr.Row():
        model_dd = gr.Dropdown(["LSTM", "Transformer"], label="Model", value="LSTM")
        sc_dd = gr.Dropdown(["iss", "dscovr", "mms1"], label="Spacecraft", value="iss")
        sub_sl = gr.Slider(1, 10, value=1, step=1, label="Subsample (1=full 1-min, 10=10-min)")
        epoch_sl = gr.Slider(10, 200, value=100, step=10, label="Epochs")

    train_btn = gr.Button("Start Training", variant="primary")
    output = gr.Textbox(label="Training Log", lines=30, max_lines=50)

    train_btn.click(run_training, [model_dd, sc_dd, sub_sl, epoch_sl], output)

    gr.Markdown(f"""
    **Dataset:** [{HF_DATASET}](https://huggingface.co/datasets/{HF_DATASET})
    **Model repo:** [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO})
    """)

demo.launch()
