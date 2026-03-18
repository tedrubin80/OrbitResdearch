---
language: en
tags:
  - orbit-prediction
  - space-weather
  - lstm
  - transformer
  - multimodal
  - time-series
  - geomagnetic-storms
  - solar-wind
license: mit
datasets:
  - datamatters24/orbital-chaos-nasa-ssc
---

# Orbital Chaos Predictor

Multi-modal deep learning for spacecraft orbit prediction incorporating solar wind perturbations via cross-attention fusion.

**[Live Dashboard](https://orbitalchaos.online)** | **[Dataset](https://huggingface.co/datasets/datamatters24/orbital-chaos-nasa-ssc)** | **[Paper](https://orbitalchaos.online)**

## Model Description

We train bidirectional LSTMs, Transformers, and a novel residual gated multi-modal architecture to predict spacecraft orbits 6 hours ahead using 24 hours of position history. The multi-modal model fuses orbit positions with solar wind measurements via cross-attention, improving predictions by 17% during geomagnetic storms.

| Architecture | Parameters | ISS MAE (6h) |
|---|---|---|
| BiLSTM | 660K | 125 km |
| Transformer | 690K | 282 km |
| Residual Gated Multi-Modal | 1.9M | 163 km (135 km during storms) |

## Checkpoints

All checkpoints are available in both PyTorch (`.pt`) and safetensors (`.safetensors`) formats.

| Checkpoint | Architecture | Spacecraft | Orbit Type |
|---|---|---|---|
| `lstm_iss_6h_best` | BiLSTM | ISS | LEO (~408 km) |
| `lstm_dscovr_6h_best` | BiLSTM | DSCOVR | L1 (~1.5M km) |
| `lstm_mms1_6h_best` | BiLSTM | MMS-1 | HEO (magnetosphere) |
| `transformer_iss_6h_best` | Transformer | ISS | LEO |
| `transformer_dscovr_6h_best` | Transformer | DSCOVR | L1 |
| `transformer_mms1_6h_best` | Transformer | MMS-1 | HEO |
| `multimodal_iss_6h_best` | Residual Gated Multi-Modal | ISS | LEO |
| `multimodal_dscovr_6h_best` | Residual Gated Multi-Modal | DSCOVR | L1 |
| `multimodal_mms1_6h_best` | Residual Gated Multi-Modal | MMS-1 | HEO |

## Usage

### Loading with safetensors (recommended)

```python
import torch
from safetensors.torch import load_file

# Load ISS LSTM checkpoint
state_dict = load_file("checkpoints/lstm_iss_6h_best.safetensors")

# Build model (match checkpoint architecture)
from your_model_file import OrbitLSTMDirect

model = OrbitLSTMDirect(
    input_dim=6, hidden_dim=128, num_layers=3,
    horizon=360, output_dim=3, dropout=0.0
)
model.load_state_dict(state_dict)
```

### Loading with PyTorch (.pt files)

```python
import torch

ckpt = torch.load(
    "checkpoints/lstm_iss_6h_best.pt",
    map_location="cpu",
    weights_only=False  # Required: checkpoints contain numpy metadata
)
model.load_state_dict(ckpt["model_state_dict"])
```

## Input / Output Specification

### Orbit-only models (LSTM, Transformer)

- **Input:** `(batch, 1440, 6)` -- 24-hour window at 1-minute resolution
  - Features: `[x_gse, y_gse, z_gse, vx_gse, vy_gse, vz_gse]` in GSE coordinates, z-score normalized
- **Output:** `(batch, 360, 3)` -- 6-hour prediction
  - Positions: `[x_gse, y_gse, z_gse]`, normalized (denormalize with training stats)

### Multi-modal model

- **Orbit input:** `(batch, 1440, 6)` -- same as above
- **Solar wind input:** `(batch, 1440, 8)` -- concurrent OMNI solar wind data
  - Features: `[Bx_GSE, By_GSE, Bz_GSE, flow_speed, proton_density, Kp, Dst, AE]`, z-score normalized
  - Time-shifted by 45 minutes (L1-to-Earth propagation delay)
- **Output:** `(batch, 360, 3)` -- same as orbit-only

## Training Data

- **Orbit positions:** NASA Satellite Situation Center (SSC) REST API
- **Solar wind:** NASA CDAWeb OMNI 1-minute high-resolution dataset
- **Period:** January 2023 -- December 2025
- **Volume:** ~4.8 million position records across 3 spacecraft
- **Split:** 70% train / 15% validation / 15% test (chronological)
- **Resolution:** 1-minute cadence
- **Preprocessing:** Velocity derived via finite differences, z-score normalization per spacecraft, gap detection at 10-minute threshold

Full dataset: [datamatters24/orbital-chaos-nasa-ssc](https://huggingface.co/datasets/datamatters24/orbital-chaos-nasa-ssc)

## Performance

### 6-Hour Prediction MAE (km)

| Model | ISS (LEO) | DSCOVR (L1) | MMS-1 (HEO) |
|---|---|---|---|
| **LSTM** | **125** | **12,797** | 18,832 |
| Transformer | 282 | 13,517 | 19,296 |
| Multi-Modal | 163 | 25,059 | 19,277 |
| Ensemble (LSTM+MM) | 126 | -- | -- |
| SGP4 Keplerian baseline | 575 | -- | **88** |

### Storm-Conditioned ISS (by Kp Index)

| Model | All (n=657) | Quiet Kp<=3 (n=489) | Active Kp 4-5 (n=71) | Storm Kp>=6 (n=7) |
|---|---|---|---|---|
| LSTM | 125 km | 126 km | 122 km | 113 km |
| Multi-Modal | 163 km | 164 km | 158 km | **135 km** |
| Ensemble | 126 km | 126 km | 124 km | **111 km** |

The multi-modal model improves **17% during geomagnetic storms** (164 to 135 km), validating the hypothesis that solar wind data provides meaningful signal for LEO orbit prediction where thermospheric drag dominates.

### Horizon Comparison (ISS)

| Model | 1 hour | 3 hours | 6 hours |
|---|---|---|---|
| LSTM | **54.5 km** | **82.1 km** | **125 km** |
| Multi-Modal | 171.5 km | 176.2 km | 163 km |

## Architecture Details

### Residual Gated Multi-Modal Fusion

```
output = base_prediction + gate * perturbation
```

- **Base prediction:** Orbit-only BiLSTM (functionally identical to standalone)
- **Perturbation:** Cross-attention fusion of orbit and solar wind features, followed by attention-weighted summary and 3-layer MLP
- **Gate:** Sigmoid network controlling correction strength per timestep
- **Guarantee:** Gate initialized near zero; model starts as pure LSTM and can only improve

**Two-phase training:**
1. Phase 1 (20 epochs, LR=1e-3): Freeze solar branch, train orbit encoder only
2. Phase 2 (80 epochs, LR=1e-4): Unfreeze all, fine-tune with lower learning rate

## Citation

```bibtex
@article{rubin2026orbital,
  author = {Rubin, Ted},
  title = {Multi-Modal Deep Learning for Spacecraft Orbit Prediction: Incorporating Solar Wind Perturbations via Cross-Attention Fusion},
  year = {2026},
  note = {arXiv preprint arXiv:XXXX.XXXXX}
}
```

## License

MIT
