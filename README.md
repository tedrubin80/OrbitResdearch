# Orbital Chaos

ML-powered spacecraft orbit prediction using 3 years of NASA telemetry data. Trains LSTM, Transformer, and multi-modal (orbit + solar wind) models to predict 6-hour trajectories for ISS, DSCOVR, and MMS-1.

**Live site:** [orbitalchaos.online](https://orbitalchaos.online) — includes real-time ISS tracker, space weather dashboard, and interactive prediction demo.

## Results

6-hour prediction MAE (km), trained on dual RTX 5090 GPUs:

| Model | ISS (LEO) | DSCOVR (L1) | MMS-1 (HEO) |
|-------|-----------|-------------|--------------|
| **LSTM** | **126 km** | **12,797 km** | **18,683 km** |
| Transformer | 295 km | 13,517 km | 19,237 km |
| Multi-Modal | 175 km | 25,059 km | 19,457 km |

The multi-modal model uses a residual gated architecture (`output = base + gate * perturbation`) that incorporates solar wind data via cross-attention, improving ISS predictions by capturing thermosphere drag during geomagnetic storms.

## Project Structure

```
├── src/
│   ├── models/          # LSTM, Transformer, Multi-modal, SGP4 baseline, TF variants
│   ├── data/            # NASA SSC client, solar wind fetcher, PyTorch dataset, preprocessing
│   ├── training/        # Trainers (PyTorch + TensorFlow), evaluation metrics
│   └── utils/           # Coordinate transforms (GSE/GEO), visualization
├── scripts/
│   ├── cron_fetch.py    # Daily automated data fetch from NASA APIs
│   ├── train_all.py     # Full training pipeline (CPU)
│   ├── train_gpu.py     # GPU training for RunPod (loads from HF, pushes checkpoints)
│   └── eval_checkpoints.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_orbit_prediction.ipynb
│   └── 03_solar_wind.ipynb
├── hf_space/            # Gradio app for HF Spaces (interactive training UI)
├── public/              # Static website (orbitalchaos.online)
├── data/                # Raw + processed datasets (4.8M+ data points)
├── checkpoints/         # Trained model weights
└── paper/               # Research paper (LaTeX)
```

## Data

- **Sources:** [NASA SSC API](https://sscweb.gsfc.nasa.gov/) (spacecraft positions) + [OMNI](https://omniweb.gsfc.nasa.gov/) (solar wind)
- **Spacecraft:** ISS (LEO, ~408 km), DSCOVR (L1, ~1.5M km), MMS-1 (HEO, magnetosphere)
- **Volume:** 4.8M+ data points at 1-minute resolution, 2023-2025
- **Solar wind features:** IMF (Bx, By, Bz), flow speed, proton density, Kp index, Dst index
- **Pipeline:** Automated daily fetch via cron at 02:00 UTC with S3 backup

## Models

**Bidirectional LSTM** — Encoder-decoder with autoregressive decoding. 24h input (1440 steps) to 6h prediction (360 steps). Best overall performer.

**Transformer** — Encoder with learned query tokens and multi-head cross-attention. Mean-pooled encoding to direct prediction.

**Multi-Modal Fusion** — Dual LSTM encoders (orbit + solar wind) with cross-modal attention. Residual gated design ensures performance floor matches standalone LSTM. Two-phase training: Phase 1 freezes solar branch (20 epochs), Phase 2 fine-tunes everything (80 epochs).

**SGP4/Kepler Baseline** — Physics-based two-body propagation for comparison.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python scripts/fetch_data.py

# Train models (CPU)
python scripts/train_all.py

# GPU training (RunPod / cloud)
python scripts/train_gpu.py
```

## Hugging Face

- **Models:** [datamatters24/orbital-chaos-predictor](https://huggingface.co/datamatters24/orbital-chaos-predictor)
- **Dataset:** [datamatters24/orbital-chaos-nasa-ssc](https://huggingface.co/datasets/datamatters24/orbital-chaos-nasa-ssc)
- **Space:** [datamatters24/orbital-chaos-training](https://huggingface.co/spaces/datamatters24/orbital-chaos-training)

## Tech Stack

PyTorch, TensorFlow, Gradio, NumPy, Pandas, Astropy, SGP4, Globe.GL (3D visualization), NOAA SWPC APIs (live space weather)
