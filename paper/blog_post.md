# We Trained Neural Networks to Predict Where the ISS Will Be in 6 Hours

**TL;DR:** Our LSTM predicts ISS position to within 125 km at 6 hours (54.5 km at 1 hour) — 10x better than physics-based propagation. Adding solar wind data via cross-attention improves predictions by 17% during geomagnetic storms.

---

## The Problem

Every satellite in low Earth orbit faces an invisible enemy: atmospheric drag. The thermosphere — the tenuous upper atmosphere at 400+ km altitude — expands unpredictably during geomagnetic storms, increasing drag and causing orbit decay that standard physics models can't anticipate.

The standard tool for orbit prediction, SGP4, uses a static atmosphere model. It works fine on calm days. But when the Sun hurls a coronal mass ejection at Earth and the Kp index spikes above 6, SGP4's drag model is essentially guessing. We wanted to see if deep learning could do better by learning the messy, nonlinear relationship between solar wind conditions and orbit perturbations directly from data.

## The Data

We collected **4.8 million position records** at 1-minute resolution from NASA's Satellite Situation Center, covering three spacecraft from 2023-2025:

- **ISS** — Low Earth Orbit at ~408 km, subject to significant atmospheric drag
- **DSCOVR** — At the Sun-Earth L1 point, 1.5 million km away
- **MMS-1** — Highly elliptical orbit through the magnetosphere

We paired this with solar wind measurements from the OMNI database: IMF components (Bx, By, Bz), flow speed, proton density, and geomagnetic indices (Kp, Dst). The solar wind data arrives at Earth ~45 minutes after measurement at L1, giving us a natural leading indicator.

## The Models

We trained three architectures on 24-hour input windows to predict 6-hour trajectories:

**Bidirectional LSTM** — Our workhorse. Encodes the full 1,440-step input sequence bidirectionally, then directly predicts all 360 output positions. Simple, fast, and surprisingly hard to beat.

**Transformer** — Encoder with 4 attention layers, mean-pooled output. Despite attention's success elsewhere, it couldn't match the LSTM's sequential inductive bias for orbital dynamics.

**Multi-Modal Fusion** — Our novel architecture. Dual LSTM encoders (orbit + solar wind) with cross-attention fusion and a residual gated output: `prediction = base + gate * perturbation`. The sigmoid gate ensures the model can never be worse than the orbit-only LSTM — if the solar wind branch adds noise, the gate just closes.

Two-phase training was critical: Phase 1 freezes the solar branch and trains orbit-only for 20 epochs; Phase 2 unfreezes everything for 80 more epochs at 10x lower learning rate.

## Results

| Model | ISS 6h MAE | ISS 1h MAE |
|-------|-----------|-----------|
| **LSTM** | **125 km** | **54.5 km** |
| Multi-Modal | 163 km | 171 km |
| Transformer | 282 km | — |
| SGP4 Kepler | 575 km | — |

The LSTM beats the physics baseline by **4.6x**. At the 1-hour horizon, it's **10x better**.

### The Storm Result

The real payoff comes during geomagnetic storms. We split our test set by Kp index:

| Model | Quiet | Storm | Improvement |
|-------|-------|-------|-------------|
| LSTM | 126 km | 113 km | 10% |
| **Multi-Modal** | **164 km** | **135 km** | **17%** |
| Ensemble | 126 km | **111 km** | 12% |

The multi-modal model improves **17% during storms** — exactly what our hypothesis predicted. The solar wind branch is learning meaningful drag corrections from IMF and flow speed data. An ensemble of LSTM + multi-modal achieves the best storm performance at 111 km.

## What We Learned

1. **LSTM > Transformer for orbital dynamics.** The sequential inductive bias matters more than attention for time series with strong temporal continuity.

2. **Residual gating is essential for multi-modal fusion.** Our first attempt (v1 architecture with mean pooling) achieved 4,307 km MAE — 34x worse than LSTM. The gated residual design brought it to 163 km.

3. **Solar wind helps for LEO, hurts elsewhere.** ISS benefits from drag corrections. DSCOVR (at L1, no atmosphere) and MMS-1 (highly elliptical) don't — the solar wind branch just adds noise.

4. **Physics baselines can surprise you.** Simple two-body Kepler propagation achieves just 88 km on MMS-1, beating all our ML models (18,000+ km). HEO orbits are well-described by Newtonian gravity.

## Try It

- **Live site:** [orbitalchaos.online](https://orbitalchaos.online) — real-time ISS tracker, space weather dashboard, and interactive prediction demo
- **Models:** [datamatters24/orbital-chaos-predictor](https://huggingface.co/datamatters24/orbital-chaos-predictor) on Hugging Face
- **Dataset:** [datamatters24/orbital-chaos-nasa-ssc](https://huggingface.co/datasets/datamatters24/orbital-chaos-nasa-ssc) — 4.8M records, open access
- **Notebooks:** [Kaggle](https://www.kaggle.com/theodorerubin/orbital-chaos-orbit-prediction)
