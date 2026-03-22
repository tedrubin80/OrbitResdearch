# We Trained Neural Networks to Predict the ISS Orbit — and Solar Wind Makes Them Better During Storms

*Using 4.8 million NASA data points, bidirectional LSTMs, and a novel cross-attention architecture to predict spacecraft trajectories 6 hours ahead*

---

The International Space Station orbits Earth every 92 minutes at 28,000 km/h. Knowing exactly where it will be hours from now matters — for collision avoidance, resupply missions, and the 30,000+ tracked objects sharing low Earth orbit.

The standard physics tool for this, SGP4, works well on calm days. But there's a problem nobody outside the space tracking community thinks much about: **geomagnetic storms**.

When the Sun blasts a coronal mass ejection toward Earth, the upper atmosphere heats up and expands. At ISS altitude (~408 km), this thin but suddenly thicker atmosphere creates **extra drag** that SGP4 can't predict — its atmospheric model is essentially static. During the severe storm of May 2024 (Dst = -406 nT), orbit prediction errors spiked across the board.

We wanted to see if deep learning could do better. Specifically: **can a neural network learn the relationship between solar wind conditions measured upstream and the resulting orbit perturbations?**

The answer is yes — with caveats.

---

## The Data: 4.8 Million Position Records

We pulled three years (2023-2025) of spacecraft position data from NASA's Satellite Situation Center at 1-minute resolution:

- **ISS** — Low Earth Orbit at ~408 km. 1.58 million data points. Subject to significant atmospheric drag.
- **DSCOVR** — At the Sun-Earth L1 Lagrange point, 1.5 million km from Earth. 131K data points. Actually *measures* the solar wind before it hits Earth.
- **MMS-1** — Highly elliptical orbit through Earth's magnetosphere. 1.54 million data points.

We paired this with solar wind measurements from the OMNI database: interplanetary magnetic field (IMF) components, flow speed, proton density, and geomagnetic indices (Kp, Dst). Critically, solar wind measured at L1 arrives at Earth ~45 minutes later — giving the model a **natural leading indicator**.

## Three Architectures, One Winner (Mostly)

We trained each model on 24-hour input windows (1,440 timesteps) to predict 6-hour trajectories (360 timesteps of XYZ positions).

### 1. Bidirectional LSTM

Our workhorse. Encodes the full sequence in both temporal directions, concatenates the final hidden states, and predicts all 360 output positions directly through an MLP head. No autoregressive decoding — just a single forward pass.

**Result: 125 km MAE on ISS at 6 hours. 54.5 km at 1 hour.**

### 2. Transformer

Encoder-only with 4 self-attention layers, sinusoidal positional encodings, mean-pooled output. Despite attention's dominance in NLP, it achieved 282 km on ISS — **2.3x worse than the LSTM**. With 1,440-step sequences, the quadratic attention complexity may limit its ability to capture fine-grained temporal patterns that the LSTM handles naturally through its recurrent gates.

### 3. Multi-Modal Fusion (Our Novel Architecture)

This is where it gets interesting. We designed a **residual gated** architecture that fuses orbit positions with solar wind data:

```
output = base_prediction + gate * perturbation
```

- **Base prediction**: An orbit-only LSTM — functionally identical to the standalone model
- **Perturbation**: A correction signal learned from solar wind features via cross-attention (orbit features attend to solar wind features)
- **Gate**: A sigmoid network that controls *how much* correction to apply, per timestep

The key insight: because the gate is initialized near zero, **the model starts as a pure LSTM and can only improve from there**. If the solar wind data is noise (as it is for DSCOVR at L1), the gate stays closed and the model degrades gracefully.

We also use **two-phase training**: Phase 1 freezes the solar wind branch entirely and trains the orbit encoder for 20 epochs at LR=10^-3. Phase 2 unfreezes everything for 80 epochs at LR=10^-4. Without this, the solar branch destabilized the orbit encoder during early training and we couldn't get below 2,000 km MAE.

**Result: 163 km MAE on ISS overall. But read on.**

## The Headline: 17% Better During Storms

We split our ISS test set by concurrent Kp index — the standard measure of geomagnetic activity:

| Model | Quiet (Kp ≤ 3) | Active (Kp 4-5) | Storm (Kp ≥ 6) | Change |
|-------|---------------|-----------------|---------------|--------|
| LSTM | 126 km | 122 km | 113 km | -10% |
| **Multi-Modal** | **164 km** | **158 km** | **135 km** | **-17%** |
| Ensemble | 126 km | 124 km | **111 km** | -12% |
| SGP4 Kepler | 575 km | 575 km | 575 km | 0% |

**The multi-modal model improves 17% during storms** (164 → 135 km), compared to LSTM's 10% improvement. The cross-attention fusion is learning meaningful solar wind corrections — exactly what the thermosphere drag hypothesis predicted.

Even more interesting: a simple average of LSTM + multi-modal predictions (the "ensemble") achieves **111 km during storms** — the best result overall. The two models capture complementary information.

The SGP4 Keplerian baseline? It doesn't change at all with conditions. It's pure physics with no atmospheric awareness — 575 km regardless of whether there's a storm or not.

## The Surprise: Physics Wins on MMS-1

Not everything went as expected. For MMS-1's highly elliptical orbit, our simple two-body Keplerian propagation achieved just **88 km MAE** — better than all three ML models (18,000-19,000 km). Turns out, when your orbit is well-described by Newtonian gravity and atmospheric drag is negligible, the physics baseline is hard to beat. The ML models struggled with the extreme velocity changes near perigee.

Lesson: **always compute the physics baseline**.

## What Failed First

Our initial multi-modal architecture (v1) was catastrophic: **4,307 km MAE** on ISS — 34x worse than the standalone LSTM. The problem? Mean pooling.

```python
# v1: destroyed all temporal information
fused = torch.cat([attended.mean(dim=1), solar.mean(dim=1)], dim=-1)
```

Collapsing 1,440 timesteps of cross-attended features into a single vector threw away everything the attention mechanism learned. Combined with no residual connection, the model had to re-learn basic orbit prediction from scratch through this bottleneck. The v2 residual gated design fixed both issues and brought the MAE from 4,307 km down to 163 km — a **96% improvement**.

## More Features Isn't Always Better

We also tested an expanded 13-feature solar wind input — adding AL/AU auroral indices, IMF clock angle, and dynamic pressure to the original 8 features. The result? **181 km MAE** — worse than the 8-feature model (163 km). The original feature set already captures the relevant signal. More data doesn't help if it's noise.

## Prediction at Every Horizon

| Model | 1 hour | 3 hours | 6 hours |
|-------|--------|---------|---------|
| LSTM | **54.5 km** | **82.1 km** | **125 km** |
| Multi-Modal | 171 km | 176 km | 163 km |

At the 1-hour horizon, our LSTM achieves 54.5 km — **more than 10x better than the physics baseline**. Error grows roughly linearly with horizon for the LSTM, consistent with accumulating drag-induced deviations.

The multi-modal model shows a curious pattern: roughly constant MAE across horizons (163-176 km). This suggests its correction is more of a coarse offset than a fine temporal adjustment — an interesting direction for future architecture work.

## Try It Yourself

We've built a live interactive platform at **[orbitalchaos.online](https://orbitalchaos.online)** featuring:

- **3D ISS Tracker** — Real-time position on a rotating globe, updated every 5 seconds
- **Space Weather Dashboard** — Live Kp index, solar wind speed, and IMF Bz from NOAA
- **Storm-Conditioned Results** — Toggle between quiet/active/storm conditions to see how model accuracy changes
- **Prediction Demo** — Run the models directly in your browser via Hugging Face Spaces

Everything is open:
- **Models & checkpoints**: [Hugging Face](https://huggingface.co/datamatters24/orbital-chaos-predictor)
- **Dataset** (4.8M records): [Hugging Face Datasets](https://huggingface.co/datasets/datamatters24/orbital-chaos-nasa-ssc)
- **Notebooks**: [Kaggle](https://www.kaggle.com/theodorerubin/orbital-chaos-orbit-prediction)

---

*Ted Rubin is an independent researcher working on ML applications for space situational awareness. This project was trained on RunPod (H200 SXM + dual RTX 5090 GPUs) using PyTorch.*
