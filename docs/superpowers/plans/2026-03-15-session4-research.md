# Session 4: Research & Visibility — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate publication-quality figures, update the paper with all Session 1 results, prepare ArXiv submission, and write a blog post.

**Architecture:** A single figure generation script reads all results JSONs and produces 6 PNG figures. The paper is updated with new tables, figures, and analysis. Blog post is a standalone markdown file.

**Tech Stack:** Python, matplotlib, seaborn, LaTeX (pdflatex + bibtex).

**Prerequisites:** Sessions 1-3 completed. All results JSONs in `results/`. Paper draft at `paper/paper.tex`.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `scripts/generate_figures.py` | Read results JSONs, produce 6 publication figures |
| `paper/figures/architecture.png` | Multi-modal architecture diagram |
| `paper/figures/mae_comparison.png` | Grouped bar chart: all models + SGP4 |
| `paper/figures/storm_vs_quiet.png` | Storm-conditioned MAE comparison |
| `paper/figures/training_curves.png` | Loss vs epoch for ISS models |
| `paper/figures/horizon_comparison.png` | MAE vs prediction horizon |
| `paper/figures/error_distribution.png` | Per-sample error histogram |
| `paper/blog_post.md` | 800-word accessible summary |

### Modified Files
| File | Changes |
|------|---------|
| `paper/paper.tex` | Add SGP4 row, storm tables, figures, updated abstract/conclusion |

---

## Chunk 1: Figure Generation

### Task 1: MAE Comparison and Storm Figures

**Files:** Create `scripts/generate_figures.py`

- [ ] **Step 1: Create the figure generation script**

Create `scripts/generate_figures.py` with functions for each figure. The script reads:
- `results/sgp4_baselines.json`
- `results/storm_conditioned_mae.json`
- `results/horizon_comparison.json`

**Style settings** (at top of script):
```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
from pathlib import Path

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'figure.facecolor': 'white',
})

COLORS = {
    'lstm': '#2196F3',
    'transformer': '#9C27B0',
    'multimodal': '#E91E63',
    'sgp4': '#607D8B',
    'ensemble': '#4CAF50',
}

FIGURES_DIR = Path('paper/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

**Figure 1: MAE Comparison** (`mae_comparison.png`)
- Grouped bar chart with models on x-axis, spacecraft as groups
- Include SGP4 baseline
- Y-axis: MAE in km
- Color-coded by model

**Figure 2: Storm vs Quiet** (`storm_vs_quiet.png`)
- Grouped bars: quiet / active / storm for ISS only
- Models: LSTM, Multi-modal, SGP4
- This is the key research figure — highlights when solar wind helps
- Include annotation arrows or text calling out the gap

**Figure 3: Horizon Comparison** (`horizon_comparison.png`)
- Line plot: x-axis = prediction horizon (1h, 3h, 6h), y-axis = MAE
- Lines: LSTM, Multi-modal (and SGP4 if available)
- ISS only

**Figure 4: Error Distribution** (`error_distribution.png`)
- Histogram or KDE of per-sample Euclidean errors
- Overlay LSTM and Multi-modal distributions
- ISS test set

Each figure function: load data, create figure, save to `paper/figures/`, close figure.

Main block: call all functions, print summary.

- [ ] **Step 2: Run figure generation**

```bash
cd /var/www/orbit && python3 scripts/generate_figures.py
```

Expected: 4 PNG files in `paper/figures/`. Verify by opening them.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_figures.py paper/figures/*.png
git commit -m "feat: generate publication figures from results"
```

---

### Task 2: Architecture Diagram

**Files:** Create `paper/figures/architecture.png` via matplotlib

- [ ] **Step 4: Add architecture diagram function to generate_figures.py**

Add `generate_architecture_diagram()` that draws:

```
Orbit Input --> [Proj] --> [BiLSTM] --> [Base Head] --> base ----+
                              |                                  |
                              v                                  | (+) --> Output
                        [Cross-Attn] <-- [BiLSTM] <-- Solar In   |
                              |                                  |
                              v                                  |
                      [Attn Summary]                             |
                              |                                  |
                              v                                  |
                      [Perturb MLP] --> delta ----> (x gate) ----+
                                                      ^
                              orbit h --> [Gate] --> sigma
```

Use `matplotlib.patches.FancyBboxPatch` for boxes, `matplotlib.patches.FancyArrowPatch` for arrows. Color-code:
- Cyan (`#2196F3`): orbit branch
- Pink (`#E91E63`): solar wind branch
- Green (`#4CAF50`): gate branch
- Gray: combination nodes

Size: 10x6 inches at 300 DPI.

- [ ] **Step 5: Generate architecture diagram**

```bash
cd /var/www/orbit && python3 -c "from scripts.generate_figures import generate_architecture_diagram; generate_architecture_diagram()"
```

Expected: `paper/figures/architecture.png` showing the residual gated architecture.

- [ ] **Step 6: Commit**

```bash
git add paper/figures/architecture.png scripts/generate_figures.py
git commit -m "feat: add architecture diagram figure"
```

---

## Chunk 2: Paper Updates

### Task 3: Update Paper with New Results

**Files:** Modify `paper/paper.tex`

- [ ] **Step 7: Add SGP4 row to main results table**

In the Results table (`\label{tab:results_all}`), add after the Multi-Modal row:

```latex
\midrule
SGP4 (TLE)     & XXX          & XXX             & XXX \\
Kepler (2-body) & XXX          & XXX             & XXX \\
```

Replace `XXX` with values from `results/sgp4_baselines.json`.

- [ ] **Step 8: Add storm-conditioned results subsection**

After Section 6.3 (Orbit Regime Analysis), add:

```latex
\subsection{Storm-Conditioned Evaluation}
\label{sec:storm}

Table~\ref{tab:storm} shows MAE on the ISS test set split by
concurrent Kp index.

\begin{table}[h]
\centering
\caption{ISS 6-hour MAE (km) by geomagnetic condition.}
\label{tab:storm}
\begin{tabular}{lcccc}
\toprule
Model & All & Quiet & Active & Storm \\
\midrule
LSTM        & XXX & XXX & XXX & XXX \\
Multi-Modal & XXX & XXX & XXX & XXX \\
SGP4 (TLE)  & XXX & XXX & XXX & XXX \\
Ensemble    & XXX & XXX & XXX & XXX \\
\bottomrule
\end{tabular}
\end{table}
```

Fill from `results/storm_conditioned_mae.json`.

- [ ] **Step 9: Add horizon comparison subsection**

```latex
\subsection{Prediction Horizon Analysis}

Figure~\ref{fig:horizon} shows MAE as a function of prediction
horizon for ISS.

\begin{figure}[h]
\centering
\includegraphics[width=0.9\columnwidth]{figures/horizon_comparison.png}
\caption{ISS MAE vs.\ prediction horizon.}
\label{fig:horizon}
\end{figure}
```

- [ ] **Step 10: Add all figure references**

Add `\includegraphics` for each figure in the appropriate section:
- `architecture.png` in Section 4.4 (Residual Gated Multi-Modal Fusion)
- `mae_comparison.png` in Section 6.1 (Overall Performance)
- `storm_vs_quiet.png` in Section 6.4 (Storm-Conditioned)
- `error_distribution.png` in Section 6.1 or 6.4

- [ ] **Step 11: Update abstract and conclusion with final numbers**

Replace any placeholder or approximate numbers with exact values from the results JSONs. Mention SGP4 comparison and storm-conditioned findings.

- [ ] **Step 12: Compile and verify PDF**

```bash
cd /var/www/orbit/paper && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
```

Expected: Clean PDF with all figures, tables, and references.

- [ ] **Step 13: Commit**

```bash
git add paper/paper.tex paper/figures/
git commit -m "feat: update paper with SGP4 baselines, storm results, figures"
```

---

## Chunk 3: Publication Prep

### Task 4: ArXiv Package

- [ ] **Step 14: Verify paper uses only standard LaTeX packages**

Check that all `\usepackage` commands use packages available in a standard TeX Live installation. Current packages (amsmath, graphicx, booktabs, hyperref, natbib, geometry, caption, subcaption, multirow) are all standard.

- [ ] **Step 15: Create ArXiv submission tarball**

```bash
cd /var/www/orbit/paper && tar -czf ../arxiv-submission.tar.gz \
    paper.tex references.bib figures/
```

- [ ] **Step 16: Verify tarball compiles**

```bash
mkdir -p /tmp/arxiv-test && cd /tmp/arxiv-test
tar -xzf /var/www/orbit/arxiv-submission.tar.gz
pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
```

Expected: Clean PDF generated from tarball alone.

- [ ] **Step 17: Commit**

```bash
cd /var/www/orbit && git add arxiv-submission.tar.gz
git commit -m "feat: create ArXiv submission package"
```

---

### Task 5: Blog Post

- [ ] **Step 18: Write blog post**

Create `paper/blog_post.md` — ~800 words covering:

1. **Hook**: "We trained neural networks to predict where the ISS will be 6 hours from now"
2. **The problem**: SGP4 fails during geomagnetic storms, thermosphere expands unpredictably
3. **The data**: 4.8M positions, 3 spacecraft, 3 years, 1-minute resolution
4. **The models**: LSTM wins (126 km MAE), but multi-modal (175 km) incorporates solar wind
5. **Key finding**: During storms, multi-modal closes the gap with LSTM / outperforms SGP4
6. **The architecture**: Residual gating — "can never be worse than LSTM"
7. **Live demo**: Link to orbitalchaos.online with live ISS tracker + space weather
8. **Links**: HF model/dataset/space, Kaggle notebooks, paper

Include markdown image references to the dark-background figure versions (to be generated later, or use the white-background ones).

- [ ] **Step 19: Commit**

```bash
git add paper/blog_post.md
git commit -m "feat: add blog post draft"
```

---

## Chunk 4: Final Push

### Task 6: Final Verification

- [ ] **Step 20: Verify all deliverables exist**

```bash
ls -la paper/figures/*.png
ls -la arxiv-submission.tar.gz
ls -la paper/blog_post.md
```

Expected: 5-6 figures, tarball, blog post.

- [ ] **Step 21: PDF page count and quality check**

```bash
pdfinfo paper/paper.pdf | grep Pages
```

Expected: 8-10 pages with figures.

- [ ] **Step 22: Push everything**

```bash
cd /var/www/orbit && git push origin main
```

---

## Session 4 Completion Checklist

- [ ] `scripts/generate_figures.py` — produces all figures from results JSONs
- [ ] 5 figures in `paper/figures/` (architecture, mae_comparison, storm_vs_quiet, horizon, error_dist)
- [ ] `paper/paper.tex` — updated with SGP4 baselines, storm table, figures, final numbers
- [ ] `arxiv-submission.tar.gz` — compiles independently
- [ ] `paper/blog_post.md` — 800-word accessible summary
- [ ] All pushed to git

**Project improvement complete across all 4 sessions.**
