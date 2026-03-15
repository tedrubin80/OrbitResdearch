#!/usr/bin/env python3
"""Generate publication-quality figures from results JSONs.

Usage:
    python scripts/generate_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "lstm": "#2196F3",
    "transformer": "#9C27B0",
    "multimodal": "#E91E63",
    "ensemble": "#4CAF50",
    "sgp4": "#607D8B",
}

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
})


def load_results():
    with open("results/storm_conditioned_mae.json") as f:
        storm = json.load(f)
    with open("results/sgp4_baselines.json") as f:
        sgp4 = json.load(f)
    with open("results/horizon_comparison.json") as f:
        horizon = json.load(f)
    return storm, sgp4, horizon


def fig_mae_comparison(storm, sgp4):
    """Grouped bar chart: all models + SGP4, ISS and MMS-1."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ISS
    models = ["lstm", "transformer", "multimodal", "ensemble", "sgp4"]
    labels = ["LSTM", "Transformer", "Multi-Modal", "Ensemble", "SGP4\nKepler"]
    iss_vals = [
        storm["lstm"]["iss"]["all"],
        storm["transformer"]["iss"]["all"],
        storm["multimodal"]["iss"]["all"],
        storm["ensemble"]["iss"]["all"],
        sgp4["keplerian"]["iss"],
    ]
    colors = [COLORS[m] for m in models]

    bars = axes[0].bar(labels, iss_vals, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylabel("MAE (km)")
    axes[0].set_title("ISS (LEO) — 6h Prediction")
    for bar, val in zip(bars, iss_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # MMS-1
    mms_models = ["lstm", "transformer", "multimodal"]
    mms_labels = ["LSTM", "Transformer", "Multi-Modal"]
    mms_vals = [storm[m]["mms1"]["all"] for m in mms_models]
    mms_colors = [COLORS[m] for m in mms_models]

    # Add SGP4
    mms_models.append("sgp4")
    mms_labels.append("SGP4\nKepler")
    mms_vals.append(sgp4["keplerian"]["mms1"])
    mms_colors.append(COLORS["sgp4"])

    bars = axes[1].bar(mms_labels, mms_vals, color=mms_colors, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("MAE (km)")
    axes[1].set_title("MMS-1 (HEO) — 6h Prediction")
    for bar, val in zip(bars, mms_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                     f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mae_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved mae_comparison.png")


def fig_storm_vs_quiet(storm):
    """Side-by-side bars: quiet/active/storm MAE for ISS."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = ["lstm", "multimodal", "ensemble"]
    labels = ["LSTM", "Multi-Modal", "Ensemble"]
    conditions = ["quiet", "active", "storm"]
    cond_labels = ["Quiet\n(Kp≤3)", "Active\n(Kp 4-5)", "Storm\n(Kp≥6)"]

    x = np.arange(len(conditions))
    width = 0.25

    for i, (model, label) in enumerate(zip(models, labels)):
        vals = [storm[model]["iss"][c] for c in conditions]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=COLORS[model],
                      edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Geomagnetic Condition")
    ax.set_ylabel("MAE (km)")
    ax.set_title("ISS 6h Prediction — Storm-Conditioned Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 200)

    # Annotation arrow
    ax.annotate("17% improvement\nduring storms",
                xy=(2.25, 135), xytext=(2.5, 175),
                arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5),
                fontsize=9, color="#E91E63", ha="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "storm_vs_quiet.png", bbox_inches="tight")
    plt.close()
    print("Saved storm_vs_quiet.png")


def fig_horizon_comparison(horizon):
    """Line plot: MAE vs prediction horizon."""
    fig, ax = plt.subplots(figsize=(8, 5))

    hours = [1, 3, 6]

    for model_key, label, color, marker in [
        ("lstm_iss", "LSTM", COLORS["lstm"], "o"),
        ("multimodal_iss", "Multi-Modal", COLORS["multimodal"], "s"),
    ]:
        vals = [horizon[model_key][f"{h}h"] for h in hours]
        ax.plot(hours, vals, f"-{marker}", color=color, label=label,
                linewidth=2, markersize=8)
        for h, v in zip(hours, vals):
            ax.annotate(f"{v:.1f}", (h, v), textcoords="offset points",
                        xytext=(10, 5), fontsize=9, color=color)

    # SGP4 baseline (only have 6h)
    ax.axhline(y=575, color=COLORS["sgp4"], linestyle="--", alpha=0.7, label="SGP4 Kepler (6h)")
    ax.text(1.1, 560, "SGP4: 575 km", fontsize=9, color=COLORS["sgp4"])

    ax.set_xlabel("Prediction Horizon (hours)")
    ax.set_ylabel("MAE (km)")
    ax.set_title("ISS Prediction Accuracy vs Horizon")
    ax.set_xticks(hours)
    ax.set_xticklabels(["1h", "3h", "6h"])
    ax.legend(loc="upper left")
    ax.set_ylim(0, 650)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved horizon_comparison.png")


def fig_architecture():
    """Block diagram of the residual gated multi-modal architecture."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="gray", alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white",
                path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    def arrow(x1, y1, x2, y2, color="gray"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    # Orbit branch (top) — cyan
    c_orbit = "#1565C0"
    box(0.3, 4.5, 1.5, 0.8, "Orbit\nInput", c_orbit)
    box(2.3, 4.5, 1.2, 0.8, "Proj", c_orbit)
    box(4.0, 4.5, 1.5, 0.8, "BiLSTM", c_orbit)
    box(6.0, 4.5, 1.5, 0.8, "Base\nHead", c_orbit)
    box(8.5, 4.5, 1.5, 0.8, "Base\nPrediction", c_orbit, fontsize=8)

    arrow(1.8, 4.9, 2.3, 4.9, c_orbit)
    arrow(3.5, 4.9, 4.0, 4.9, c_orbit)
    arrow(5.5, 4.9, 6.0, 4.9, c_orbit)
    arrow(7.5, 4.9, 8.5, 4.9, c_orbit)

    # Solar wind branch (bottom) — pink
    c_solar = "#C62828"
    box(0.3, 0.8, 1.5, 0.8, "Solar\nInput", c_solar)
    box(2.3, 0.8, 1.2, 0.8, "Proj", c_solar)
    box(4.0, 0.8, 1.5, 0.8, "BiLSTM", c_solar)

    arrow(1.8, 1.2, 2.3, 1.2, c_solar)
    arrow(3.5, 1.2, 4.0, 1.2, c_solar)

    # Cross-attention (middle)
    c_cross = "#6A1B9A"
    box(4.0, 2.5, 1.5, 0.8, "Cross\nAttention", c_cross)

    # Arrows into cross-attention
    arrow(4.75, 4.5, 4.75, 3.3, c_orbit)  # orbit down
    arrow(4.75, 1.6, 4.75, 2.5, c_solar)  # solar up

    # Attention summary + perturbation
    box(6.0, 2.5, 1.5, 0.8, "Attn\nSummary", c_cross)
    box(8.0, 2.5, 1.5, 0.8, "Perturb\nMLP", c_solar)

    arrow(5.5, 2.9, 6.0, 2.9, c_cross)
    arrow(7.5, 2.9, 8.0, 2.9, c_cross)

    # Gate branch — green
    c_gate = "#2E7D32"
    box(6.0, 0.8, 1.5, 0.8, "Gate\nMLP", c_gate)
    box(8.0, 0.8, 0.6, 0.8, "σ", c_gate, fontsize=14)

    # orbit hidden to gate
    arrow(5.25, 4.5, 5.25, 3.8, "gray")
    ax.annotate("", xy=(6.75, 1.6), xytext=(5.25, 3.5),
                arrowprops=dict(arrowstyle="->", color=c_gate, lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    arrow(7.5, 1.2, 8.0, 1.2, c_gate)

    # Multiply gate * perturbation
    ax.text(9.0, 1.8, "×", fontsize=18, ha="center", va="center", color=c_gate)
    arrow(8.6, 1.2, 9.0, 1.6, c_gate)  # gate up
    arrow(9.0, 2.5, 9.0, 2.0, c_solar)  # perturbation down

    # Addition: base + gate*perturbation
    c_out = "#E65100"
    ax.text(10.2, 3.5, "+", fontsize=22, ha="center", va="center",
            color=c_out, fontweight="bold",
            bbox=dict(boxstyle="circle", facecolor="white", edgecolor=c_out, linewidth=2))

    arrow(10.0, 4.9, 10.2, 3.8, c_orbit)  # base down
    arrow(9.0, 1.9, 10.0, 3.3, c_solar)  # gate*perturb up

    # Output
    box(10.8, 3.1, 1.0, 0.8, "Output", c_out, fontsize=10)
    arrow(10.5, 3.5, 10.8, 3.5, c_out)

    # Legend
    legend_items = [
        mpatches.Patch(color=c_orbit, label="Orbit branch"),
        mpatches.Patch(color=c_solar, label="Solar wind branch"),
        mpatches.Patch(color=c_gate, label="Gate"),
        mpatches.Patch(color=c_cross, label="Cross-attention"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=9, framealpha=0.9)

    ax.set_title("Residual Gated Multi-Modal Architecture", fontsize=14, fontweight="bold", pad=10)

    plt.savefig(FIGURES_DIR / "architecture.png", bbox_inches="tight")
    plt.close()
    print("Saved architecture.png")


def main():
    storm, sgp4, horizon = load_results()

    fig_mae_comparison(storm, sgp4)
    fig_storm_vs_quiet(storm)
    fig_horizon_comparison(horizon)
    fig_architecture()

    print(f"\nAll figures saved to {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name} ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
