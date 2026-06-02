"""
Scaling law plot: Chinchilla compute-optimal frontier with GraphMERT positions.

Chinchilla (Hoffmann et al., 2022): optimal training uses N_tokens ≈ 20 × N_params.
This plot visualises how far the networking corpus sits below the frontier,
and marks the compute-optimal model size for that corpus (~15k params).

Run: python3 graphrag/plot_scaling_law.py
Output: graphrag/scaling_law.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "scaling_law.png"

# ── Data points ──────────────────────────────────────────────────────────────
# (label, params, tokens, marker, color, annotate_side)
MODELS = [
    # General LMs — on or near the Chinchilla frontier
    ("BERT-base",          110e6,   3.3e9,   "o", "#4e79a7", "right"),
    ("RoBERTa-base",       125e6,   160e9,   "o", "#59a14f", "right"),
    ("GPT-2",              1.5e9,   10e9,    "o", "#f28e2b", "right"),
    ("GPT-3",              175e9,   300e9,   "o", "#e15759", "right"),
    ("Chinchilla\n(70B)",  70e9,    1.4e12,  "o", "#76b7b2", "left"),
    ("LLaMA-7B",           7e9,     1e12,    "o", "#b07aa1", "right"),
    # GraphMERT — medical (close to frontier) and networking (far below)
    ("GraphMERT\n(medical)",    67e6,  125e6,  "D", "#ff6b35", "right"),
    ("GraphMERT\n(networking)", 67e6,  296e3,  "*", "#d62728", "left"),
]

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xscale("log")
ax.set_yscale("log")

# ── Chinchilla optimal frontier: tokens = 20 × params ────────────────────────
params_range = np.logspace(3, 12, 400)
chinchilla_tokens = 20 * params_range
ax.plot(params_range, chinchilla_tokens,
        color="#888888", linewidth=1.8, linestyle="--", zorder=1,
        label="Chinchilla optimal (tokens = 20 × params)")

# Shaded "under-trained" region below the frontier
ax.fill_between(params_range, 1e3, chinchilla_tokens,
                alpha=0.07, color="#d62728", zorder=0)
ax.text(2e10, 5e8, "under-trained\nregion", color="#d62728",
        fontsize=9, alpha=0.7, ha="center", style="italic")

# ── Plot each model ───────────────────────────────────────────────────────────
for label, params, tokens, marker, color, side in MODELS:
    zorder = 5 if "networking" in label else 3
    ms = 14 if marker == "*" else (10 if marker == "D" else 8)
    ax.scatter(params, tokens, marker=marker, color=color, s=ms**2,
               zorder=zorder, edgecolors="white", linewidths=0.6)

    # Label offset
    x_off = 1.35 if side == "right" else 0.72
    y_off = 1.0
    if "networking" in label:
        x_off = 0.60
        y_off = 0.35
    if "GPT-3" in label:
        y_off = 0.5
    if "RoBERTa" in label:
        y_off = 2.5

    ax.annotate(
        label,
        xy=(params, tokens),
        xytext=(params * x_off, tokens * y_off),
        fontsize=8.5,
        color=color,
        ha="left" if side == "right" else "right",
        va="center",
        fontweight="bold" if "networking" in label or "medical" in label else "normal",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

# ── Optimal-size annotation for 296k tokens ───────────────────────────────────
optimal_params = 296e3 / 20   # ≈ 14,800 parameters
ax.axhline(296e3, color="#d62728", linewidth=0.9, linestyle=":", alpha=0.6)
ax.axvline(optimal_params, color="#d62728", linewidth=0.9, linestyle=":", alpha=0.6)
ax.scatter(optimal_params, 296e3, marker="x", color="#d62728", s=120, zorder=6, linewidths=2)
ax.annotate(
    f"compute-optimal for\n296k tokens\n≈ 14,800 params",
    xy=(optimal_params, 296e3),
    xytext=(optimal_params * 6, 296e3 * 4),
    fontsize=8.5, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.9),
)

# Gap arrow: from networking point down to optimal
ax.annotate(
    "",
    xy=(67e6, 296e3),
    xytext=(67e6, 67e6 * 20),
    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2),
)
ax.text(67e6 * 1.12, np.sqrt(296e3 * 67e6 * 20),
        "×4,500\ntokens gap", fontsize=8, color="#555555", va="center")

# ── Formatting ────────────────────────────────────────────────────────────────
ax.set_xlabel("Model parameters", fontsize=12)
ax.set_ylabel("Training tokens", fontsize=12)
ax.set_title("Scaling laws: compute-optimal frontier vs GraphMERT corpus sizes",
             fontsize=13, fontweight="bold")

ax.set_xlim(5e3, 5e11)
ax.set_ylim(1e4, 2e13)

# Custom tick labels
def fmt_axis(val, _):
    if val >= 1e12: return f"{val/1e12:.0f}T"
    if val >= 1e9:  return f"{val/1e9:.0f}B"
    if val >= 1e6:  return f"{val/1e6:.0f}M"
    if val >= 1e3:  return f"{val/1e3:.0f}K"
    return str(int(val))

from matplotlib.ticker import FuncFormatter
ax.xaxis.set_major_formatter(FuncFormatter(fmt_axis))
ax.yaxis.set_major_formatter(FuncFormatter(fmt_axis))

ax.grid(True, which="both", alpha=0.2, linewidth=0.6)
ax.grid(True, which="major", alpha=0.35, linewidth=0.8)

# Legend
legend_elements = [
    Line2D([0], [0], linestyle="--", color="#888888", linewidth=1.8,
           label="Chinchilla optimal (tokens = 20 × params)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7",
           markersize=8, label="General LMs"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="#ff6b35",
           markersize=8, label="GraphMERT — medical (125M tokens)"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#d62728",
           markersize=12, label="GraphMERT — networking (296k tokens)  ←"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
