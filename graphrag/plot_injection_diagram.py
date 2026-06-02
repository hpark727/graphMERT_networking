"""
Visual diagram of seed KG injection into a chain graph.

Shows a concrete 128-token chunk with two head entities found,
each injected with a triple from the seed KG as leaf nodes.

Run: python3 graphrag/plot_injection_diagram.py
Output: graphrag/injection_diagram.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "injection_diagram.png"

# ── Layout constants ──────────────────────────────────────────────────────────
ROOT_Y      = 3.0      # y-position of root token row
LEAF_Y      = 1.0      # y-position of leaf token row
LEAF_STEP   = 0.58     # x-spacing between leaf tokens
ROOT_STEP   = 1.2      # x-spacing between root tokens
BOX_W       = 0.9      # root box width
BOX_H       = 0.48     # root box height
LEAF_W      = 0.50     # leaf box width
LEAF_H      = 0.40     # leaf box height

# ── Colours ───────────────────────────────────────────────────────────────────
C_ROOT_NORM  = "#dce8f7"   # normal root token
C_ROOT_HEAD  = "#f4a261"   # head entity root token
C_LEAF_FULL  = "#2a9d8f"   # leaf with real content
C_LEAF_PAD   = "#e8e8e8"   # padded leaf
C_EDGE       = "#555555"
C_REL        = "#e76f51"
C_TEXT       = "#1a1a1a"

def box(ax, x, y, w, h, text, fc, fontsize=8, bold=False, alpha=1.0, ec="#aaaaaa"):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.04",
                          fc=fc, ec=ec, linewidth=0.8, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=C_TEXT, fontweight="bold" if bold else "normal", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#888888", lw=1.0, style="->"):
    ax.annotate("", xy=(x2, y2 + LEAF_H/2 + 0.02),
                xytext=(x1, y1 - BOX_H/2 - 0.02),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

# ── Example sentence ─────────────────────────────────────────────────────────
# "TCP provides reliable data transfer . The router forwards the IP datagram ."
# We'll show ~10 root tokens, with head entities at positions 0 (tcp) and 6 (router)
#
# Seed KG triples injected:
#   tcp, provides, reliable data transfer
#   router, forwards, ip datagram

ROOT_TOKENS = [
    ("[CLS]",  False),
    ("tcp",    True),   # head entity → inject (tcp, provides, reliable data transfer)
    ("pro-",   False),
    ("vides",  False),
    ("reli-",  False),
    ("able",   False),
    ("router", True),   # head entity → inject (router, forwards, ip datagram)
    ("for-",   False),
    ("wards",  False),
    ("ip",     False),
    ("…",      False),  # rest of 128-token chunk
    ("[SEP]",  False),
]

INJECTIONS = {
    1: {  # tcp at index 1
        "relation": "provides",
        "leaves": ["reliable", "data", "trans-", "fer", "[PAD]", "[PAD]", "[PAD]"],
    },
    6: {  # router at index 6
        "relation": "forwards",
        "leaves": ["ip", "data-", "gram", "[PAD]", "[PAD]", "[PAD]", "[PAD]"],
    },
}

# ── Draw ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.5, len(ROOT_TOKENS) * ROOT_STEP + 1)
ax.set_ylim(-0.2, 4.8)
ax.axis("off")

# Dimension annotations
ax.annotate("", xy=(0, 4.5), xytext=(len(ROOT_TOKENS)*ROOT_STEP - 0.4, 4.5),
            arrowprops=dict(arrowstyle="<->", color="#999999", lw=1.0))
ax.text(len(ROOT_TOKENS)*ROOT_STEP/2 - 0.2, 4.65,
        "128 root tokens  (one text chunk)", ha="center", fontsize=9, color="#555555")

# Root token row label
ax.text(-0.3, ROOT_Y, "roots\n(text)", ha="right", va="center",
        fontsize=8.5, color="#444444", style="italic")

# Leaf row labels
ax.text(-0.3, LEAF_Y, "leaves\n(KG tail)", ha="right", va="center",
        fontsize=8.5, color="#444444", style="italic")

for i, (tok, is_head) in enumerate(ROOT_TOKENS):
    rx = i * ROOT_STEP

    # Root box
    fc = C_ROOT_HEAD if is_head else C_ROOT_NORM
    ec = C_REL if is_head else "#aaaaaa"
    lw = 1.6 if is_head else 0.8
    rect = FancyBboxPatch((rx - BOX_W/2, ROOT_Y - BOX_H/2), BOX_W, BOX_H,
                          boxstyle="round,pad=0.04",
                          fc=fc, ec=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(rx, ROOT_Y, tok, ha="center", va="center", fontsize=8.5,
            color=C_TEXT, fontweight="bold" if is_head else "normal", zorder=4)

    # Root-to-root sequential edge (chain)
    if i < len(ROOT_TOKENS) - 1:
        ax.annotate("", xy=((i+1)*ROOT_STEP - BOX_W/2 - 0.02, ROOT_Y),
                    xytext=(rx + BOX_W/2 + 0.02, ROOT_Y),
                    arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.8),
                    zorder=1)

    # Leaf nodes
    inj = INJECTIONS.get(i)
    if inj:
        leaves = inj["leaves"]
        n = len(leaves)
        leaf_span = (n - 1) * LEAF_STEP
        leaf_x0 = rx - leaf_span / 2

        # Relation label on the downward edge
        mid_x = rx
        mid_y = (ROOT_Y + LEAF_Y) / 2
        ax.annotate("", xy=(mid_x, LEAF_Y + LEAF_H/2 + 0.03),
                    xytext=(mid_x, ROOT_Y - BOX_H/2 - 0.03),
                    arrowprops=dict(arrowstyle="->", color=C_REL, lw=1.4,
                                    connectionstyle="arc3,rad=0.0"),
                    zorder=2)
        ax.text(mid_x + 0.18, mid_y, inj["relation"],
                fontsize=8, color=C_REL, fontstyle="italic", va="center",
                bbox=dict(fc="white", ec="none", pad=1.5))

        for j, leaf_tok in enumerate(leaves):
            lx = leaf_x0 + j * LEAF_STEP
            is_pad = leaf_tok == "[PAD]"
            fc_l = C_LEAF_PAD if is_pad else C_LEAF_FULL
            tc_l = "#aaaaaa" if is_pad else "white"
            rect_l = FancyBboxPatch((lx - LEAF_W/2, LEAF_Y - LEAF_H/2), LEAF_W, LEAF_H,
                                    boxstyle="round,pad=0.03",
                                    fc=fc_l, ec="#999999" if is_pad else C_LEAF_FULL,
                                    linewidth=0.7, zorder=3)
            ax.add_patch(rect_l)
            ax.text(lx, LEAF_Y, leaf_tok, ha="center", va="center",
                    fontsize=7.5, color=tc_l, zorder=4)

            # Leaf-to-leaf sequential edges
            if j < n - 1:
                ax.annotate("", xy=(lx + LEAF_STEP - LEAF_W/2 - 0.01, LEAF_Y),
                            xytext=(lx + LEAF_W/2 + 0.01, LEAF_Y),
                            arrowprops=dict(arrowstyle="-", color="#bbbbbb", lw=0.6),
                            zorder=1)
    else:
        # Empty leaf column — 7 padded leaves
        leaf_span = 6 * LEAF_STEP
        leaf_x0 = rx - leaf_span / 2
        # Draw connecting line from root to pad zone
        ax.plot([rx, rx], [ROOT_Y - BOX_H/2, LEAF_Y + LEAF_H/2 + 0.03],
                color="#dddddd", lw=0.7, zorder=1)
        for j in range(7):
            lx = leaf_x0 + j * LEAF_STEP
            # only draw first leaf pad box to avoid clutter on non-head roots
            if j == 0:
                rect_l = FancyBboxPatch((lx - LEAF_W/2, LEAF_Y - LEAF_H/2), LEAF_W, LEAF_H,
                                        boxstyle="round,pad=0.03",
                                        fc=C_LEAF_PAD, ec="#dddddd",
                                        linewidth=0.5, alpha=0.6, zorder=3)
                ax.add_patch(rect_l)
                ax.text(lx, LEAF_Y, "[PAD]×7", ha="center", va="center",
                        fontsize=6.5, color="#bbbbbb", zorder=4)
                break  # just show one collapsed pad block per non-head root

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(fc=C_ROOT_HEAD, ec=C_REL, label="head entity root token"),
    mpatches.Patch(fc=C_ROOT_NORM, ec="#aaaaaa", label="non-head root token"),
    mpatches.Patch(fc=C_LEAF_FULL, ec=C_LEAF_FULL, label="leaf: injected tail token"),
    mpatches.Patch(fc=C_LEAF_PAD, ec="#999999", label="leaf: [PAD] (no injection)"),
]
ax.legend(handles=legend_items, loc="upper right", fontsize=8.5,
          framealpha=0.95, edgecolor="#cccccc")

# ── Callout boxes ─────────────────────────────────────────────────────────────
ax.text(1 * ROOT_STEP, ROOT_Y + 0.6,
        "seed KG:\n(tcp, provides, reliable data transfer)",
        ha="center", va="bottom", fontsize=8, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f0", ec=C_REL, linewidth=0.8))

ax.text(6 * ROOT_STEP, ROOT_Y + 0.6,
        "seed KG:\n(router, forwards, ip datagram)",
        ha="center", va="bottom", fontsize=8, color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f0", ec=C_REL, linewidth=0.8))

ax.set_title(
    "Chain graph injection: seed KG triples attached as leaf nodes at head entity positions\n"
    "128 roots × 7 leaves = 1024 total tokens  |  non-head roots get 7 [PAD] leaves",
    fontsize=10.5, fontweight="bold", pad=12,
)

plt.tight_layout()
plt.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT}")
