"""
Conceptual 'long tail' of Earth-observation demand.

Schematic figure (no real data) for the MSc proposal presentation:
  - Slide 2: the problem. A high viability threshold leaves the long tail unserved.
  - Slide 3: the lever (GFMs). The threshold drops, so much of the tail becomes viable.

Both versions are produced from one curve so they are visually consistent.
Exports PNG (300 dpi) and SVG for each version.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401  (kept for easy font swapping)

OUT_DIR = Path(__file__).resolve().parent

# Colourblind-safe palette (Okabe-Ito)
HEAD_COLOR = "#0072B2"   # blue
TAIL_COLOR = "#E69F00"   # orange
CURVE_COLOR = "#2b2b2b"
THRESH_COLOR = "#D55E00"  # vermillion

HEAD_FRAC = 0.15          # first 15% of the x-range is the "head"


def long_tail_curve(n=60, exponent=0.9):
    """Power-law decay over ranked use cases, normalised to a 0..1 y-range."""
    rank = np.arange(1, n + 1)
    y = 1.0 / np.power(rank, exponent)
    y = y / y.max()
    x = np.linspace(0.0, 1.0, n)
    return x, y


def make_figure(threshold_y, title, outname):
    x, y = long_tail_curve()
    head_mask = x <= HEAD_FRAC

    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Filled long-tail curve. Paint the whole area as tail first, then overlay
    # the head on top, so the head/tail boundary never shows a white gap.
    ax.fill_between(x, y, color=TAIL_COLOR, alpha=0.80, zorder=2,
                    label="Long tail")
    ax.fill_between(x, y, where=head_mask, color=HEAD_COLOR, alpha=0.95,
                    interpolate=True, zorder=2.5, label="Head")
    ax.plot(x, y, color=CURVE_COLOR, lw=1.6, zorder=3)

    # Viability threshold.
    ax.axhline(threshold_y, color=THRESH_COLOR, lw=1.8, ls="--", zorder=4)
    ax.text(0.995, threshold_y + 0.018, "Bespoke-pipeline viability threshold",
            ha="right", va="bottom", color=THRESH_COLOR, fontsize=11,
            fontstyle="italic")

    # Head annotation.
    ax.annotate(
        "Head: well funded, served\n"
        "continental cropland mapping,\n"
        "commercial farms,\n"
        "national land-use statistics",
        xy=(0.06, 0.55), xytext=(0.18, 0.74),
        fontsize=11, color=HEAD_COLOR, va="center",
        arrowprops=dict(arrowstyle="->", color=HEAD_COLOR, lw=1.4),
    )

    # Tail annotation.
    ax.annotate(
        "Long tail: many small-budget needs, unserved\n"
        "regional crop monitoring, cooperative agronomic advisory,\n"
        "insurance loss assessment, smallholder monitoring",
        xy=(0.55, y[np.searchsorted(x, 0.55)] + 0.02),
        xytext=(0.42, 0.40),
        fontsize=11, color="#9a6a00", va="center",
        arrowprops=dict(arrowstyle="->", color=TAIL_COLOR, lw=1.4),
    )

    # Axes styling.
    ax.set_xlabel("EO-derived products and services (ranked by funding)", fontsize=13)
    ax.set_ylabel("Funding available per use case  (€)", fontsize=13)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    png = OUT_DIR / f"{outname}.png"
    svg = OUT_DIR / f"{outname}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def main():
    outputs = []
    # Slide 2: the problem. High threshold, tail below it.
    outputs += make_figure(
        threshold_y=0.52,
        title="The long tail of Earth-observation demand",
        outname="long_tail_slide2_problem",
    )
    # Slide 3: the lever. Threshold drops, much of the tail rises above it.
    outputs += make_figure(
        threshold_y=0.16,
        title="Foundation models lower the threshold, opening the tail",
        outname="long_tail_slide3_lever",
    )
    for p in outputs:
        print("wrote", p)


if __name__ == "__main__":
    main()
