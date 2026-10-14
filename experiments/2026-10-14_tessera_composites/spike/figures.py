"""Figures of the composite spike, from ``metrics.json``, into the experiment's ``figures/``.

* ``spike_probe_f1.png``: pooled probe Macro-F1 per representation, every seed and the mean, native series against
  our composites;
* ``spike_cosine.png``: per-pixel cosine to the native v1.1 embedding, as percentile ranges, against the
  reference scales within the native embedding;
* ``spike_per_class.png``: per-class probe F1 of each composite representation minus the native v1.1 one.

    poetry run python experiments/2026-10-14_tessera_composites/spike/figures.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from common import FIGURES, OUT  # noqa: E402

# The template's report.py palette: validated categorical slots, chart chrome, blue to red diverging ramp.
SLOT_NATIVE, SLOT_COMPOSITE = "#eb6834", "#2a78d6"
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": INK,
})

LABELS = {
    "store_v10": "v1.0, native series (current arm)", "store_v11": "v1.1 AWS, native series",
    "v11_aws": "v1.1 AWS, composites", "v11_mpc": "v1.1 MPC, composites",
    "v2_nano": "v2 Nano, composites", "v2_small": "v2 Small, composites",
    "v2_medium": "v2 Medium, composites", "v2_large": "v2 Large, composites",
}
NATIVE = {"store_v10", "store_v11"}


def probe_f1(m: dict) -> None:
    reps = list(m["probe"])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(reps) + 1.2))
    for i, r in enumerate(reps[::-1]):
        f = np.array(m["probe"][r]["macro_f1"])
        col = SLOT_NATIVE if r in NATIVE else SLOT_COMPOSITE
        ax.plot([f.min(), f.max()], [i, i], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(f, [i] * len(f), s=22, color=col, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.scatter([f.mean()], [i], s=70, marker="D", color=col, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(f"{f.mean():.3f}", (f.max(), i), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK_2, fontsize=8.5)
    ax.set_yticks(range(len(reps)), [LABELS[r] for r in reps[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Pooled Macro-F1, leave-one-block-out (5 blocks, 57 chips)")
    ax.scatter([], [], color=SLOT_NATIVE, s=40, marker="D", label="native series (store)")
    ax.scatter([], [], color=SLOT_COMPOSITE, s=40, marker="D", label="our monthly S2 + S1 composites")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, handletextpad=0.3)
    ax.set_title("Probe Macro-F1: native series against our monthly composites", pad=24)
    fig.text(0.01, 0.01, "Diamond: mean of 3 seeds; dots: each seed. MLP (512, 256) per pixel, 15 epochs.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "spike_probe_f1.png", dpi=170)
    plt.close(fig)


def cosine(m: dict) -> None:
    rows = [("v11_aws vs store_v11", "v1.1 AWS on composites vs native", SLOT_COMPOSITE),
            ("v11_mpc vs store_v11", "v1.1 MPC on composites vs native", SLOT_COMPOSITE),
            ("store_v11, same parcel", "reference: another pixel, same parcel", MUTED),
            ("store_v11, same class, other parcel", "reference: same class, other parcel", MUTED),
            ("store_v11, other class", "reference: another class", MUTED)]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for i, (key, label, col) in enumerate(rows[::-1]):
        p5, p25, p50, p75, p95 = m["cosine"][key]["pct"]
        ax.plot([p5, p95], [i, i], color=col, lw=1.5, solid_capstyle="round")
        ax.plot([p25, p75], [i, i], color=col, lw=7, solid_capstyle="butt", alpha=0.9)
        ax.scatter([p50], [i], s=46, color=SURFACE, edgecolor=col, linewidth=2, zorder=3)
        ax.annotate(f"{p50:.2f}", (p95, i), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK_2, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.axhline(2.5, color=BASELINE, lw=0.8)
    ax.set_xlim(-0.2, 1.05)
    ax.set_xlabel("Per-pixel cosine similarity (labelled pixels)")
    ax.set_title("How close is the composite embedding to the native one?")
    fig.text(0.01, 0.01, "Thin line p5 to p95, bar p25 to p75, ring the median (labelled).\nReferences: "
             "pairs of pixels within the native v1.1 embedding.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "spike_cosine.png", dpi=170)
    plt.close(fig)


def per_class(m: dict) -> None:
    classes = {int(k): v for k, v in m["classes"].items()}
    pix = {int(k): v for k, v in m["class_pixels"].items()}
    order = sorted(classes, key=lambda c: -pix[c])
    pos = {c: i for i, c in enumerate(sorted(classes))}
    reps = [r for r in m["probe"] if r not in NATIVE]
    native = np.array(m["probe"]["store_v11"]["per_class_f1"])
    D = np.array([[m["probe"][r]["per_class_f1"][pos[c]] - native[pos[c]] for r in reps] for c in order])
    fig, ax = plt.subplots(figsize=(7.6, 0.33 * len(order) + 1.6))
    lim = max(0.05, np.abs(D).max())
    ax.imshow(D, cmap=DIVERGING, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=7.5,
                    color=INK if abs(D[i, j]) < 0.6 * lim else SURFACE)
    ax.set_xticks(range(len(reps)), [LABELS[r].replace(", composites", "") for r in reps], rotation=30, ha="right")
    ax.set_yticks(range(len(order)), [f"{classes[c].replace('_', ' ')[:30]}  ({native[pos[c]]:.2f}, "
                                      f"{pix[c] / 1000:.1f}k px)" for c in order])
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Per-class F1, composites minus native v1.1")
    fig.text(0.01, 0.01, "Mean of 3 seeds. Rows by labelled pixels; in brackets the native v1.1 F1 and the pixel "
             "count.\nBlue: better on composites; red: worse.", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "spike_per_class.png", dpi=170)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    m = json.loads((OUT / "metrics.json").read_text())
    probe_f1(m)
    cosine(m)
    per_class(m)
    print("figures in", FIGURES)


if __name__ == "__main__":
    main()
