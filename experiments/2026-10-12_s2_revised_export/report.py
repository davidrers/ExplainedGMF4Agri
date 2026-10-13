"""Figure and tables of the re-export, from ``check.json`` and the two manifests, for REPORT.md.

    poetry run python experiments/2026-10-12_s2_revised_export/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHIPS = REPO / "data" / "eurocrops_chips"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 110,
})


def main() -> None:
    chk = json.loads((HERE / "check.json").read_text())
    old = json.loads((CHIPS / "EE_2021" / "manifest.json").read_text())["chips"]
    new = json.loads((CHIPS / "EE_2021_rev" / "manifest.json").read_text())["chips"]
    x = np.arange(1, 13)
    filled = np.array([chk["filled_share_mean"][m] for m in MONTHS])
    white = np.array([chk["white_pixels_sample"][m] for m in MONTHS], dtype=float)
    n_white = chk["white_pixels_sample"]["chips"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    for i, (label, color) in enumerate((("EE_2021, export screening", SLOT[0]), ("EE_2021_rev, REVISED", SLOT[1]))):
        axes[0].plot(x, filled[:, i], color=color, marker="o", ms=5, mec=SURFACE, label=label)
        axes[1].plot(x, white[:, i] / (n_white * 224 * 224) * 1e4, color=color, marker="o", ms=5, mec=SURFACE,
                     label=label)
    axes[0].set_title("Share of chip pixels gap-filled, all 7,402 chips")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title(f"White pixels per 10,000, {n_white} chips")
    axes[1].set_yscale("symlog", linthresh=1)
    for a in axes:
        a.set_xticks(x, MONTHS)
    axes[0].legend(loc="upper center", fontsize=8.5)
    fig.suptitle("Monthly composites of Estonia 2021 before and after the re-export", x=0.01, ha="left", fontsize=11)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "filled_and_white.png")
    plt.close(fig)

    def totals(chips, key):
        return sum(sum(c["imagery"].get(key) or [0]) for c in chips)

    unread_old = sum(len(c["imagery"].get("unreadable_items", [])) for c in old)
    unread_new = sum(len(c["imagery"].get("unreadable_items", [])) for c in new)
    never_old = sum(c["imagery"].get("never_observed_pixels", 0) for c in old)
    never_new = sum(c["imagery"].get("never_observed_pixels", 0) for c in new)
    groups = sorted({tuple(c["imagery"].get("window_bounds_3035") or c["bounds_3035"]) for c in new})
    sizes = np.bincount([c["imagery"].get("window_chips", 1) for c in new])
    lines = ["# Tables", "",
             "| | EE_2021 | EE_2021_rev |", "|---|---|---|",
             f"| Chips | {len(old):,} | {len(new):,} |",
             f"| Items read, summed over chip reports | {totals(old, 'items_per_month'):,} | {totals(new, 'items_per_month'):,} |",
             f"| Distinct overpasses, summed over chip reports | | {totals(new, 'overpasses_per_month'):,} |",
             f"| Chips with an unreadable item | {sum(bool(c['imagery'].get('unreadable_items')) for c in old)} "
             f"({unread_old} items) | {sum(bool(c['imagery'].get('unreadable_items')) for c in new)} ({unread_new} items) |",
             f"| Pixels never observed in the year | {never_old:,} | {never_new:,} |",
             f"| Read windows | {len(old):,}, one per chip | {len(groups):,}: " +
             ", ".join(f"{int(n)} chips in windows of {k}" for k, n in enumerate(sizes) if n and k) + " |",
             "", "| Month | Filled, EE_2021 | Filled, EE_2021_rev | Chips > 50 % filled, EE_2021 | EE_2021_rev | "
             f"White pixels in {n_white} chips, EE_2021 | EE_2021_rev |", "|---|---|---|---|---|---|---|"]
    for i, m in enumerate(MONTHS):
        h = chk["chips_over_half_filled"][m]
        lines.append(f"| {m} | {filled[i, 0]:.1%} | {filled[i, 1]:.1%} | {h[0]:,} | {h[1]:,} | "
                     f"{int(white[i, 0]):,} | {int(white[i, 1]):,} |")
    (HERE / "figures" / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
