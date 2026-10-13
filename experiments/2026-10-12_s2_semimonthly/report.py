"""Figures, tables and checks of the two chip sets this folder built, for REPORT.md.

Compares ``EE_2021_sample_24`` (24 half-month frames) with ``EE_2021_sample_rev`` (12 monthly frames) from their
manifests and per-chip reports, checks that they differ only in the temporal grid, and writes ``figures/frames.png``
and ``figures/tables.md``:

    poetry run python experiments/2026-10-12_s2_semimonthly/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHIPS = REPO / "data" / "eurocrops_chips"
S24, S12 = CHIPS / "EE_2021_sample_24", CHIPS / "EE_2021_sample_rev"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5,
    "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 110,
})


def checks(m24: dict, m12: dict) -> dict:
    ids24 = [c["chip_id"] for c in m24["chips"]]
    ids12 = [c["chip_id"] for c in m12["chips"]]
    r12 = {c["chip_id"]: c["imagery"] for c in m12["chips"]}
    bounds = {c["chip_id"]: c["bounds_3035"] for c in m12["chips"]}
    # The core export records a window only for groups of two or more chips; a lone chip's window is itself.
    window = lambda cid, rep: rep.get("window_bounds_3035") or bounds[cid]
    same_window = sum(window(c["chip_id"], c["imagery"]) == window(c["chip_id"], r12[c["chip_id"]])
                      for c in m24["chips"])
    same_screening = sum(c["imagery"].get("screening") == r12[c["chip_id"]].get("screening") for c in m24["chips"])
    split24 = sorted((S24 / "splits").resolve().glob("*/*_data.txt"))
    split12 = sorted((S12 / "splits").glob("*/*_data.txt"))
    same_split = [a.name for a in split24] == [b.name for b in split12] and all(
        a.read_text().split() == b.read_text().split() for a, b in zip(split24, split12))
    labels_equal = 0
    for cid in ids24:
        with rasterio.open(S24 / "chips" / f"{cid}.mask.tif") as a, rasterio.open(S12 / "chips" / f"{cid}.mask.tif") as b:
            labels_equal += bool(np.array_equal(a.read(), b.read()))
    bands24 = sum(rasterio.open(S24 / "chips" / f"{cid}_merged.tif").count == 288 for cid in ids24)
    return {"chips_24": len(ids24), "chips_12": len(ids12), "same_chip_list": sorted(ids24) == sorted(ids12),
            "same_split_lists": same_split, "same_screening": same_screening, "same_window": same_window,
            "labels_identical": labels_equal, "288_bands": bands24}


def main() -> None:
    m24 = json.loads((S24 / "manifest.json").read_text())
    m12 = json.loads((S12 / "manifest.json").read_text())
    names = m24["imagery"]["periods"]
    f24 = np.array([c["imagery"]["filled_share_per_period"] for c in m24["chips"]])
    o24 = np.array([c["imagery"]["clear_obs_mean_per_period"] for c in m24["chips"]])
    p24 = np.array([c["imagery"]["overpasses_per_period"] for c in m24["chips"]])
    f12 = np.array([c["imagery"]["filled_share_per_month"] for c in m12["chips"]])
    o12 = np.array([c["imagery"]["clear_obs_mean_per_month"] for c in m12["chips"]])
    p12 = np.array([c["imagery"]["overpasses_per_month"] for c in m12["chips"]])
    chk = checks(m24, m12)

    FIG = HERE / "figures"
    FIG.mkdir(exist_ok=True)
    x24 = np.arange(24) / 2 + 0.75          # half-month centres on a month axis: Jan1 at 0.75, Jan2 at 1.25
    x12 = np.arange(1, 13)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    for ax, (a12, a24, title) in zip(axes, ((f12, f24, "Share of chip pixels gap-filled"),
                                            (o12, o24, "Clear overpasses per pixel"),
                                            ((p12 == 0), (p24 == 0), "Share of chips with no overpass at all"))):
        ax.plot(x12, a12.mean(0), color=SLOT[0], marker="o", ms=5, mec=SURFACE, label="12 monthly frames")
        ax.plot(x24, a24.mean(0), color=SLOT[1], marker="o", ms=4, mec=SURFACE, label="24 half-month frames")
        ax.set_title(title)
        ax.set_xticks(x12, MONTHS)
    axes[0].set_ylim(0, 1.02)
    axes[2].set_ylim(0, 1.02)
    axes[0].legend(loc="upper center", fontsize=8.5)
    fig.suptitle("The 733 chips of EE_2021_sample, both under the REVISED screening; mean over chips",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "frames.png")
    plt.close(fig)

    pct = lambda v: f"{v:.0%}"
    rows = ["| Frame | Filled, mean | Clear overpasses per pixel | Chips with no overpass |", "|---|---|---|---|"]
    for i, n in enumerate(names):
        m = i // 2
        rows.append(f"| {n} | {pct(f24[:, i].mean())} | {o24[:, i].mean():.2f} | {int((p24[:, i] == 0).sum())} |")
        if i % 2 == 1:
            rows.append(f"| **{MONTHS[m]}, 12 frames** | **{pct(f12[:, m].mean())}** | **{o12[:, m].mean():.2f}** | "
                        f"**{int((p12[:, m] == 0).sum())}** |")
    any_empty = int((p24 == 0).any(1).sum())
    text = ["# Tables", "",
            "## Checks", "", "| Check | Result |", "|---|---|",
            *[f"| {k} | {v} |" for k, v in chk.items()], "",
            "## Frames", "",
            f"{any_empty} of {len(p24)} chips have at least one half month with no overpass; "
            f"{(p24 == 0).sum(1).mean():.2f} such half months per chip on average. "
            f"{int((p12 == 0).any(1).sum())} chips have a month with none.", "",
            *rows]
    (FIG / "tables.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
