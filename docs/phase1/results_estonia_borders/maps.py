"""Fields and borders of held-out test areas, as each arm of the two-head workflow draws them (REPORT.md section 3.5).

The places are those of the crop report's section 3.5 (``../results_estonia/maps.py``): three complete test blocks
of 4 x 4 tiles (8.96 km) and two single tiles (2.24 km, at full resolution), areas no model saw in training. This
script reads the stored test predictions of the two-head grid (``results/2026-11-09_borders_all_arms/EE_2021_rev/``:
``prob``, the border classes, and ``pred``, the crop head's map), refits nothing, and writes into ``figures/``:

* ``map_<place>_K<k>.png`` at K = 100, 20 and 5 %: the July true colour and the declared crops, then one panel per
  arm with **its own fields** (PRUE's interior regions grown into their rims, ``prue.parcels(..., "grow")``, over the
  whole tile with no test label), each coloured by the crop most of its pixels were predicted as by the arm's crop
  head. Black lines are the borders of the arm's fields, white lines the declared parcel borders: a white line
  across a coloured field is a declared border the arm missed, a black line inside a declared parcel a false cut.
  Land the arm calls background is left near-white. The panel title gives the arm's border F1 within 1 pixel over
  the place's tiles, from the counts the workflow scored;

and ``maps_tables.md`` beside this file: per place, arm and budget, the border F1 within 1 pixel and the number of
fields against the declared parcels.

Before drawing, the script recomputes each tile's border counts from the same fields and stops if they differ from
the cell's ``borders_per_chip.npz``, so every map shows exactly what was scored.

    poetry run python docs/phase1/results_estonia_borders/maps.py
"""

from __future__ import annotations

import gc
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gfm4agri.benchmark.parcel_metrics import read_parcels  # noqa: E402
from gfm4agri.borders import prue, scoring  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
_spec = importlib.util.spec_from_file_location("crop_maps", HERE.parent / "results_estonia" / "maps.py")
CM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CM)          # the crop report's places, palette, mosaics, legend and palette PNG

TWO = REPO / "results/2026-11-09_borders_all_arms/EE_2021_rev"
WORLDCOVER = REPO / "data/worldcover/EE_2021_rev/worldcover.npz"
OUT = HERE / "figures"
KS = [100, 20, 5]
ARMS = [("TerraMind v1 L, S2 + S1", "terramind_v1_large_s2s1"), ("THOR v1 L, S2 + S1", "thor_v1_large_s2s1"),
        ("Prithvi-EO-2.0 600M", "prithvi_eo_v2_600_tl"), ("TESSERA v2 M + PRUE's U-Net", "tessera_v2_medium"),
        ("AlphaEarth v1 + PRUE's U-Net", "alphaearth_v1"), ("Baseline: PRUE, S2 + S1", "unet_b7_s2s1")]
SURFACE, INK, INK_2, MUTED, NO_PARCEL = CM.SURFACE, CM.INK, CM.INK_2, CM.MUTED, CM.NO_PARCEL
DECLARED = "#ffffff"


def load_cell(arm: str, k: int, ids: set[str]) -> dict:
    """The border probabilities and crop map of the place tiles, and the stored per-tile border counts."""
    cell = TWO / arm / f"P{k}_draw0_seed0"
    with np.load(cell / "predictions_test.npz") as d:
        all_ids = [str(c) for c in d["ids"]]
        idx = [i for i, c in enumerate(all_ids) if c in ids]
        prob, pred, mask = d["prob"][idx], d["pred"][idx], d["mask"][idx]
    with np.load(cell / "borders_per_chip.npz") as b:
        counts = dict(zip([str(c) for c in b["ids"]], b["counts"], strict=True))
    sel = [all_ids[i] for i in idx]
    return {"prob": dict(zip(sel, prob, strict=True)), "pred": dict(zip(sel, pred, strict=True)),
            "mask": dict(zip(sel, mask, strict=True)), "counts": counts}


def fields_of(cell: dict, cid: str, parcels: np.ndarray, wc: np.ndarray, n: int) -> dict:
    """One tile's fields, their voted crop, their border, and its border counts (checked against the stored ones)."""
    seg = prue.parcels(cell["prob"][cid], "grow")
    mask = cell["mask"][cid].astype(np.int64)
    E, scored, true_border = scoring.regions(mask, parcels, wc)
    types = scoring.border_types(parcels, mask, wc, true_border)
    c = scoring.border_counts(seg, true_border, scored, E, types)["tol"][scoring.TOLERANCES.index(1)]
    if not np.array_equal(c, cell["counts"][cid]):
        raise SystemExit(f"{cid}: border counts {c.tolist()} differ from the stored {cell['counts'][cid].tolist()}")
    cls = scoring.vote(seg, cell["pred"][cid], n)
    return {"crop": np.where(seg > 0, cls[seg], -1), "border": prue.boundary_map(seg), "counts": c,
            "fields": int(len(np.unique(seg[seg > 0])))}


def _f1(counts: np.ndarray) -> float:
    tp_p, n_p, tp_r, n_r = (float(x) for x in counts)
    p, r = tp_p / max(n_p, 1), tp_r / max(n_r, 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def draw(crop: np.ndarray, border: np.ndarray, declared: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    img = np.empty(crop.shape + (3,))
    img[crop < 0] = to_rgb(NO_PARCEL)
    ok = crop >= 0
    img[ok] = rgb[crop[ok]]
    img[declared] = to_rgb(DECLARED)
    img[border] = to_rgb(INK)
    return img


def legend(ax, rgb: np.ndarray, names: list[str]) -> None:
    """Every crop, grouped as in the crop report, then the lines and the land the maps show."""
    from matplotlib.patches import Patch

    handles = []
    for g, _, members in CM.GROUPS:
        handles += [Patch(color="none", label=g)] + [Patch(color=rgb[names.index(m)], label=CM.SHORT[m])
                                                     for m in members]
    handles += [Patch(color="none", label=" "),
                Line2D([], [], color=INK, lw=2, label="border of the arm's fields"),
                Line2D([], [], color=MUTED, lw=2, label="declared parcel border (white)"),
                Patch(color=NO_PARCEL, label="background, or no declared parcel"),
                Patch(facecolor=NO_PARCEL, edgecolor=CM.HATCH, hatch="////", label="declared, crop not scored")]
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center", ncol=1, fontsize=9.5, frameon=False, handlelength=1.3,
                    labelspacing=0.32)
    for t, h in zip(leg.get_texts(), handles, strict=True):
        if isinstance(h, Patch) and h.get_facecolor()[3] == 0 and t.get_text().strip():
            t.set_fontweight("bold")
            t.set_color(INK_2)


def fig_place(pl: dict, k: int, tiles: dict, parcels: dict, bands: list[str], rgb: np.ndarray,
              names: list[str]) -> tuple[Path, dict]:
    b, win = pl["tiles"], pl["win"]
    tc = CM.stretch(CM.mosaic(b, {c: CM.true_colour(c, bands) for c in b.chip_id})[win])
    pid = CM.mosaic(b, parcels)[win]
    mask = CM.mosaic(b, tiles[ARMS[0][1]]["mask"], fill=-1)[win]
    declared = CM.mosaic(b, {c: prue.boundary_map(parcels[c]) for c in b.chip_id}, fill=False)[win]
    in_view = [c for c in b.chip_id if CM.mosaic(b, {x: np.full((CM.TILE, CM.TILE), x == c) for x in b.chip_id},
                                                 fill=False)[win].any()]

    fig = plt.figure(figsize=(21, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 5, left=0.01, right=0.995, top=0.84, bottom=0.01, hspace=0.13, wspace=0.03,
                          width_ratios=[1, 1, 1, 1, 0.95])
    fig.text(0.01, 0.962, pl["head"] + f"; K = {k} % of the training parcels labelled", fontsize=14, color=INK,
             fontweight="bold")
    fig.text(0.01, 0.9, "Never seen in training. Each arm's own fields, built from its border head over the whole "
             "tile with no test label, coloured by the crop its crop head gives most of their pixels. Black: the "
             "borders of its fields. White: the declared parcel borders;\na white line across a coloured field is a "
             "declared border missed. Near-white: land the arm calls background. Titles: border F1 within 1 pixel over "
             "the tiles shown, and the number of fields against the declared parcels.", fontsize=10.5, color=INK_2,
             linespacing=1.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tc, interpolation="nearest")
    CM.style(ax, "July 2021, true colour")
    ax = fig.add_subplot(gs[1, 0])
    ref = CM.reference(mask, pid)
    img = CM.paint(ref, rgb)
    img[declared] = to_rgb(DECLARED)
    ax.imshow(img, interpolation="nearest")
    n_declared = int(len(np.unique(pid[pid > 0])))
    CM.style(ax, f"Declared crops and borders (reference), {n_declared:,} parcels")

    stats = {}
    for j, (label, arm) in enumerate(ARMS):
        cell = tiles[arm]
        per = {c: fields_of(cell, c, parcels[c], WC[c], len(names)) for c in b.chip_id}
        crop = CM.mosaic(b, {c: per[c]["crop"] for c in b.chip_id}, fill=-1)[win]
        border = CM.mosaic(b, {c: per[c]["border"] for c in b.chip_id}, fill=False)[win]
        counts = np.sum([per[c]["counts"] for c in in_view], axis=0)
        n_fields = sum(per[c]["fields"] for c in in_view)
        f1 = _f1(counts)
        stats[arm] = (f1, n_fields, n_declared)
        ax = fig.add_subplot(gs[j // 3, 1 + j % 3])
        ax.imshow(draw(crop, border, declared, rgb), interpolation="nearest")
        CM.style(ax, f"{label}\nborder F1 {f1:.2f}, {n_fields:,} fields")
    legend(fig.add_subplot(gs[:, 4]), rgb, names)
    path = OUT / f"map_{pl['name']}_K{k}.png"
    CM.save(fig, path, 150 if pl["width"] == 4 else 110, rgb, tc)
    return path, stats


def tables(stats: dict, pls: list[dict]) -> str:
    L = ["# Tables of the maps of REPORT.md section 3.5", "", "Written by [maps.py](maps.py). One draw, one seed per "
         "cell; a place's numbers come from its tiles alone and illustrate the maps, they are not test scores.", "",
         "## Border F1 within 1 px, K = 100 / 20 / 5 %", "",
         "| Arm | " + " | ".join(p["name"].replace("_", " ") for p in pls) + " |", "|---" * (1 + len(pls)) + "|"]
    for label, arm in ARMS:
        L.append(f"| {label} | " + " | ".join(" / ".join(f"{stats[(p['name'], k)][arm][0]:.2f}" for k in KS)
                                              for p in pls) + " |")
    L += ["", "## Fields drawn, K = 100 / 20 / 5 %, against the declared parcels", "",
          "| Arm | " + " | ".join(f"{p['name'].replace('_', ' ')} ({stats[(p['name'], 100)][ARMS[0][1]][2]:,} "
                                  "declared)" for p in pls) + " |", "|---" * (1 + len(pls)) + "|"]
    for label, arm in ARMS:
        L.append(f"| {label} | " + " | ".join(" / ".join(f"{stats[(p['name'], k)][arm][1]:,}" for k in KS)
                                              for p in pls) + " |")
    return "\n".join(L) + "\n"


WC: dict = {}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((CM.CHIPS / "manifest.json").read_text())
    names = [c["name"] for c in manifest["classes"]]
    bands = manifest["imagery"]["band_names"]
    rgb = CM.class_colours(names)
    pls = CM.places(pd.read_csv(CM.SPLIT))
    ids = {c for p in pls for c in p["tiles"].chip_id}
    parcels = {c: read_parcels(CM.CHIPS / "chips", c) for c in ids}
    wc = np.load(WORLDCOVER)
    WC.update({c: wc[c] for c in ids})
    stats = {}
    for k in KS:
        tiles = {arm: load_cell(arm, k, ids) for _, arm in ARMS}
        for pl in pls:
            path, stats[(pl["name"], k)] = fig_place(pl, k, tiles, parcels, bands, rgb, names)
            print(path, {a: round(v[0], 3) for a, v in stats[(pl["name"], k)].items()}, flush=True)
        del tiles
        gc.collect()
    (HERE / "maps_tables.md").write_text(tables(stats, pls))
    print(HERE / "maps_tables.md")


if __name__ == "__main__":
    main()
