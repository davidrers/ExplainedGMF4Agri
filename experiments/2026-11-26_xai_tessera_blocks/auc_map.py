"""The Estonian AUC map on the blocks of this run, from the model's own input.

    poetry run python experiments/2026-11-26_xai_tessera_blocks/auc_map.py [--workers 24]

**Fields.** Every labelled field with 20 interior pixels or more (``chain.chip_fields``) in the **training chips** of
``EE_2021_rev`` (split ``blocks4_buf1600_seed0__6b0eb4cb``), so no explained test field enters. A field's value is the
mean over its interior pixels of the input exactly as TESSERA gets it (``chain.read_chip``: the ten Sentinel-2 bands,
reflectance x 10,000; Sentinel-1 VV and VH in dB, averaged in dB like the mean field the methods replace the input
with). The crop-signature map (``experiments/2026-11-20_xai_tessera/reference``) came from another export of the
composites (``EE_2021_transfer``).

**Blocks**, as ``docs/research/xai/2026-11-26_window_cues/window_cues.py`` defines them (read, not changed): a band's
value over a three-month window is the mean of its three monthly values; NDVI (B08, B04), NDMI (B8A, B11) and EVI
(B08, B04, B02, reflectance 0 to 1) are computed month by month from the field means, then averaged over the window.

**AUC**, as the crop-signature study computes it (``separability.auc_columns``, the draw of ``species_cues``): each
species with 200 fields or more against a rest pool of the other species (up to 3,000 fields per species, the rest
balanced over the others), P(species > rest), folded to 0.5 to 1; a band group takes its best band.

Writes ``reference/ee_block_auc.csv`` (one row per species, scale month or window, period, band group or index) and
the field table to ``results/<name>/EE_2021_rev/auc_map/fields.parquet``; checks the monthly cells against the
crop-signature map.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CS = REPO / "docs/research/xai/2026-11-16_crop_signatures"
sys.path[:0] = [str(HERE), str(REPO / "src"), str(CS)]
import chain  # noqa: E402

CHIPS = REPO / "data/eurocrops_chips/EE_2021_rev"
MONTHS, FEAT = chain.MONTHS, chain.FEATURES
GROUPS = {g: idx for g, idx in chain.BAND_GROUPS.items()}
WINDOWS = chain.WINDOWS
WNAMES = [chain.window_name(w) for w in WINDOWS]
INDICES = ["NDVI", "NDMI", "EVI"]
CAP, MIN_FIELDS = 3000, 200                        # species_cues.CAP and MIN_FIELDS


def chip_means(chip: str) -> list[tuple]:
    d = chain.read_chip(CHIPS / "chips", chip)
    out = []
    for f in chain.chip_fields(d["mask"], d["parcels"]):
        x = d["X"][f["interior"]].astype(np.float64)
        with np.errstate(invalid="ignore"):
            out.append((chip, f["field"], f["cls"], len(f["interior"]), np.nanmean(x, 0).ravel().astype(np.float32)))
    return out


def vegetation_indices(x: np.ndarray) -> np.ndarray:
    """``(n, 12, 12)`` field means in TESSERA's order -> ``(n, 12, 3)`` NDVI, NDMI, EVI (window_cues' formulas)."""
    b = {k: x[..., FEAT.index(k)].astype(np.float64) / 10000.0 for k in ("B02", "B04", "B08", "B8A", "B11")}
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b["B08"] - b["B04"]) / (b["B08"] + b["B04"])
        ndmi = (b["B8A"] - b["B11"]) / (b["B8A"] + b["B11"])
        evi = 2.5 * (b["B08"] - b["B04"]) / (b["B08"] + 6.0 * b["B04"] - 7.5 * b["B02"] + 1.0)
    out = np.stack([ndvi, ndmi, evi], -1)
    out[~np.isfinite(out)] = np.nan
    return out


def windowed(m: np.ndarray) -> np.ndarray:
    """``(n, 12, k)`` -> ``(n, 10, k)``: three-month means (NaN where a month is missing)."""
    return np.stack([m[:, a:b].mean(1) for a, b in WINDOWS], 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    from gfm4agri.benchmark.segmentation_data import split_list
    from gfm4agri.pipeline.config import resolve_split
    from separability import auc_columns

    out_dir = chain.results_dir() / "auc_map"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "fields.parquet"
    names = json.loads((chain.CELL / "results.json").read_text())["class_names"]
    species, sidx = chain.species_of_classes(names)
    if not table.exists():
        train = split_list(CHIPS, "train", resolve_split(CHIPS, None)).read_text().split()
        rows, t0 = [], time.time()
        with ProcessPoolExecutor(args.workers) as pool:
            for k, r in enumerate(pool.map(chip_means, train, chunksize=8)):
                rows += r
                if (k + 1) % 500 == 0:
                    print(f"{k + 1} / {len(train)} chips, {len(rows):,} fields, {(time.time() - t0) / 60:.1f} min", flush=True)
        f = pd.DataFrame({"chip": [r[0] for r in rows], "field": [r[1] for r in rows], "cls": [r[2] for r in rows],
                          "n_interior": [r[3] for r in rows]})
        f = pd.concat([f, pd.DataFrame(np.stack([r[4] for r in rows]),
                                       columns=[f"{m}_{b}" for m in MONTHS for b in FEAT])], axis=1)
        f["species"] = [species[sidx[c]] for c in f.cls]
        f.to_parquet(table, index=False)
    f = pd.read_parquet(table)
    print(f"{len(f):,} training fields; per species: {f.species.value_counts().to_dict()}", flush=True)

    x = f[[f"{m}_{b}" for m in MONTHS for b in FEAT]].to_numpy(np.float64).reshape(len(f), 12, 12)
    vi = vegetation_indices(x)
    cols = {("month", "band"): x, ("month", "index"): vi, ("window", "band"): windowed(x), ("window", "index"): windowed(vi)}
    flat = np.concatenate([v.reshape(len(f), -1) for v in cols.values()], 1)

    rng = np.random.default_rng(0)
    lab = f.species.to_numpy()
    n = pd.Series(lab).value_counts()
    keep = sorted(n[n >= MIN_FIELDS].index)
    sp = {s: rng.choice(np.flatnonzero(lab == s), min(n[s], CAP), replace=False) for s in keep}
    rows = []
    for s in keep:
        others = [k for k in keep if k != s]
        per = max(CAP // len(others), 50)
        rest = np.concatenate([rng.choice(sp[k], min(len(sp[k]), per), replace=False) for k in others])
        raw = auc_columns(flat[sp[s]], flat[rest])
        off = 0
        for (scale, kind), v in cols.items():
            periods = MONTHS if scale == "month" else WNAMES
            k = v.shape[-1]
            a = raw[off:off + len(periods) * k].reshape(len(periods), k)
            off += len(periods) * k
            fold = np.maximum(a, 1 - a)
            for pi, per_name in enumerate(periods):
                if kind == "band":
                    for g, idx in GROUPS.items():
                        j = idx[int(np.nanargmax(fold[pi, idx]))]
                        rows.append((s, scale, per_name, "group", g, fold[pi, j], FEAT[j], a[pi, j], len(sp[s]), len(rest)))
                else:
                    for j, name in enumerate(INDICES):
                        rows.append((s, scale, per_name, "index", name, fold[pi, j], name, a[pi, j], len(sp[s]), len(rest)))
    auc = pd.DataFrame(rows, columns=["species", "scale", "period", "kind", "name", "auc", "best_band", "auc_signed",
                                      "n_a", "n_b"])
    (HERE / "reference").mkdir(exist_ok=True)
    auc.round(4).to_csv(HERE / "reference" / "ee_block_auc.csv", index=False)

    # check: the monthly band-group cells against the crop-signature Estonian map (another export, all splits)
    old = pd.read_csv(REPO / "experiments/2026-11-20_xai_tessera/reference/ee_auc_map.csv")
    from scipy.stats import spearmanr
    for s in keep:
        a = auc[(auc.species == s) & (auc.scale == "month") & (auc.kind == "group")]
        b = old[old.species == s]
        if b.empty:
            print(f"check {s}: no crop-signature map")
            continue
        m = a.merge(b, left_on=["period", "name"], right_on=["month", "group"], suffixes=("", "_cs"))
        print(f"check {s}: {len(m)} monthly cells, Spearman {spearmanr(m.auc, m.auc_cs).statistic:.2f}, "
              f"best here {a.loc[a.auc.idxmax(), 'period']} {a.loc[a.auc.idxmax(), 'name']} {a.auc.max():.2f}, "
              f"crop signatures {b.loc[b.auc.idxmax(), 'month']} {b.loc[b.auc.idxmax(), 'group']} {b.auc.max():.2f}",
              flush=True)
    print(f"wrote {HERE / 'reference' / 'ee_block_auc.csv'}")


if __name__ == "__main__":
    main()
