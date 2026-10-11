"""Apply every screening variant to the downloaded acquisitions and write the measures.

Per chip, for each variant: the monthly per-pixel median composite (the export's rule, applied to one observation
per acquisition or per overpass), and per month over the labelled pixels the share gap-filled, the clear dates per
pixel, the white pixels and the NDVI dips of the composite. Per screening rule: the observations it removes from
``export``, compared with the same pixel's clean observations of that month (the ``revised`` composite).

Writes ``results/<this folder>/tables/`` and one ``composites/<chip>.npz`` per chip for the figures.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_dilation

from gfm4agri.chips.s2_monthly import REVISED, _fill_months, _merge_overpasses, _screen
from gfm4agri.chips.s2_monthly import BANDS as ALL_BANDS
from gfm4agri.data.sentinel import BOA_ADD_OFFSET, OFFSET_BASELINE, SCL_DROP

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = REPO / "data" / "eurocrops_chips" / "EE_2021"
RES = REPO / "results" / HERE.name
BANDS = ["B02", "B03", "B04", "B08", "B11"]
B = {b: i for i, b in enumerate(BANDS)}
GROW = (3, 8, 9, 10, 11)            # classes the buffer grows: shadow, cloud, cirrus, snow
WHITE = 0.25                        # observed and brighter than this in B02, B03 and B04: snow or cloud
DIP = 0.15                          # NDVI below both neighbouring months by more than this
VARIANTS = ["export", "export_merged", "revised", "revised_no_haze", "revised_keep7", "revised_27_near",
            "revised_keep2"]
RULES = ["class_2", "class_7", "buffer", "snow", "haze", "merge"]
#: Classes 2 and 7 split by distance to cloud, shadow and snow (5 px), to see where they are contaminated.
SPLITS = ["class_2_near", "class_2_far", "class_7_near", "class_7_far"]
SEASONS = {"Apr-Sep": range(4, 10), "Oct-Mar": [10, 11, 12, 1, 2, 3]}
D_B02 = np.linspace(-0.2, 1.0, 241)           # histogram bins of the blue difference, reflectance
D_NDVI = np.linspace(-1.5, 1.0, 251)          # and of the NDVI difference
D_NIR = np.linspace(-1.0, 2.0, 301)           # and of the near infrared, relative: B08 / clean B08 - 1


def rules(dn: np.ndarray, scl: np.ndarray, zero: np.ndarray) -> dict[str, np.ndarray]:
    """Per-acquisition flags of every screening component, ``(n, H, W)`` each.

    ``dn`` is after the BOA offset; ``zero`` marks pixels whose raw digital number is 0, the archive's nodata.
    """
    r = {b: dn[:, B[b]] / 10000.0 for b in BANDS}
    grow = np.isin(scl, GROW)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndsi = (r["B03"] - r["B11"]) / (r["B03"] + r["B11"])
    return {
        "nodata": (scl == 255) | (scl == 0) | zero,
        "export": np.isin(scl, SCL_DROP),
        "class_2": scl == 2,
        "class_7": scl == 7,
        "buffer": np.stack([binary_dilation(g, iterations=2) for g in grow]) & ~grow,
        "near5": np.stack([binary_dilation(g, iterations=5) for g in grow]),
        "snow": (ndsi > 0.15) & (r["B08"] > 0.11) & (r["B03"] > 0.1),
        "haze": r["B02"] - 0.5 * r["B04"] - 0.08 > 0,
    }


def variant_flags(f: dict) -> dict[str, tuple[np.ndarray, bool]]:
    base = f["export"] | f["buffer"] | f["snow"]
    return {
        "export": (f["export"], False),
        "export_merged": (f["export"], True),
        "revised": (base | f["class_2"] | f["class_7"] | f["haze"], True),
        "revised_no_haze": (base | f["class_2"] | f["class_7"], True),
        "revised_keep7": (base | f["class_2"] | f["haze"], True),
        "revised_27_near": (base | ((f["class_2"] | f["class_7"]) & f["near5"]) | f["haze"], True),
        # Added after the first results: class 7 dropped everywhere, class 2 kept.
        "revised_keep2": (base | f["class_7"] | f["haze"], True),
    }


def overpass_flag(flag: np.ndarray, nodata: np.ndarray, keys: list) -> np.ndarray:
    """Per item: True where any copy of its overpass that covers the pixel is flagged."""
    out = np.zeros_like(flag)
    for key in dict.fromkeys(keys):
        g = np.flatnonzero([k == key for k in keys])
        out[g] = ((flag[g] & ~nodata[g]).any(0))[None]
    return out


def ndvi(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x[..., B["B08"], :, :] - x[..., B["B04"], :, :]) / (x[..., B["B08"], :, :] + x[..., B["B04"], :, :])


def analyse_chip(chip_id: str) -> dict:
    z = np.load(RES / "acq" / f"{chip_id}.npz")
    meta = json.loads(str(z["meta"]))
    raw = z["spec"]
    scl_all = z["scl"]
    zero_all = (raw == 0).any(1)
    shift = np.array([BOA_ADD_OFFSET if m["baseline"] >= OFFSET_BASELINE else 0.0 for m in meta], np.float32)
    dn_all = raw.astype(np.float32) + shift[:, None, None, None]
    months = np.array([int(m["datetime"][5:7]) for m in meta])
    keys = [(m["platform"], m["datetime"]) for m in meta]
    with rasterio.open(ROOT / "chips" / f"{chip_id}.mask.tif") as src:
        lab = src.read(1) >= 0
    h, w = lab.shape

    comp = {v: np.full((12, len(BANDS), h, w), np.nan, np.float32) for v in VARIANTS}
    dates = {v: np.zeros((12, h, w), np.int16) for v in VARIANTS}
    removed_obs = []                  # per month: rule masks, exclusive masks and reflectance, for after the median
    rows, counts = [], []
    checked = False
    for m in range(1, 13):
        idx = np.flatnonzero(months == m)
        if not len(idx):
            continue
        dn, scl = dn_all[idx], scl_all[idx]
        mkeys = [keys[i] for i in idx]
        f = rules(dn, scl, zero_all[idx])
        data = ~f["nodata"]
        vf = variant_flags(f)
        if not checked:
            # The revised variant must be exactly REVISED of the core module.
            full = np.zeros((len(idx), len(ALL_BANDS), h, w), np.float32)
            for b in BANDS:
                full[:, ALL_BANDS.index(b)] = np.where(data, dn[:, B[b]], np.nan)
            nod, flg = _screen(full, np.where(scl == 255, np.nan, scl).astype(np.float32), REVISED)
            assert np.array_equal(nod | flg, f["nodata"] | vf["revised"][0]), chip_id
            checked = True
        for v, (flag, merge) in vf.items():
            vals = np.where(data[:, None], dn, np.nan)
            if merge:
                vals, bad = _merge_overpasses(vals, f["nodata"], flag & data, mkeys)
            else:
                bad = f["nodata"] | flag
            vals = np.where(bad[:, None], np.nan, vals)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                comp[v][m - 1] = np.nanmedian(vals, axis=0)
            if merge:
                dates[v][m - 1] = (~bad).sum(0)
            else:
                # Clear dates: distinct overpasses with at least one clear copy, comparable across variants.
                ok = ~bad
                dates[v][m - 1] = np.stack([ok[[i for i, k in enumerate(mkeys) if k == key]].any(0)
                                            for key in dict.fromkeys(mkeys)]).sum(0)
        # Rule membership of every observation export keeps (clear in its own copy under SCL_DROP).
        kept_export = data & ~f["export"]
        rev_item = vf["revised"][0]
        merge_rule = overpass_flag(rev_item, f["nodata"], mkeys) & ~rev_item
        member = {r: f[r] & kept_export for r in RULES if r != "merge"}
        member["merge"] = merge_rule & kept_export
        any_rule = np.zeros_like(kept_export)
        for r in RULES:
            any_rule |= member[r]
        exclusive = {r: member[r] & ~np.logical_or.reduce([member[o] for o in RULES if o != r]) for r in RULES}
        member["kept"] = kept_export & ~any_rule
        for r in ("class_2", "class_7"):
            member[f"{r}_near"], member[f"{r}_far"] = member[r] & f["near5"], member[r] & ~f["near5"]
            exclusive[f"{r}_near"], exclusive[f"{r}_far"] = exclusive[r] & f["near5"], exclusive[r] & ~f["near5"]
        removed_obs.append((m, member, exclusive, dn / 10000.0))
        for r in RULES + SPLITS:
            counts.append({"chip_id": chip_id, "month": m, "rule": r,
                           "removed": int(member[r][:, lab].sum()), "removed_only": int(exclusive[r][:, lab].sum()),
                           "export_clear": int(kept_export[:, lab].sum())})

    # Per-variant monthly measures over labelled pixels.
    for v in VARIANTS:
        filled = np.isnan(comp[v][:, 0])
        gf, _ = _fill_months(comp[v].copy())
        nd = ndvi(gf)
        vis = comp[v][:, [B["B02"], B["B03"], B["B04"]]] / 10000.0
        white = (np.nan_to_num(vis, nan=0).min(1) > WHITE) & ~filled
        for m in range(1, 13):
            obs = lab & ~filled[m - 1]
            dip = np.zeros_like(obs)
            if 1 < m < 12:
                with np.errstate(invalid="ignore"):
                    dip = obs & (nd[m - 1] < np.minimum(nd[m - 2], nd[m]) - DIP)
            rows.append({"chip_id": chip_id, "variant": v, "month": m, "labelled_px": int(lab.sum()),
                         "filled_share": float(filled[m - 1][lab].mean()),
                         "clear_dates": float(dates[v][m - 1][lab].mean()),
                         "white": int(white[m - 1][lab].sum()), "observed": int(obs.sum()),
                         "dip": int(dip.sum()), "dip_denominator": int(obs.sum()) if 1 < m < 12 else 0})

    # Differences of the observations each rule removes from the pixel's clean ones (the revised composite).
    ref = comp["revised"] / 10000.0
    ref_ndvi = ndvi(ref)
    hist = {}
    for m, member, exclusive, refl in removed_obs:
        season = next(s for s, ms in SEASONS.items() if m in ms)
        has_ref = lab & ~np.isnan(ref[m - 1, 0])
        d_b02 = refl[:, B["B02"]] - ref[m - 1, B["B02"]][None]
        d_ndvi = ndvi(refl) - ref_ndvi[m - 1][None]
        with np.errstate(invalid="ignore", divide="ignore"):
            d_nir = refl[:, B["B08"]] / ref[m - 1, B["B08"]][None] - 1.0
        for kind, masks in (("any", member), ("only", {**exclusive, "kept": member["kept"]})):
            for r, mask in masks.items():
                sel = mask & has_ref[None]
                hb, _ = np.histogram(d_b02[sel], bins=D_B02)
                hn, _ = np.histogram(d_ndvi[sel][np.isfinite(d_ndvi[sel])], bins=D_NDVI)
                hr, _ = np.histogram(d_nir[sel][np.isfinite(d_nir[sel])], bins=D_NIR)
                no_ref = int((mask & (lab & ~has_ref)[None]).sum())
                key = (kind, r, season)
                if key not in hist:
                    hist[key] = [np.zeros(len(D_B02) - 1, np.int64), np.zeros(len(D_NDVI) - 1, np.int64),
                                 np.zeros(len(D_NIR) - 1, np.int64), 0]
                hist[key][0] += hb
                hist[key][1] += hn
                hist[key][2] += hr
                hist[key][3] += no_ref

    out = RES / "composites"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{chip_id}.npz", lab=lab,
                        **{v: np.nan_to_num(comp[v], nan=0).astype(np.uint16) for v in VARIANTS})
    info = {"chip_id": chip_id, "items": len(meta), "overpasses": len(dict.fromkeys(keys)),
            "reprocessed_items": int(sum(m["baseline"] >= OFFSET_BASELINE for m in meta)),
            "unreadable": len(json.loads(str(z["unreadable"])))}
    return {"rows": rows, "counts": counts, "hist": hist, "info": info}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    chips = sorted(p.stem for p in (RES / "acq").glob("EE_*.npz") if not p.stem.endswith(".tmp"))
    print(f"{len(chips)} chips", flush=True)
    rows, counts, infos, hist = [], [], [], {}
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(analyse_chip, c): c for c in chips}
        for fut in as_completed(futs):
            r = fut.result()
            rows += r["rows"]
            counts += r["counts"]
            infos.append(r["info"])
            for k, (hb, hn, hr, nr) in r["hist"].items():
                if k not in hist:
                    hist[k] = [hb.copy(), hn.copy(), hr.copy(), nr]
                else:
                    hist[k][0] += hb
                    hist[k][1] += hn
                    hist[k][2] += hr
                    hist[k][3] += nr
            print(f"{futs[fut]} done", flush=True)
    tab = RES / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(tab / "month_metrics.parquet")
    pd.DataFrame(counts).to_parquet(tab / "removed_counts.parquet")
    pd.DataFrame(infos).to_csv(tab / "chips.csv", index=False)
    keys = list(hist)
    np.savez(tab / "removed_hist.npz", keys=np.array(["|".join(k) for k in keys]),
             b02=np.stack([hist[k][0] for k in keys]), ndvi=np.stack([hist[k][1] for k in keys]),
             nir=np.stack([hist[k][2] for k in keys]), no_ref=np.array([hist[k][3] for k in keys]),
             b02_edges=D_B02, ndvi_edges=D_NDVI, nir_edges=D_NIR)
    print("written", tab)


if __name__ == "__main__":
    main()
