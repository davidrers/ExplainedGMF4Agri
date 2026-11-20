"""Step R: the Estonian reference, the crop-signature analysis on the Estonian fields alone.

``docs/research/xai/2026-11-16_crop_signatures`` measured, per species and GAEZ zone, which month and band group
tell a species apart from the other species (AUC), and how much recall a random forest loses when part of the input
is hidden. Zone 21 pools nine countries, whose calendars differ (a median 1.35 months inside a zone). The model here
saw Estonia only, so its reference is the same analysis run on the Estonian fields of the transfer chip set
(``data/xai/crop_signatures_2021/fields_EE.parquet``, 65,122 fields), with the code of that folder imported as it
is (``species_cues.one_zone``). The zone-21 rows of its ``tables/species_reference.csv`` stay beside it.

TESSERA reads no atmospheric band, so the cues are read without the Atmospheric group (it stays in the forest's
input, as there).

    poetry run python experiments/2026-11-20_xai_tessera/reference_ee.py [--threads 16]

Writes ``reference/``: ``ee_species_reference.csv`` (one row per species), ``ee_auc_map.csv`` (species against the
rest, per month and band group), ``ee_pairs.csv`` (the cue of every pair of species), ``ee_importance.csv`` (the
forest's recall lost per hidden unit), ``ee_confusion.csv``, ``ee_classifier.csv``, ``ee_profiles.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
SIG = REPO / "docs/research/xai/2026-11-16_crop_signatures"
sys.path.insert(0, str(SIG))

OUT = HERE / "reference"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TESSERA_GROUPS = ["Visible", "Red-edge", "NIR", "SWIR", "VV", "VH"]


def verdict(a: float) -> str:
    return "clear" if a >= 0.90 else "partial" if a >= 0.75 else "no single cue"


def month_name(m: str) -> str:
    return MONTHS[int(m[1:]) - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    import species_cues as sc
    from common import FEATURES, indices
    from separability import INDEX_COLS

    OUT.mkdir(exist_ok=True)
    df = load(sc)
    rng = np.random.default_rng(0)
    spl, sow, grl = df.species.to_numpy(), df.sowing.to_numpy(), df.group.to_numpy()
    species = sc.draw(spl, rng, sc.CAP)
    forms = {}
    for s in species:
        for f in ("winter", "spring", "unspecified"):
            r = np.flatnonzero((spl == s) & (sow == f))
            if len(r) >= sc.MIN_FIELDS:
                forms[(s, f)] = rng.choice(r, min(len(r), sc.CAP), replace=False)
    groups = sc.draw(grl, rng, sc.CAP)
    keep = np.unique(np.concatenate([*species.values(), *forms.values(), *groups.values()]))
    pos = np.full(len(df), -1)
    pos[keep] = np.arange(len(keep))
    rows = {"species": {k: pos[v] for k, v in species.items()}, "forms": {k: pos[v] for k, v in forms.items()},
            "groups": {k: pos[v] for k, v in groups.items()}}
    part = df.iloc[keep]
    xz = np.concatenate([part[FEATURES].to_numpy(np.float32), indices(part)[INDEX_COLS].to_numpy(np.float32)], 1)
    n_sp = df.species.value_counts()
    meta = {"group_of": df.drop_duplicates("species").set_index("species").group.to_dict(),
            "fields": {k: int(n_sp[k]) for k in species},
            "form_share": {(s, f): round(float((sow[spl == s] == f).mean()), 3) for s, f in forms}}
    rest, pair, form_rest, form_pair, group_rest, group_pair, cls, confusion, imp = sc.one_zone(
        ("EE", xz, part.country.to_numpy(), part.block.to_numpy(), rows, meta, args.threads))
    prof = sc.profiles(df)

    # the maps: species against the rest, per month and band group TESSERA reads
    g = rest[(rest.kind == "group") & rest.name.isin(TESSERA_GROUPS)].copy()
    g["month"] = g.month.map(month_name)
    g = g.rename(columns={"name": "group"})[["species", "month", "group", "auc", "best_band", "auc_signed"]]
    g.to_csv(OUT / "ee_auc_map.csv", index=False)
    p = pair[(pair.kind == "group") & pair.name.isin(TESSERA_GROUPS)].copy()
    p["month"] = p.month.map(month_name)
    p = p.rename(columns={"name": "group"})
    best_pair = p.loc[p.groupby(["a", "b"]).auc.idxmax(), ["a", "b", "month", "group", "best_band", "auc",
                                                           "auc_signed"]]
    best_pair["a_is"] = np.where(best_pair.auc_signed > 0.5, "higher", "lower")
    best_pair["verdict"] = best_pair.auc.map(verdict)
    conf = confusion[confusion.true != confusion.predicted]
    share = conf.set_index(["true", "predicted"]).share
    best_pair["a_taken_for_b"] = [round(float(share.get((a, b), 0)), 3) for a, b in zip(best_pair.a, best_pair.b)]
    best_pair["b_taken_for_a"] = [round(float(share.get((b, a), 0)), 3) for a, b in zip(best_pair.a, best_pair.b)]
    best_pair.round(3).to_csv(OUT / "ee_pairs.csv", index=False)
    imp.to_csv(OUT / "ee_importance.csv", index=False)
    confusion.to_csv(OUT / "ee_confusion.csv", index=False, float_format="%.4f")
    cls.to_csv(OUT / "ee_classifier.csv", index=False)
    prof.to_csv(OUT / "ee_profiles.csv", index=False)

    # one row per species
    ndvi = df.assign(**indices(df)[[c for c in INDEX_COLS if c.endswith("_NDVI")]])
    out = []
    for s in species:
        m = g[g.species == s]
        best = m.loc[m.auc.idxmax()]
        im = imp[imp.species == s]
        win = im[im.unit == "window"].sort_values("recall_drop").iloc[-1]
        grp = im[(im.unit == "group") & (im.group != "Atmospheric")].sort_values("recall_drop").iloc[-1]
        sens = im[im.unit == "sensor"].set_index("group").recall_drop
        mon = im[im.unit == "month"].sort_values("recall_drop").iloc[-1]
        cf = conf[conf.true == s].sort_values("share").iloc[-1]
        peak = MONTHS[int(np.argmax([ndvi.loc[ndvi.species == s, f"M{k:02d}_NDVI"].median() for k in range(1, 13)]))]
        out.append({
            "species": s, "fields": meta["fields"][s], "sowing_mix": ", ".join(
                f"{f} {meta['form_share'][(s, f)]:.0%}" for f in ("winter", "spring", "unspecified")
                if (s, f) in meta["form_share"]),
            "peak": peak, "cue_month": best.month, "cue_group": best.group, "cue_band": best.best_band,
            "cue_auc": round(float(best.auc), 3), "cue_direction": "higher" if best.auc_signed > 0.5 else "lower",
            "verdict": verdict(best.auc),
            "strong_cells": int((m.auc >= 0.75).sum()), "null_cells": int((m.auc < 0.55).sum()),
            "hide_window": "-".join(month_name(x) for x in win.month.split("-")),
            "hide_window_drop": round(float(win.recall_drop), 3),
            "hide_month": month_name(mon.month), "hide_month_drop": round(float(mon.recall_drop), 3),
            "hide_band_group": grp.group, "hide_band_group_drop": round(float(grp.recall_drop), 3),
            "hide_sentinel2_drop": round(float(sens.get("Sentinel-2", np.nan)), 3),
            "hide_sentinel1_drop": round(float(sens.get("Sentinel-1", np.nan)), 3),
            "recall": float(cls.set_index("species").recall[s]),
            "confused_with": cf.predicted, "confused_share": round(float(cf.share), 3)})
    ref = pd.DataFrame(out)
    ref.to_csv(OUT / "ee_species_reference.csv", index=False)
    print(ref[["species", "fields", "peak", "cue_month", "cue_group", "cue_auc", "verdict", "hide_window",
               "hide_window_drop", "hide_band_group", "recall", "confused_with"]].to_string(index=False))


def load(sc) -> pd.DataFrame:
    from common import load_all

    df = load_all(countries=["EE"])
    print(f"{len(df):,} Estonian fields, {df.species.nunique()} species", flush=True)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    main()
