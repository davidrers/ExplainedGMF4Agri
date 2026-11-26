"""Figures and tables of the 18-country comparison: ``figures/`` and ``figures/tables.md``.

Run in the Poetry environment, after ``score.py``::

    poetry run python experiments/2026-11-26_delineation_countries/report.py

**No choice is made on this sample.** Every open model runs with the variant chosen on the Estonian validation chips
(``experiments/2026-11-26_open_delineation``), and FTW also with each country's own FTW windows, its published
setting. Our arms are their Estonian checkpoints. The per-country best FTW pair is shown only in the pair figure,
marked as chosen on the test chips.

**Intervals.** 95 % from a block bootstrap over the sample's 108 blocks (1,000 resamples, seed 0, the same resamples
for every model), and per country over its six blocks (wide, since six blocks are few).
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

import common as C  # noqa: E402
import score as SC  # noqa: E402

OUT = C.HERE / "figures"
SUBSET = "test"
TITLE = {"all": "all parcels", "no_grass": "grassland parcels set aside", "min25": "pieces under 0.25 ha set aside"}
EE_TRUTHS = ("all", "no_grass")                         # the Estonian run has these two
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})

#: key -> (name, label here, fields, colour, group, (label, fields) of the same setting in the Estonian experiment)
MODELS = {
    "unet_b7_s2s1": ("Ours: PRUE from ImageNet, S2 + S1 (baseline)", "arm_unet_b7_s2s1/P100", "grow", SLOTS[5],
                     "ours", ("arm/unet_b7_s2s1/P100", "grow")),
    "tessera_v2_medium": ("Ours: TESSERA v2 M + PRUE's U-Net", "arm_tessera_v2_medium/P100", "grow", SLOTS[3],
                          "ours", ("arm/tessera_v2_medium/P100", "grow")),
    "prithvi_eo_v2_600_tl": ("Ours: Prithvi-EO-2.0 600M", "arm_prithvi_eo_v2_600_tl/P100", "grow", SLOTS[1], "ours",
                             ("arm/prithvi_eo_v2_600_tl/P100", "grow")),
    "terramind_v1_large_s2s1": ("Ours: TerraMind v1 L, S2 + S1", "arm_terramind_v1_large_s2s1/P100", "grow",
                                SLOTS[0], "ours", ("arm/terramind_v1_large_s2s1/P100", "grow")),
    "thor_v1_large_s2s1": ("Ours: THOR v1 L, S2 + S1", "arm_thor_v1_large_s2s1/P100", "grow", SLOTS[2], "ours",
                           ("arm/thor_v1_large_s2s1/P100", "grow")),
    "prue_b7_own": ("FTW PRUE B7, each country's FTW windows", "prue_b7/own_windows", "cc", SLOTS[6], "open", None),
    "prue_b7": ("FTW PRUE B7, April + July (chosen on Estonia)", "prue_b7/AApr_BJul", "cc", SLOTS[6], "open",
                ("prue_b7/AApr_BJul", "cc")),
    "v2_multi_own": ("FTW v2, 2 windows, each country's FTW windows", "v2_multi/own_windows", "grow", SLOTS[6], "open",
                     None),
    "v2_single_mean": ("FTW v2 one-window, 6 dates averaged", "v2_single/season_mean", "grow", SLOTS[6], "open",
                       ("v2_single/season_mean", "grow")),
    "delany_v2": ("Delineate Anything v2, 1 date (June)", "delany_v2/official_Jun", "0.15", SLOTS[7], "open",
                  ("delany_v2/official_Jun", "0.15")),
    "delany_v2_fused": ("Delineate Anything v2, 6 dates fused", "delany_v2/official_fused6", "grow", SLOTS[7], "open",
                        ("delany_v2/official_fused6", "grow")),
    "delany_v1": ("Delineate Anything v1, 1 date (June)", "delany_v1/official_Jun", "0.15", SLOTS[7], "open",
                  ("delany_v1/official_Jun", "0.15")),
    "utae_paps": ("U-TAE + PaPs (PASTIS), 12 dates", "utae_paps/fold1_pastis_all12", "0.2", INK_2, "open",
                  ("utae_paps/fold1_pastis_all12", "0.2")),
}


def load() -> pd.DataFrame:
    df = pd.read_csv(SC.SCORES / "summary.csv", dtype={"fields": str})
    return df


def estonia() -> pd.DataFrame:
    return pd.read_csv(C.ROOT / "results" / "2026-11-26_open_delineation" / "EE_2021_rev" /
                       "scores" / "summary.csv", dtype={"fields": str})


def row(df, label, fields, subset, truth):
    r = df[(df.label == label) & (df.fields == fields) & (df.subset == subset) & (df.truth == truth)]
    return r.iloc[0] if len(r) else None


def models(df) -> list[dict]:
    out = []
    for k, (name, label, fields, col, grp, ee) in MODELS.items():
        if row(df, label, fields, SUBSET, "all") is not None:
            out.append({"key": k, "name": name, "label": label, "fields": fields, "colour": col, "group": grp,
                        "ee": ee})
    return out


# ------------------------------------------------------------------------------------------- block bootstrap

def _counts(label: str, fields: str):
    z = np.load(SC.SCORES / f"{label.replace('/', '__')}__test.npz")
    return [str(c) for c in z["ids"]], z[f"v_{fields}"]


def _metrics(s):
    tp, fp, fn, iou, tpp, npred, tpr, ntrue = s.T
    p, r = tpp / np.maximum(npred, 1), tpr / np.maximum(ntrue, 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return f1, iou / np.maximum(tp + fp / 2 + fn / 2, 1e-12)


def bootstrap(ms: list[dict], n: int = 1000, seed: int = 0) -> dict:
    """Per model, truth and scope (``all`` or a country): border F1 and PQ with 95 % block intervals."""
    from gfm4agri.borders import scoring as S

    d1 = 4 + 3 + len(S.FP_WC) + 4
    sc = pd.read_csv(C.HERE / "reference" / "sample_chips.csv")
    block = dict(zip(sc.chip_id, sc.country + "_" + sc.bx.astype(str) + "_" + sc.by.astype(str)))
    ids = C.split_ids("test")
    bl = pd.Index(sorted(set(block[c] for c in ids)))
    bidx = bl.get_indexer([block[c] for c in ids])
    bcc = np.array([b[:2] for b in bl])
    rng = np.random.default_rng(seed)
    draws = {"all": np.stack([np.bincount(rng.integers(0, len(bl), len(bl)), minlength=len(bl)) for _ in range(n)])}
    for cc in C.COUNTRIES:                                   # resample within the country's own blocks
        idx = np.nonzero(bcc == cc)[0]
        w = np.zeros((n, len(bl)))
        for i in range(n):
            np.add.at(w[i], rng.choice(idx, len(idx)), 1)
        draws[cc] = w
    out = {}
    for m in ms:
        cid, c = _counts(m["label"], m["fields"])
        pos = {k: i for i, k in enumerate(cid)}
        c = c[[pos[k] for k in ids]]
        res = {}
        for t, name in enumerate(SC.TRUTHS):
            per = np.zeros((len(bl), 8))
            np.add.at(per, bidx, np.concatenate([c[:, t, 0:4], c[:, t, d1:d1 + 4]], axis=1))
            for scope, W in draws.items():
                full = per[[True] * len(bl)] if scope == "all" else per[bcc == scope]
                f1, pq = _metrics(full.sum(0)[None])
                bf1, bpq = _metrics(W @ per)
                res[(name, scope)] = {"border_f1": float(f1[0]), "pq": float(pq[0]),
                                      "border_f1_ci": np.percentile(bf1, [2.5, 97.5]).tolist(),
                                      "pq_ci": np.percentile(bpq, [2.5, 97.5]).tolist(), "_f1": bf1, "_pq": bpq}
        out[m["key"]] = res
    out["_blocks"], out["_chips"] = len(bl), len(ids)
    return out


def paired(bs, a, b, truth, scope="all"):
    x, y = bs[a][(truth, scope)], bs[b][(truth, scope)]
    return {"border_f1": x["border_f1"] - y["border_f1"],
            "border_f1_ci": np.percentile(x["_f1"] - y["_f1"], [2.5, 97.5]).tolist(),
            "pq": x["pq"] - y["pq"], "pq_ci": np.percentile(x["_pq"] - y["_pq"], [2.5, 97.5]).tolist()}


# ------------------------------------------------------------------------------------------------------ figures

def fig_headline(ms, bs, ee):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 8.2), sharey=True)
    names = [m["name"] for m in ms][::-1]
    for j, truth in enumerate(SC.TRUTHS):
        for i, (metric, title, col) in enumerate((("border_f1", "Border F1 within 1 px", "border_f1_d1"),
                                                  ("pq", "Fields: panoptic quality, class-agnostic",
                                                   "pq_agnostic"))):
            ax = axes[i, j]
            for y, m in enumerate(ms[::-1]):
                r = bs[m["key"]][(truth, "all")]
                v, (lo, hi) = r[metric], r[f"{metric}_ci"]
                ax.barh(y, v, height=0.62, color=m["colour"], alpha=0.9 if m["group"] == "ours" else 0.75,
                        hatch=None if m["group"] == "ours" else "///", edgecolor=SURFACE, linewidth=0.6)
                ax.plot([lo, hi], [y, y], color=INK, lw=1)
                ax.text(hi + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
                if m["ee"] is not None and truth in EE_TRUTHS:
                    e = row(ee, m["ee"][0], m["ee"][1], "test_clean", truth)
                    if e is not None:
                        ax.plot(e[col], y, marker="|", color=INK, markersize=13, mew=2)
            ax.set_yticks(range(len(names)), names, fontsize=8.5)
            ax.set_xlim(0, 1.0 if metric == "border_f1" else 0.6)
            ax.set_title(f"{title}\n{TITLE[truth]}")
    fig.suptitle(f"18 countries, {bs['_chips']} chips in {bs['_blocks']} blocks; bars: point, whiskers: 95 % block "
                 "bootstrap; black tick: the same model on Estonia.\nSolid: our arms, trained on Estonia only; "
                 "hatched: open models, zero-shot, settings chosen on Estonia", x=0.01, ha="left", fontsize=10,
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "headline.png", dpi=150)
    plt.close(fig)


def fig_countries(ms, bs):
    keep = [m for m in ms if m["key"] not in ("prue_b7", "delany_v1", "v2_multi_own")]
    fig, axes = plt.subplots(3, 1, figsize=(13, 3 * (0.36 * len(keep) + 1.2)))
    for ax, truth in zip(axes, SC.TRUTHS):
        g = np.array([[bs[m["key"]][(truth, cc)]["border_f1"] for cc in C.COUNTRIES] for m in keep])
        im = ax.imshow(g, cmap="Blues", vmin=0.2, vmax=0.95, aspect="auto")
        for i in range(g.shape[0]):
            for k in range(g.shape[1]):
                ax.text(k, i, f"{g[i, k]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if g[i, k] > 0.62 else INK)
        ax.set_xticks(range(len(C.COUNTRIES)), C.COUNTRIES, fontsize=8)
        ax.set_yticks(range(len(keep)), [m["name"] for m in keep], fontsize=7.5)
        ax.grid(False)
        ax.set_title(f"Border F1 within 1 px by country, {TITLE[truth]}")
    fig.colorbar(im, ax=axes, shrink=0.7)
    fig.savefig(OUT / "countries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_estonia(ms, bs, ee):
    keep = [m for m in ms if m["ee"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 0.4 * len(keep) + 1.8), sharey=True)
    for ax, truth in zip(axes, EE_TRUTHS):
        for y, m in enumerate(keep[::-1]):
            e = row(ee, m["ee"][0], m["ee"][1], "test_clean", truth)
            v = bs[m["key"]][(truth, "all")]["border_f1"]
            ax.plot([e.border_f1_d1, v], [y, y], color=MUTED, lw=1.2)
            ax.plot(e.border_f1_d1, y, "o", color=SURFACE, mec=m["colour"], mew=2)
            ax.plot(v, y, "o", color=m["colour"])
            ax.text(max(e.border_f1_d1, v) + 0.012, y, f"{v - e.border_f1_d1:+.3f}", va="center", fontsize=8,
                    color=INK_2)
        ax.set_yticks(range(len(keep)), [m["name"] for m in keep][::-1], fontsize=8)
        ax.set_xlim(0.2, 1.0)
        ax.set_xlabel("border F1 within 1 px")
        ax.set_title(f"Estonia (open circle) against the 18 countries (filled), "
                     f"{'all parcels' if truth == 'all' else 'grassland aside'}", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "estonia_vs_countries.png", dpi=150)
    plt.close(fig)


def fig_temporal(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.grid(axis="y")
    for pre, keep, lab, col, ls, f in (("delany_v2/", "official_", "Delineate Anything v2", SLOTS[7], "-", "0.15"),
                                       ("delany_v1/", "official_", "Delineate Anything v1", SLOTS[7], "--", "0.15"),
                                       ("v2_single/", "W", "FTW v2, one window", SLOTS[6], ":", "grow")):
        for truth, mk in (("all", "o"), ("no_grass", "s")):
            ys = []
            for mo in C.SEASON:
                r = row(df, f"{pre}{keep}{C.MONTHS[mo]}", f, SUBSET, truth)
                ys.append(np.nan if r is None else r.border_f1_d1)
            ax.plot(range(6), ys, color=col, ls=ls, marker=mk, label=f"{lab}, {'all' if truth == 'all' else 'grass aside'}",
                    alpha=1 if truth == "all" else 0.55)
    ax.set_xticks(range(6), [C.MONTHS[m] for m in C.SEASON])
    ax.set_ylabel("border F1 within 1 px")
    ax.set_title("One date: which month")
    ax.legend(fontsize=6.5)
    for ax, pre, f, title in ((axes[1], "prue_b7/", "cc", "FTW PRUE B7"), ):
        grid = np.full((3, 3), np.nan)
        for i, a in enumerate((3, 4, 5)):
            for k, b in enumerate((6, 7, 8)):
                r = row(df, f"{pre}A{C.MONTHS[a]}_B{C.MONTHS[b]}", f, SUBSET, "no_grass")
                grid[i, k] = np.nan if r is None else r.border_f1_d1
        im = ax.imshow(grid, cmap="Blues", vmin=np.nanmin(grid) - 0.02, vmax=np.nanmax(grid))
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{grid[i, k]:.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if grid[i, k] > np.nanmean(grid) else INK)
        own = row(df, f"{pre}own_windows", f, SUBSET, "no_grass")
        pm = row(df, f"{pre}pairs_mean", f, SUBSET, "no_grass")
        ax.set_xticks(range(3), ["Jul", "Aug", "Sep"])
        ax.set_yticks(range(3), ["Apr", "May", "Jun"])
        ax.set_xlabel("window B")
        ax.set_ylabel("window A")
        ax.grid(False)
        ax.set_title(f"{title}, grassland aside: pairs\nown FTW windows {own.border_f1_d1:.3f}, "
                     f"nine pairs averaged {pm.border_f1_d1:.3f}", fontsize=9.5)
    ax = axes[2]
    ax.grid(axis="y")
    for truth, mk in (("all", "o"), ("no_grass", "s")):
        ys = [row(df, f"utae_paps/fold1_pastis_{ms}", "0.2", SUBSET, truth) for ms in ("pair2", "season6", "all12")]
        ax.plot([2, 6, 12], [np.nan if r is None else r.border_f1_d1 for r in ys], color=INK_2, marker=mk,
                label="all parcels" if truth == "all" else "grassland aside")
    for truth, mk in (("all", "o"), ("no_grass", "s")):
        a = row(df, "delany_v2/official_Jun_3class", "grow", SUBSET, truth)
        b = row(df, "delany_v2/official_fused6", "grow", SUBSET, truth)
        if a is not None and b is not None:
            ax.plot([1, 6], [a.border_f1_d1, b.border_f1_d1], color=SLOTS[7], marker=mk, ls="--",
                    label=f"Delineate Anything v2, June vs fused ({'all' if truth == 'all' else 'grass aside'})")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 6, 12], ["1", "2", "6", "12"])
    ax.set_xlabel("monthly composites read")
    ax.set_title("U-TAE + PaPs: 2, 6, 12 months; DelAny v2: 1 vs 6 fused")
    ax.legend(fontsize=7)
    fig.suptitle("Dates on the 18-country sample (no choice made on this sample)", x=0.01, ha="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "temporal.png", dpi=150)
    plt.close(fig)


def fig_examples(ms, pick=("FR", "ES", "IE", "FI")):
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    from gfm4agri.borders import prue

    keep = [m for m in ms if m["key"] in ("unet_b7_s2s1", "tessera_v2_medium", "prue_b7_own", "delany_v2",
                                          "utae_paps")]
    chosen = []
    for cc in pick:
        cands = [c for c in C.split_ids("test") if c.startswith(cc)]
        best, score = None, -1
        for c in cands[:: max(1, len(cands) // 25)]:
            p = read_parcels(C.CHIP_DIR, c)
            n = len(np.unique(p[p > 0]))
            share = float((C.read_mask(c) >= 0).mean())
            if 20 <= n <= 90 and share > score:
                best, score = c, share
        chosen.append(best or cands[0])
    segs = {}
    for m in keep:
        items = {it["id"]: it for it in SC.load_items(C.RESULTS / m["label"] / "predictions_test.npz")
                 if it["id"] in chosen}
        segs[m["key"]] = {c: SC.fields_of(items[c], m["fields"]) for c in chosen}
    cols = 2 + len(keep)
    fig, axes = plt.subplots(len(chosen), cols, figsize=(2.6 * cols, 2.75 * len(chosen)))
    rng = np.random.default_rng(0)
    for i, cid in enumerate(chosen):
        s2 = C.read_s2(cid)[6][C.RGB].astype(np.float32)
        rgb = np.clip(np.moveaxis(s2, 0, -1) / np.maximum(np.percentile(s2, 99, axis=(1, 2)), 1), 0, 1)
        parcels, mask = read_parcels(C.CHIP_DIR, cid), C.read_mask(cid)
        kind = np.where(np.isin(mask, SC.GRASS), 1, np.where(mask >= 0, 2, np.where(parcels > 0, 3, 0)))
        panels = [("July composite", None), ("declared: grassland green,\nother crops orange", parcels)]
        panels += [(m["name"].replace("Ours: ", "ours: ").split(",")[0], segs[m["key"]][cid]) for m in keep]
        for k, (title, seg) in enumerate(panels):
            ax = axes[i, k]
            ax.imshow(rgb if seg is None else 0.45 + 0.55 * rgb.mean(-1, keepdims=True).repeat(3, -1))
            if seg is not None:
                if k == 1:
                    lut = np.array([[1, 1, 1], to_rgb(SLOTS[2]), to_rgb(SLOTS[1]), to_rgb(MUTED)])
                    fill = lut[kind]
                else:
                    lut = rng.uniform(0.2, 0.95, (int(seg.max()) + 1, 3))
                    lut[0] = 1
                    fill = lut[seg]
                ax.imshow(fill, alpha=np.where(seg > 0, 0.5, 0.0))
                ax.imshow(np.where(prue.boundary_map(seg), 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1,
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(title, fontsize=8)
        axes[i, 0].set_ylabel(cid, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- tables

def tables(df, ms, bs, ee) -> str:
    L = ["# Tables of the 18-country comparison", "",
         "Generated by `report.py` from `results/2026-11-26_delineation_countries/EU18_2021_delin/scores/summary.csv`.",
         ""]
    for subset, title in (("test", "all 1,513 chips"), ("test_clean", "chips no FTW chip touches"),
                          ("not_ftw_countries", "BG, CZ, IE, IT (countries FTW does not cover)")):
        for truth in SC.TRUTHS:
            L += ["", f"## {title}, {TITLE[truth]}", "",
                  "| Model | Border P / R / F1 (1 px) | PQ / SQ / RQ (agnostic) | Fields predicted / true |",
                  "|---|---|---|---|"]
            for m in ms:
                r = row(df, m["label"], m["fields"], subset, truth)
                if r is None:
                    continue
                L.append(f"| {m['name']} | {r.border_precision_d1:.3f} / {r.border_recall_d1:.3f} / "
                         f"**{r.border_f1_d1:.3f}** | {r.pq_agnostic:.3f} / {r.sq_agnostic:.3f} / "
                         f"{r.rq_agnostic:.3f} | {int(r.predicted_parcels):,} / {int(r.true_parcels):,} |")
    L += ["", f"## 95 % block-bootstrap intervals, all chips ({bs['_chips']} chips, {bs['_blocks']} blocks)", "",
          "| Model | Border F1, all | PQ, all | Border F1, grassland aside | PQ, grassland aside | "
          "Border F1, small pieces aside | PQ, small pieces aside |", "|---|---|---|---|---|---|---|"]
    for m in ms:
        def f(t, k, m=m):
            r = bs[m["key"]][(t, "all")]
            return f"{r[k]:.3f} [{r[k + '_ci'][0]:.3f}, {r[k + '_ci'][1]:.3f}]"
        L.append(f"| {m['name']} | " + " | ".join(f(t, k) for t in SC.TRUTHS for k in ("border_f1", "pq")) + " |")
    L += ["", "## Paired differences against our baseline (PRUE from ImageNet, trained on Estonia)", "",
          "| Model minus baseline | Border F1, all | PQ, all | Border F1, grassland aside | PQ, grassland aside | "
          "Border F1, small pieces aside | PQ, small pieces aside |", "|---|---|---|---|---|---|---|"]
    for m in ms:
        if m["key"] == "unet_b7_s2s1":
            continue
        cells = []
        for t in SC.TRUTHS:
            p = paired(bs, m["key"], "unet_b7_s2s1", t)
            cells += [f"{p['border_f1']:+.3f} [{p['border_f1_ci'][0]:+.3f}, {p['border_f1_ci'][1]:+.3f}]",
                      f"{p['pq']:+.3f} [{p['pq_ci'][0]:+.3f}, {p['pq_ci'][1]:+.3f}]"]
        L.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    L += ["", "## Estonia against the 18 countries, border F1 within 1 px (Estonia: test chips no FTW chip touches)",
          "", "| Model | Estonia, all | 18 countries, all | change | Estonia, grass aside | 18 countries, grass aside "
          "| change |", "|---|---|---|---|---|---|---|"]
    for m in ms:
        if m["ee"] is None:
            continue
        cells = []
        for t in EE_TRUTHS:
            e = row(ee, m["ee"][0], m["ee"][1], "test_clean", t).border_f1_d1
            v = bs[m["key"]][(t, "all")]["border_f1"]
            cells += [f"{e:.3f}", f"{v:.3f}", f"{v - e:+.3f}"]
        L.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    for truth in SC.TRUTHS:
        L += ["", f"## Border F1 by country, {TITLE[truth]}", "",
              "| Model | " + " | ".join(C.COUNTRIES) + " |", "|---" * (len(C.COUNTRIES) + 1) + "|"]
        for m in ms:
            L.append(f"| {m['name']} | " + " | ".join(f"{bs[m['key']][(truth, cc)]['border_f1']:.2f}"
                                                   for cc in C.COUNTRIES) + " |")
    L += ["", "## Who leads in each country (border F1 within 1 px; all parcels / grassland aside / small pieces "
          "aside)", "",
          "| Country | best of ours | best open model | ours minus open |", "|---|---|---|---|"]
    ours = [m for m in ms if m["group"] == "ours"]
    opens = [m for m in ms if m["group"] == "open"]
    for cc in C.COUNTRIES:
        cells = []
        for t in SC.TRUTHS:
            bo = max(ours, key=lambda m: bs[m["key"]][(t, cc)]["border_f1"])
            bp = max(opens, key=lambda m: bs[m["key"]][(t, cc)]["border_f1"])
            cells.append((bo, bp, t))
        L.append(f"| {cc} | " + " / ".join(f"{b['name'].replace('Ours: ', '')} {bs[b['key']][(t, cc)]['border_f1']:.2f}"
                                          for b, _, t in cells) + " | "
                 + " / ".join(f"{p['name']} {bs[p['key']][(t, cc)]['border_f1']:.2f}" for _, p, t in cells) + " | "
                 + " / ".join(f"{bs[b['key']][(t, cc)]['border_f1'] - bs[p['key']][(t, cc)]['border_f1']:+.2f}"
                              for b, p, t in cells) + " |")
    L += ["", "## Recall by what lies across the border (1 px, all parcels), all chips", "",
          "| Model | same crop | other crop | undeclared farmland | non-farm land |", "|---|---|---|---|---|"]
    for m in ms:
        r = row(df, m["label"], m["fields"], SUBSET, "all")
        L.append(f"| {m['name']} | {r.recall_same_crop:.3f} | {r.recall_other_crop:.3f} | "
                 f"{r.recall_undeclared_farmland:.3f} | {r['recall_non-farm_land']:.3f} |")
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(exist_ok=True)
    df, ee = load(), estonia()
    ms = models(df)
    bs = bootstrap(ms)
    fig_headline(ms, bs, ee)
    fig_countries(ms, bs)
    fig_estonia(ms, bs, ee)
    fig_temporal(df)
    fig_examples(ms)
    (OUT / "tables.md").write_text(tables(df, ms, bs, ee))
    keep = {k: {f"{t}|{s}": {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for (t, s), v in r.items()}
            for k, r in bs.items() if not k.startswith("_")}
    (OUT / "bootstrap.json").write_text(json.dumps({"chips": bs["_chips"], "blocks": bs["_blocks"],
                                                    "models": keep}, indent=1))
    print("figures and tables in", OUT, "models:", [m["key"] for m in ms])


if __name__ == "__main__":
    main()
