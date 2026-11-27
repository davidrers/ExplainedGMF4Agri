"""Block for block: what the XAI methods find against the Estonian AUC map on the same blocks.

    poetry run python experiments/2026-11-26_xai_tessera_blocks/compare.py

**Fields.** Only fields predicted correctly by both the stored predictions (``sample/``) and the explained score
(the argmax of the field's species scores).

**XAI per block** (larger = drives the species more): occlusion, the fall of log p when the block is hidden (mean
replacement, the index swap for NDVI and NDMI; the windows also under drop); SHAP, the months' Shapley values summed
over each window, and the band groups' exact Shapley values; IG summed over the block's months and inputs, the index
part of each pair for NDVI and NDMI. The **consensus** ranking is the mean of the methods' within-field ranks (windows:
occlusion, SHAP, IG; properties and cells: occlusion, IG).

**AUC per block** (``reference/ee_block_auc.csv``, the species against the rest, training fields): a cell is a
window x property (band group by its best band, or index); a window takes its best property, a property its best
window. The **main cues** are the blocks within 0.02 AUC of the best, so that near ties count as one.

**Agreement**, per field: *hit*, its top block is a main cue; *Spearman* of its block values with the AUC; each also
against the other species' maps (mean over them), the baseline, since every map peaks in summer. A field is *diffuse*
at a level when its top block is under 10 % of its total effect (field score minus the mean field's). Intervals: 95 %
bootstrap over the fields of a species (2,000 draws).

Writes ``figures/`` and ``figures/tables.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import blocks as B  # noqa: E402
import chain  # noqa: E402

FIG = HERE / "figures"
K_CLS = 20
TIE = 0.02
DIFFUSE = 0.10
WN, GR, IX = B.WNAMES, list(B.GROUPS), list(B.INDEX_PAIRS)
PROPS = GR + IX
CELLS = [f"{w} {p}" for w in WN for p in PROPS]
LEVEL_UNITS = {"window": WN, "property": PROPS, "cell": CELLS}
METHODS = {"window": ["Occlusion", "SHAP", "IG", "Drop"], "property": ["Occlusion", "IG", "SHAP (groups)"],
           "cell": ["Occlusion", "IG"]}
CONSENSUS = {"window": ["Occlusion", "SHAP", "IG"], "property": ["Occlusion", "IG"], "cell": ["Occlusion", "IG"]}
COLOR = {"Consensus": "#3a3a37", "Occlusion": "#2a78d6", "SHAP": "#1baf7a", "SHAP (groups)": "#1baf7a", "IG": "#eda100",
         "Drop": "#eb6834"}
INK, INK2 = "#1f1f1d", "#55554f"
PLURAL = {"window": "windows", "property": "properties", "cell": "cells"}


def style():
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                         "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def md(df: pd.DataFrame, fmt: str = ".2f") -> str:
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if not np.isfinite(v) else format(v, fmt)
        return str(v)
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)])


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.allclose(a[ok], a[ok][0]) or np.allclose(b[ok], b[ok][0]):
        return np.nan
    return float(spearmanr(a[ok], b[ok]).statistic)


def boot(v: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    s = stat(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(stat(v)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load():
    names = json.loads((chain.CELL / "results.json").read_text())["class_names"]
    species, _ = chain.species_of_classes(names)
    rows, X = [], {lv: {m: [] for m in METHODS[lv]} for lv in METHODS}
    total, secs = [], []
    for f in sorted((chain.results_dir() / "explain").glob("*.npz")):
        if f.name.endswith(".part.npz"):
            continue
        d = dict(np.load(f))
        n = len(d["field"])
        r = np.arange(n)
        a = d["a_idx"]
        base_a = d["base"][r, a]
        kind, name = list(d["occl_kind"]), list(d["occl_name"])
        fall = base_a[None] - d["occl"][:, r, a]                                     # (blocks, n)
        get = lambda k, units: fall[[next(i for i, (kk, nn) in enumerate(zip(kind, name)) if kk == k and nn == u)  # noqa: E731
                                     for u in units]].T
        X["window"]["Occlusion"].append(get("window", WN))
        X["window"]["Drop"].append(get("window_drop", WN))
        X["property"]["Occlusion"].append(get("property", PROPS))
        X["cell"]["Occlusion"].append(get("cell", CELLS))
        sm = d["shap_months"][r, :, a]                                               # (n, 12)
        X["window"]["SHAP"].append(np.stack([sm[:, s:e].sum(1) for s, e in B.WINDOWS], 1))
        X["property"]["SHAP (groups)"].append(np.concatenate([d["shap_groups"][r, :, a], np.full((n, 2), np.nan)], 1))
        ig = d["ig"][:, 0]                                                           # (n, 12, 12)
        ix = d["ig_index"][:, 0]                                                     # (n, 12, 2)
        X["window"]["IG"].append(np.stack([ig[:, s:e].sum((1, 2)) for s, e in B.WINDOWS], 1))
        X["property"]["IG"].append(np.concatenate([np.stack([ig[:, :, f_].sum((1, 2)) for f_ in B.GROUPS.values()], 1),
                                                   ix.sum(1)], 1))
        cells = []
        for s, e in B.WINDOWS:
            cells += [ig[:, s:e][:, :, f_].sum((1, 2)) for f_ in B.GROUPS.values()]
            cells += [ix[:, s:e, j].sum(1) for j in range(len(IX))]
        X["cell"]["IG"].append(np.stack(cells, 1))
        sp = d["base"][:, K_CLS:]
        for i in range(n):
            rows.append({"chip": f.stem, "field": int(d["field"][i]), "species": species[a[i] - K_CLS],
                         "p": float(np.exp(base_a[i])), "explained_right": int(sp[i].argmax()) == a[i] - K_CLS,
                         "n_interior": int(d["n_interior"][i])})
        total.append(base_a - d["empty"][r, a])
        span = d["ig_f1"][:, 0] - d["ig_f0"][:, 0]
        secs.append({"chip": f.stem, "fields": n, **dict(zip(d["seconds_names"], d["seconds"])),
                     "shap_eff": float(np.abs(sm.sum(1) - (base_a - d["empty"][r, a])).max()),
                     "shap_groups_eff": float(np.abs(d["shap_groups"][r, :, a].sum(1) - (base_a - d["empty"][r, a])).max()),
                     "ig_comp": float(np.median(np.abs(ig.sum((1, 2)) - span) / np.maximum(np.abs(span), 1e-6)))})
    df = pd.DataFrame(rows)
    X = {lv: {m: np.concatenate(v, 0) for m, v in ms.items()} for lv, ms in X.items()}
    return df, X, np.concatenate(total), pd.DataFrame(secs)


def auc_maps() -> dict:
    a = pd.read_csv(HERE / "reference" / "ee_block_auc.csv")
    a = a[a.scale == "window"]
    out = {}
    for s, g in a.groupby("species"):
        cell = g.set_index(["period", "name"]).auc
        grid = np.array([[cell[(w, p)] for p in PROPS] for w in WN])                 # (10, 8)
        sign = g.set_index(["period", "name"]).auc_signed
        out[s] = {"window": grid.max(1), "property": grid.max(0), "cell": grid.ravel(), "grid": grid,
                  "evi": np.array([cell[(w, "EVI")] for w in WN]),
                  "dir": np.array([[sign[(w, p)] for p in PROPS] for w in WN])}
    return out


def consensus(X: dict, lv: str) -> np.ndarray:
    from scipy.stats import rankdata

    rk = [rankdata(X[lv][m], axis=1) / X[lv][m].shape[1] for m in CONSENSUS[lv]]
    return np.mean(rk, 0)


def chance(r: np.ndarray, x: np.ndarray | None = None) -> float:
    """The hit rate of a random top block: the share of the (available) blocks that are main cues."""
    ok = np.isfinite(r) if x is None else np.isfinite(x) & np.isfinite(r)
    return float((r[ok] >= np.nanmax(r[ok]) - TIE).mean())


def agree(x: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """hit, Spearman, AUC gap of one field's block values ``x`` against an AUC vector ``r``."""
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan, np.nan, np.nan
    top = int(np.nanargmax(np.where(ok, x, -np.inf)))
    best = np.nanmax(r[ok])
    return float(r[top] >= best - TIE), spearman(x[ok], r[ok]), float(best - r[top])


def main() -> None:
    import matplotlib.pyplot as plt

    style()
    FIG.mkdir(exist_ok=True)
    df, X, total, secs = load()
    maps = auc_maps()
    keep = df.explained_right.to_numpy() & df.species.isin(list(maps)).to_numpy()
    sp_list = [s for s in df.species.value_counts().index if s in maps and (keep & (df.species == s).to_numpy()).sum() >= 5]
    for lv in METHODS:
        X[lv]["Consensus"] = consensus(X, lv)
    T = []
    T.append(("Fields", f"{len(df)} explained, {int(df.explained_right.sum())} also right by the explained score; "
                        f"{int(keep.sum())} compared, in {len(sp_list)} species"))
    T.append(("Checks per chip (largest SHAP efficiency gap; median IG completeness error)", md(pd.DataFrame([{
        "chips": len(secs), "SHAP months, largest gap": secs.shap_eff.max(), "SHAP groups, largest gap": secs.shap_groups_eff.max(),
        "IG completeness, median over chips": secs.ig_comp.median(),
        "seconds per field": float(secs[[c for c in ("setup", "occlusion", "shap", "ig") if c in secs]].sum().sum() / secs.fields.sum())}]), ".4f")))

    # ---- per field
    rows = []
    for i in np.flatnonzero(keep):
        s = df.species.iat[i]
        for lv in METHODS:
            for m in ["Consensus"] + METHODS[lv]:
                x = X[lv][m][i]
                h, rho, gap = agree(x, maps[s][lv])
                oth = np.array([agree(x, maps[t][lv]) for t in sp_list if t != s])
                top = np.nanmax(x) if np.isfinite(x).any() else np.nan
                rows.append({"field": i, "species": s, "p": df.p.iat[i], "level": lv, "method": m, "hit": h, "rho": rho,
                             "gap": gap, "hit_other": np.nanmean(oth[:, 0]), "rho_other": np.nanmean(oth[:, 1]),
                             "top": LEVEL_UNITS[lv][int(np.nanargmax(np.where(np.isfinite(x), x, -np.inf)))],
                             "diffuse": bool(m not in ("Consensus",) and top < DIFFUSE * total[i])})
    F = pd.DataFrame(rows)
    F["d_rho"] = F.rho - F.rho_other
    F["d_hit"] = F.hit - F.hit_other

    # ---- per species verdict (consensus), with intervals
    vrows = []
    for s in sp_list:
        for lv in METHODS:
            g = F[(F.species == s) & (F.level == lv) & (F.method == "Consensus")]
            h, hl, hh = boot(g.hit.to_numpy())
            dr, drl, drh = boot(g.d_rho.to_numpy())
            r_, _, _ = boot(g.rho.to_numpy(), np.median)
            specific = drl > 0
            verdict = "agrees" if specific and h >= 0.5 else ("partly" if specific or h >= 0.5 else "does not agree")
            ch = chance(maps[s][lv])
            if h >= 0.5 and h < ch + 0.05:                 # the hit is no better than chance
                verdict += " (hit at chance)"
            vrows.append({"species": s, "level": lv, "fields": len(g), "verdict": verdict,
                          "hit": f"{h:.2f} [{hl:.2f}, {hh:.2f}]", "chance": ch, "hit, other maps": g.hit_other.mean(),
                          "Spearman (median)": r_, "Spearman own - other": f"{dr:+.2f} [{drl:+.2f}, {drh:+.2f}]",
                          "_h": h, "_dr": dr, "_drl": drl, "_drh": drh})
    V = pd.DataFrame(vrows)
    T.append(("Verdict per species (consensus of the methods; 95 % intervals over fields). agrees: the block ranking "
              "follows the species' own map more than the other species' maps (interval of the difference above 0) and "
              "the top block is a main cue for half the fields or more; partly: one of the two",
              md(V[[c for c in V.columns if not c.startswith("_")]])))

    # ---- the same verdict rule for every method
    mv = []
    for s in sp_list:
        row = {"species": s}
        for lv in METHODS:
            for m in METHODS[lv]:
                g = F[(F.species == s) & (F.level == lv) & (F.method == m)]
                h = g.hit.mean()
                _, lo, _ = boot(g.d_rho.to_numpy())
                v = "agrees" if lo > 0 and h >= 0.5 else ("partly" if lo > 0 or h >= 0.5 else "no")
                ch = chance(maps[s][lv], X[lv][m][0])
                row[f"{lv}: {m}"] = f"{v}{'*' if h >= 0.5 and h < ch + 0.05 else ''} ({h:.0%}; {ch:.0%})"
        mv.append(row)
    T.append(("The same verdict rule for each method (in brackets: share of fields whose top block is a main cue; "
              "chance, the share of the blocks that are main cues; * = the hit is within 5 points of chance). SHAP (groups) "
              "ranks the 6 band groups only, with the main cues re-ranked among them",
              md(pd.DataFrame(mv))))

    # ---- per method, pooled and per species
    def summ(g):
        return pd.Series({"fields": len(g), "hit": g.hit.mean(), "hit, other maps": g.hit_other.mean(),
                          "Spearman (median)": g.rho.median(), "own - other (mean)": g.d_rho.mean(),
                          "AUC gap (median)": g.gap.median(), "diffuse": g.diffuse.mean()})
    P = F.groupby(["level", "method"], sort=False).apply(summ, include_groups=False).reset_index()
    T.append(("Every method, pooled over species (hit: share of fields whose top block is a main cue; diffuse: top "
              "block under 10 % of the field's total effect)", md(P)))
    PS = F.groupby(["level", "method", "species"], sort=False).apply(summ, include_groups=False).reset_index()
    T.append(("Every method per species", md(PS)))
    conf = F[F.p >= 0.95].groupby(["level", "method"], sort=False).apply(summ, include_groups=False).reset_index()
    T.append(("Only predictions with p >= 0.95", md(conf)))

    # ---- model level: the species' mean explanation
    mrows = []
    for s in sp_list:
        sel = keep & (df.species == s).to_numpy()
        for lv in METHODS:
            for m in ["Consensus"] + METHODS[lv]:
                x = np.nanmean(X[lv][m][sel], 0)
                h, rho, gap = agree(x, maps[s][lv])
                oth = np.array([agree(x, maps[t][lv]) for t in sp_list if t != s])
                mrows.append({"species": s, "level": lv, "method": m,
                              "top block": LEVEL_UNITS[lv][int(np.nanargmax(np.where(np.isfinite(x), x, -np.inf)))],
                              "hit": h, "Spearman": rho, "Spearman, other maps": np.nanmean(oth[:, 1])})
    Mo = pd.DataFrame(mrows)
    T.append(("Model level: each species' mean explanation", md(Mo)))

    # ---- the AUC map itself
    arows = []
    for s in sp_list:
        g = maps[s]["grid"]
        w, p = np.unravel_index(np.argmax(g), g.shape)
        mw = [WN[k] for k in np.flatnonzero(maps[s]["window"] >= maps[s]["window"].max() - TIE)]
        mp = [PROPS[k] for k in np.flatnonzero(maps[s]["property"] >= maps[s]["property"].max() - TIE)]
        arows.append({"species": s, "best cell": f"{WN[w]} {PROPS[p]} ({g[w, p]:.2f})", "main windows": ", ".join(mw),
                      "main properties": ", ".join(mp), "main cells": int((g >= g.max() - TIE).sum()),
                      "best NDVI": maps[s]["grid"][:, PROPS.index("NDVI")].max(), "best EVI": maps[s]["evi"].max(),
                      "best NDMI": maps[s]["grid"][:, PROPS.index("NDMI")].max()})
    T.append(("The Estonian AUC map on the blocks (main = within 0.02 of the best)", md(pd.DataFrame(arows))))

    # ---- figure 1: verdict per species
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 0.36 * len(sp_list) + 1.6), sharey=True)
    for ax, lv in zip(axes, METHODS):
        g = V[V.level == lv].set_index("species").reindex(sp_list)
        y = np.arange(len(sp_list))[::-1]
        ax.axvline(0, color="#c3c2b7", lw=1)
        ax.hlines(y, g._drl, g._drh, color="#8a8a83", lw=2)
        ax.scatter(g._dr, y, s=46, zorder=3, c=["#2a78d6" if v == "agrees" else ("#86b6ef" if v == "partly" else "#e34948")
                                                for v in g.verdict])
        for yy, h in zip(y, g._h):
            ax.text(1.02, yy, f"{h:.0%}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, sp_list)
        ax.set_title(PLURAL[lv], loc="left")
        ax.set_xlabel("Spearman with own map minus other species' maps")
    fig.text(0.99, 0.98, "right margin: share of fields whose top block is a main cue", ha="right", fontsize=8, color=INK2)
    fig.suptitle("Does the model's block ranking follow its species' AUC map? (consensus of the methods, 95 % "
                 "intervals; blue = agrees, light blue = partly, red = does not)", x=0.01, ha="left", y=1.03, fontsize=10)
    fig.savefig(FIG / "verdict.png")
    plt.close(fig)

    # ---- figure 2: maps side by side
    n = len(sp_list)
    fig, axes = plt.subplots(n, 3, figsize=(11, 1.55 * n + 0.8))
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("sq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    div = LinearSegmentedColormap.from_list("dv", ["#8f2424", "#e34948", "#f0efec", "#3987e5", "#0d366b"])
    for r_, s in enumerate(sp_list):
        sel = keep & (df.species == s).to_numpy()
        panels = [("AUC (species vs rest)", maps[s]["grid"], seq, (0.5, 1)),
                  ("Occlusion, mean fall", np.nanmean(X["cell"]["Occlusion"][sel], 0).reshape(10, 8), div, None),
                  ("IG, mean", np.nanmean(X["cell"]["IG"][sel], 0).reshape(10, 8), div, None)]
        for c_, (t, m, cmap, lim) in enumerate(panels):
            ax = axes[r_, c_]
            if lim is None:
                v = np.nanmax(np.abs(m)) or 1
                lim = (-v, v)
            ax.imshow(m.T, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            tw, tp = np.unravel_index(np.nanargmax(m), m.shape)
            ax.plot(tw, tp, "o", color=INK, ms=3.5)
            ax.set_yticks(range(8), PROPS if c_ == 0 else [], fontsize=6.5)
            ax.set_xticks(range(10), [w.split("-")[0][0] + w.split("-")[1][0] for w in WN] if r_ == n - 1 else [], fontsize=6.5)
            if r_ == 0:
                ax.set_title(t, loc="left", fontsize=9)
            if c_ == 0:
                ax.set_ylabel(s, fontsize=8, rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("Window x property: the AUC map and the model's mean evidence per species (dot: the top block; "
                 "window labels: first letters of its months, e.g. MJ = May-Jul)", x=0.01, ha="left", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "maps.png")
    plt.close(fig)

    # ---- figure 3: window and property profiles per species
    for lv, units in (("window", WN), ("property", PROPS)):
        cols = 4
        rws = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rws, cols, figsize=(13, 2.5 * rws), squeeze=False)
        for k, s in enumerate(sp_list):
            ax = axes[k // cols, k % cols]
            sel = keep & (df.species == s).to_numpy()
            a = maps[s][lv]
            ax.bar(range(len(units)), (a - 0.5) / max(a.max() - 0.5, 1e-9), color="#e7e6e1", width=0.8, label="AUC (scaled)")
            for m in METHODS[lv]:
                v = np.nanmean(X[lv][m][sel], 0)
                ax.plot(range(len(units)), v / (np.nanmax(np.abs(v)) or 1), marker="o", ms=3, lw=1.4, color=COLOR[m], label=m)
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_xticks(range(len(units)), [u.split("-")[0][:3] if lv == "window" else u[:4] for u in units],
                          fontsize=6.5, rotation=45 if lv == "property" else 0)
            ax.set_title(f"{s} ({int(sel.sum())})", loc="left", fontsize=9)
        for k in range(n, rws * cols):
            axes[k // cols, k % cols].axis("off")
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=len(l), bbox_to_anchor=(0.5, -0.03))
        fig.suptitle(f"{PLURAL[lv].capitalize()}: the species' AUC (grey bars, 0.5 to its best) and each method's mean "
                     "evidence (scaled to its largest)", x=0.01, ha="left", y=1.0, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"{PLURAL[lv]}.png")
        plt.close(fig)

    (FIG / "tables.md").write_text("\n\n".join(f"### {t}\n\n{b}" for t, b in T) + "\n")
    F.to_csv(FIG / "fields.csv.gz", index=False, float_format="%.4f")
    print(md(V[[c for c in V.columns if not c.startswith("_")]]))


if __name__ == "__main__":
    main()
