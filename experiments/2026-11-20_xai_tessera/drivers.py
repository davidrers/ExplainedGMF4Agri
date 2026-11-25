"""Main drivers of each correctly predicted field against the AUC map of its species (inference level), and the
species' mean explanation against the same map (model level).

    poetry run python experiments/2026-11-20_xai_tessera/drivers.py

**Fields.** Only those the model gets right: the stored prediction's species and the argmax of the explained score
(field-interior mean log p per species) both equal the field's species. Species with an Estonian AUC map only
(potatoes has none; orchards and vegetables have no correct field).

**AUC main drivers** (``reference/ee_auc_map.csv``, species against the rest, best band of each group): cells ranked
by AUC; months and band groups ranked by their best cell. Main drivers: the top ``K`` of each level.

**XAI main drivers** of a field: the units with the largest supporting attribution to its species (positive =
supports): cells from IG and occlusion (mean); months from SHAP, IG, occlusion (mean) and occlusion (drop); band
groups from SHAP, IG and occlusion (mean).

**Agreement**, per field: *hit*, the field's top driver is among the AUC main drivers; *overlap*, the share of the
field's top ``K`` that are AUC main drivers; *AUC gap*, best AUC of the species minus the AUC of the field's top
driver; *Spearman* of the whole ranking with the AUC ranking. **Baselines**: chance (``K / N`` for hit and
overlap), and the same field's drivers against the *other* species' AUC maps (mean over them): every map peaks in
summer, so chance alone is too easy.

Writes ``figures/drivers_inference.png``, ``figures/drivers_species.png`` and ``figures/drivers.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import chain  # noqa: E402
import report as R  # noqa: E402

MONTHS, GROUPS = chain.MONTHS, list(chain.BAND_GROUPS)
K = {"cell": 5, "month": 3, "band group": 2}
N = {"cell": 72, "month": 12, "band group": 6}
LEVELS = list(K)
COLOR = {"IG": "#eda100", "Occlusion (mean)": "#2a78d6", "Occlusion (drop)": "#eb6834", "SHAP": "#1baf7a"}


def auc_levels() -> dict[str, dict[str, np.ndarray]]:
    """Per species: the AUC of every cell (72, month-major), month (12, its best cell) and band group (6)."""
    a = pd.read_csv(HERE / "reference" / "ee_auc_map.csv")
    out = {}
    for s, m in a.groupby("species"):
        amap = m.pivot(index="month", columns="group", values="auc").reindex(index=MONTHS, columns=GROUPS).to_numpy()
        out[s] = {"cell": amap.ravel(), "month": np.nanmax(amap, 1), "band group": np.nanmax(amap, 0)}
    return out


def xai_levels(A: dict, n: int) -> dict[str, dict[str, np.ndarray]]:
    """Per level and method: ``(n, units)`` supporting attribution to the field's species (larger = drives it more)."""
    mu, du = A["occl_mean_units"], A["occl_drop_units"]
    igc = R.group_sum(A["ig_a"])                                                        # (n, 12, 6)
    col = lambda units, names: [units.index(u) for u in names]                         # noqa: E731
    return {
        "cell": {"IG": igc.reshape(n, -1),
                 "Occlusion (mean)": A["occl_mean_a"][:, col(mu, [f"cell|{m} {g}" for m in MONTHS for g in GROUPS])]},
        "month": {"SHAP": A["shap_m0_a"], "IG": A["ig_a"].sum(-1),
                  "Occlusion (mean)": A["occl_mean_a"][:, col(mu, [f"month|{m}" for m in MONTHS])],
                  "Occlusion (drop)": A["occl_drop_a"][:, col(du, [f"month|{m}" for m in MONTHS])]},
        "band group": {"SHAP": A["shap_g_a"], "IG": igc.sum(1),
                       "Occlusion (mean)": A["occl_mean_a"][:, col(mu, [f"band group|{g}" for g in GROUPS])]},
    }


def agreement(x: np.ndarray, r: np.ndarray, k: int) -> dict:
    """One field's (or one species') attribution ``x`` against one AUC ranking ``r``."""
    top_x = np.argsort(-x, kind="stable")[:k]
    top_r = set(np.argsort(-r, kind="stable")[:k].tolist())
    main = int(top_x[0])
    return {"hit": float(main in top_r), "overlap": len(set(top_x.tolist()) & top_r) / k,
            "gap": float(np.nanmax(r) - r[main]), "rho": R.spearman(x, r)}


def unit_name(level: str, i: int) -> str:
    if level == "cell":
        return f"{MONTHS[i // 6]} {GROUPS[i % 6]}"
    return MONTHS[i] if level == "month" else GROUPS[i]


def main() -> None:
    import matplotlib.pyplot as plt

    R.style()
    df, A = R.load(chain.results_dir() / "explain")
    n = len(df)
    maps = auc_levels()
    pred = np.asarray(A["base_all"])[:, R.K:].argmax(1)
    correct = df.correct_species.to_numpy(bool) & (pred == df.a.to_numpy())
    keep = correct & df.species.isin(list(maps)).to_numpy()
    sp = [s for s in R.species_in(df[keep]) if (keep & (df.species == s).to_numpy()).sum() >= 5]
    X = xai_levels(A, n)
    print(f"{correct.sum()} of {n} fields predicted correctly (stored and explained score agree on "
          f"{(df.correct_species.to_numpy(bool) == (pred == df.a.to_numpy())).mean():.3f} of fields); "
          f"{keep.sum()} of them in {len(sp)} species with an AUC map")

    # ---- inference level: every correct field against its species' map and against the other species' maps
    rows = []
    for i in np.flatnonzero(keep):
        s = df.species.iat[i]
        if s not in sp:
            continue
        for level in LEVELS:
            for meth, v in X[level].items():
                own = agreement(v[i], maps[s][level], K[level])
                oth = pd.DataFrame([agreement(v[i], maps[t][level], K[level]) for t in sp if t != s]).mean()
                rows.append({"field": i, "species": s, "level": level, "method": meth,
                             "main driver": unit_name(level, int(np.argmax(v[i]))),
                             **{f"{k}": own[k] for k in own}, **{f"{k} (other species)": oth[k] for k in own}})
    F = pd.DataFrame(rows)
    conf = np.exp(np.asarray(A["base_a"]))           # the field's species probability (geometric mean over its interior)
    F["p"] = conf[F.field.to_numpy()]

    def summary(g):
        return pd.Series({"fields": len(g), "hit": g.hit.mean(), "hit, other species' maps": g["hit (other species)"].mean(),
                          "overlap": g.overlap.mean(), "overlap, other species' maps": g["overlap (other species)"].mean(),
                          "AUC gap, median": g.gap.median(), "AUC gap, other species' maps": g["gap (other species)"].median(),
                          "gap ≤ 0.05": (g.gap <= 0.05).mean(),
                          "Spearman, median": g.rho.median(), "Spearman, other species' maps": g["rho (other species)"].median()})

    pooled = F.groupby(["level", "method"], sort=False).apply(summary, include_groups=False).reset_index()
    pooled.insert(2, "chance (hit, overlap)", [K[lv] / N[lv] for lv in pooled.level])
    per_sp = F.groupby(["level", "method", "species"], sort=False).apply(summary, include_groups=False).reset_index()

    # the species' AUC main drivers, and the fields' most frequent main drivers
    ref_rows, freq_rows = [], []
    for s in sp:
        sel = F[(F.species == s)]
        r = {"species": s}
        for level in LEVELS:
            top = np.argsort(-maps[s][level], kind="stable")[:K[level]]
            r[f"AUC top {K[level]} {level}s"] = ", ".join(f"{unit_name(level, int(j))} ({maps[s][level][j]:.2f})" for j in top)
        ref_rows.append(r)
        for level, meth in (("cell", "IG"), ("month", "SHAP"), ("band group", "SHAP")):
            vc = sel[(sel.level == level) & (sel.method == meth)]["main driver"].value_counts(normalize=True)
            freq_rows.append({"species": s, "level": level, "method": meth,
                              "most frequent main drivers (share of fields)": ", ".join(f"{k} {v:.0%}" for k, v in vc.head(3).items())})

    # ---- model level: each species' mean explanation over its correct fields, against the same maps
    mrows = []
    for s in sp:
        sel = keep & (df.species == s).to_numpy()
        for level in LEVELS:
            for meth, v in X[level].items():
                m = v[sel].mean(0)
                own = agreement(m, maps[s][level], K[level])
                oth = pd.DataFrame([agreement(m, maps[t][level], K[level]) for t in sp if t != s]).mean()
                mrows.append({"species": s, "level": level, "method": meth, "main driver": unit_name(level, int(np.argmax(m))),
                              **own, **{f"{k} (other species)": oth[k] for k in own}})
    Mo = pd.DataFrame(mrows)
    mpool = Mo.groupby(["level", "method"], sort=False).agg(
        species=("species", "size"), hit=("hit", "mean"), hit_other=("hit (other species)", "mean"),
        overlap=("overlap", "mean"), overlap_other=("overlap (other species)", "mean"),
        gap=("gap", "median"), rho=("rho", "median"), rho_other=("rho (other species)", "median")).reset_index()
    mpool.columns = ["level", "method", "species", "hit", "hit, other species' maps", "overlap",
                     "overlap, other species' maps", "AUC gap, median", "Spearman, median", "Spearman, other species' maps"]

    # ---- figure 1: inference level, pooled
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), gridspec_kw={"width_ratios": [2, 4, 3], "wspace": 0.55})
    for ax, level in zip(axes, LEVELS):
        g = pooled[pooled.level == level].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        ax.barh(y, g.hit, color=[COLOR[m] for m in g.method], height=0.55)
        ax.scatter(g["hit, other species' maps"], y, marker="o", s=36, facecolor="#fcfcfb", edgecolor="#3a3a37", lw=1.2,
                   zorder=3, label="the same fields against the other species' maps")
        ax.axvline(K[level] / N[level], color="#8a8a83", ls="--", lw=1, label="chance")
        for yy, h, o in zip(y, g.hit, g["hit, other species' maps"]):
            ax.text(max(h, o) + 0.04, yy, f"{h:.0%}", va="center", fontsize=8, color="#3a3a37")
        ax.set_yticks(y, g.method)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(f"{level}s (AUC top {K[level]} of {N[level]})", loc="left")
        ax.set_xlabel("share of correct fields")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Inference level: is a field's main driver one of its species' main AUC cues? "
                 f"({keep.sum()} correctly predicted fields, {len(sp)} species)", x=0.01, ha="left", y=1.08, fontsize=10)
    fig.savefig(R.FIG / "drivers_inference.png")
    plt.close(fig)

    # ---- figure 2: per species, hit share against its own map minus against the other species' maps
    cols = [(lv, m) for lv in LEVELS for m in X[lv]]
    H = np.array([[per_sp[(per_sp.level == lv) & (per_sp.method == m) & (per_sp.species == s)].hit.iat[0] for lv, m in cols]
                  for s in sp])
    B = np.array([[per_sp[(per_sp.level == lv) & (per_sp.method == m) & (per_sp.species == s)]["hit, other species' maps"].iat[0]
                   for lv, m in cols] for s in sp])
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(sp) + 1.6))
    im = ax.imshow(H - B, cmap=R.diverging(), vmin=-1, vmax=1, aspect="auto")
    for i in range(len(sp)):
        for j in range(len(cols)):
            ax.text(j, i, f"{H[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if abs(H[i, j] - B[i, j]) > 0.6 else "#1f1f1d")
    nf = [int((keep & (df.species == s).to_numpy()).sum()) for s in sp]
    ax.set_yticks(range(len(sp)), [f"{s} ({k})" for s, k in zip(sp, nf)])
    short = {"IG": "IG", "Occlusion (mean)": "Occl. mean", "Occlusion (drop)": "Occl. drop", "SHAP": "SHAP"}
    ax.set_xticks(range(len(cols)), [short[m] for _, m in cols], fontsize=7)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    for x in (1.5, 5.5):
        ax.axvline(x, color="#fcfcfb", lw=3)
    for x0, x1, lv in ((0, 1, "cells"), (2, 5, "months"), (6, 8, "band groups")):
        ax.text((x0 + x1) / 2, len(sp) - 0.35, lv, ha="center", va="top", fontsize=8, color="#3a3a37")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("hit share: own map minus other species' maps", fontsize=8)
    ax.set_title("Per species: share of correct fields whose main driver is a main AUC cue of their species "
                 "(colour: against the other species' maps)", loc="left", pad=22)
    fig.savefig(R.FIG / "drivers_species.png")
    plt.close(fig)

    # ---- confidence: only the most confident correct predictions
    KEY = [("cell", "IG"), ("month", "SHAP"), ("band group", "SHAP")]
    q95 = float(np.quantile(conf[keep], 0.95))
    rank_in_sp = F.drop_duplicates("field").set_index("field").groupby("species").p.rank(pct=True)
    F["top half of its species"] = F.field.map(rank_in_sp) > 0.5
    subsets = [("all correct", F.p >= 0), ("p >= 0.9", F.p >= 0.9), ("p >= 0.95", F.p >= 0.95),
               ("p >= 0.99", F.p >= 0.99), (f"top 5 % (p >= {q95:.3f})", F.p >= q95)]
    crow = []
    for name, sel in subsets:
        for level, meth in KEY:
            g = F[sel & (F.level == level) & (F.method == meth)]
            crow.append({"fields": name, "n": len(g), "species (5 fields or more)": int((g.species.value_counts() >= 5).sum()),
                         "level": level, "method": meth, "hit": g.hit.mean(), "hit, other species' maps": g["hit (other species)"].mean(),
                         "chance": K[level] / N[level], "AUC gap, median": g.gap.median()})
    Cf = pd.DataFrame(crow)
    srow = []
    for s in sp:
        for level, meth in KEY:
            g = F[(F.species == s) & (F.level == level) & (F.method == meth)]
            hi, lo = g[g["top half of its species"]], g[~g["top half of its species"]]
            c = g[g.p >= 0.95]
            srow.append({"species": s, "level": level, "method": meth, "fields": len(g), "hit, all correct": g.hit.mean(),
                         "fields p >= 0.95": len(c), "hit, p >= 0.95": c.hit.mean() if len(c) >= 5 else np.nan,
                         "hit, other maps, p >= 0.95": c["hit (other species)"].mean() if len(c) >= 5 else np.nan,
                         "hit, most confident half": hi.hit.mean(), "hit, least confident half": lo.hit.mean()})
    Sc = pd.DataFrame(srow)
    macro = Sc.groupby(["level", "method"], sort=False)[["hit, all correct", "hit, p >= 0.95", "hit, most confident half",
                                                         "hit, least confident half"]].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 0.33 * len(sp) + 1.8), sharey=True)
    for ax, (level, meth) in zip(axes, KEY):
        g = Sc[(Sc.level == level) & (Sc.method == meth)].reset_index(drop=True)
        y = np.arange(len(g))[::-1]
        for yy, a, b in zip(y, g["hit, all correct"], g["hit, p >= 0.95"]):
            if np.isfinite(b):
                ax.plot([a, b], [yy, yy], color="#c3c2b7", lw=2, zorder=1)
        ax.scatter(g["hit, all correct"], y, s=40, facecolor="#fcfcfb", edgecolor="#3a3a37", lw=1.2, zorder=2,
                   label="all correct fields")
        ax.scatter(g["hit, p >= 0.95"], y, s=40, color="#2a78d6", zorder=3, label="correct with p >= 0.95")
        ax.scatter(g["hit, other maps, p >= 0.95"], y, marker="|", s=90, color="#8a8a83", zorder=2,
                   label="p >= 0.95 against the other species' maps")
        ax.axvline(K[level] / N[level], color="#8a8a83", ls="--", lw=1)
        ax.set_yticks(y, [f"{s} ({n})" for s, n in zip(g.species, g["fields p >= 0.95"])])
        ax.set_xlim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_title(f"{meth}, {level}s (AUC top {K[level]})", loc="left")
    axes[1].set_xlabel("share of fields whose main driver is one of their species' main AUC cues")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Confident predictions only: all correct fields against those with p >= 0.95 (in brackets: how many; "
                 "fewer than 5 = no filled dot)", x=0.01, ha="left", y=1.04, fontsize=10)
    fig.savefig(R.FIG / "drivers_confidence.png")
    plt.close(fig)

    # ---- tables
    T = [("Inference level: every correctly predicted field, pooled over species (hit and overlap: shares; "
          "chance = K / N)", R.md(pooled)),
         ("Model level: each species' mean explanation over its correct fields, against the same maps",
          R.md(mpool)),
         ("Per species, inference level", R.md(per_sp[["level", "method", "species", "fields", "hit",
                                                        "hit, other species' maps", "overlap", "AUC gap, median",
                                                        "AUC gap, other species' maps", "Spearman, median"]])),
         ("The species' AUC main drivers (AUC in brackets)", R.md(pd.DataFrame(ref_rows))),
         ("The fields' most frequent main drivers", R.md(pd.DataFrame(freq_rows))),
         ("Model level per species", R.md(Mo[["species", "level", "method", "main driver", "hit", "overlap", "gap", "rho"]])),
         ("Confidence: only the most confident correct predictions, pooled (p = the field's species probability, "
          "geometric mean over its interior pixels)", R.md(Cf)),
         ("Confidence per species: all correct, p >= 0.95 (5 fields or more), and the most against the least "
          "confident half of each species' correct fields", R.md(Sc)),
         ("Confidence, the mean over species (each species weighs the same)", R.md(macro))]
    text = (f"Main drivers against the AUC map. {keep.sum()} correctly predicted fields of {len(sp)} species; "
            f"K = {K}.\n\n" + "\n\n".join(f"### {t}\n\n{b}" for t, b in T) + "\n")
    (R.FIG / "drivers.md").write_text(text)
    F.to_csv(HERE / "figures" / "drivers_fields.csv.gz", index=False, float_format="%.4f")
    print(R.md(pooled))
    print(R.md(mpool))


if __name__ == "__main__":
    main()
