"""Figures and tables of the XAI experiment, from ``results/<name>/EE_2021_rev/{step0,explain,sanity}/``.

    poetry run python experiments/2026-11-20_xai_tessera/report.py [--explain <dir>] [--sanity <dir>]

Every quantity is a change of the field's species log-probability (A: its own species, B: its rival). Occlusion
gives the fall of A's score when a group is replaced (positive = the group supported A); SHAP and IG give signed
contributions relative to the mean field, which add up to f(x) − f(mean field). Species profiles average the fields
the model got right at species level; the misclassified ones enter the pairs (Q3). Writes ``figures/`` and
``figures/tables.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import chain
from chain import BAND_GROUPS, MONTHS, WINDOWS, window_name

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
K = 20
GROUPS = list(BAND_GROUPS)
METHOD_COLOR = {"occlusion_mean": "#2a78d6", "occlusion_drop": "#eb6834", "shap": "#1baf7a", "ig": "#eda100"}
METHOD_LABEL = {"occlusion_mean": "Occlusion (mean)", "occlusion_drop": "Occlusion (drop)", "shap": "SHAP",
                "ig": "Integrated Gradients"}
GRAY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"
S2_NAME = {"RED": "B04", "BLUE": "B02", "GREEN": "B03", "NIR_BROAD": "B08", "NIR_NARROW": "B8A", "RED_EDGE_1": "B05",
           "RED_EDGE_2": "B06", "RED_EDGE_3": "B07", "SWIR_1": "B11", "SWIR_2": "B12", "VV": "VV", "VH": "VH"}
SPECIES_ORDER = ["grassland", "forage legumes", "wheat", "barley", "rye", "oats", "rapeseed", "dry pulses",
                 "buckwheat", "potatoes", "maize", "vegetables", "orchards"]


# ------------------------------------------------------------------------------------------------------ loading

def load(explain_dir: Path) -> tuple[pd.DataFrame, dict]:
    """One row per field, and the method arrays stacked over fields (A's and B's scores taken out)."""
    names = json.loads((chain.CELL / "results.json").read_text())["class_names"]
    species, sidx = chain.species_of_classes(names)
    sample = pd.read_parquet(HERE / "sample" / "fields.parquet")
    rows, arr = [], {}

    def add(k, v):
        arr.setdefault(k, []).append(v)

    for f in sorted(explain_dir.glob("*.npz")):
        if f.name.endswith(".part.npz"):
            continue
        d = dict(np.load(f, allow_pickle=False))
        n = len(d["field"])
        r = np.arange(n)
        a, b = d["a_idx"], d["b_idx"]
        for i in range(n):
            rows.append({"chip": f.stem, "field": int(d["field"][i]), "cls": int(d["cls"][i]),
                         "a": int(a[i] - K), "b": int(b[i] - K), "n_pixels": int(d["n_pixels"][i])})
        base_a, base_b = d["base"][r, a], d["base"][r, b]
        add("base_a", base_a)
        add("base_b", base_b)
        add("empty_a", d["empty"][r, a])
        add("empty_b", d["empty"][r, b])
        add("base_all", d["base"])
        for op in ("drop", "mean"):
            add(f"occl_{op}_a", base_a[None] - d[f"occl_{op}"][:, r, a])          # (units, n)
            add(f"occl_{op}_ab", (base_a - base_b)[None] - (d[f"occl_{op}"][:, r, a] - d[f"occl_{op}"][:, r, b]))
        add("occl_chip_a", d["chip_base"][r, a][None] - d["occl_chip"][:, r, a])
        for h in range(3):
            add(f"shap_m{h}_a", d["shap_months"][h][r, :, a])
        add("shap_m_ab", d["shap_months"][0][r, :, a] - d["shap_months"][0][r, :, b])
        add("shap_g_a", d["shap_groups"][r, :, a])
        add("shap_g_ab", d["shap_groups"][r, :, a] - d["shap_groups"][r, :, b])
        add("ig_a", d["ig"][:, 0])
        add("ig_ab", d["ig"][:, 0] - d["ig"][:, 1])
        add("ig_grad_a", d["ig_grad"][:, 0])
        add("ig_f0", d["ig_f0"])
        add("ig_f1", d["ig_f1"])
        add("ig_check", d["ig_check"][:, 0] if "ig_check" in d else np.full(d["ig"][:, 0].shape, np.nan))
        add("ig_check_f", np.stack([d["ig_f0_check"], d["ig_f1_check"]], 1) if "ig_check" in d
            else np.full((n, 2, 2), np.nan))
        cur = np.take_along_axis(d["del_curves"], a[None, :, None, None], 3)[..., 0]  # (R, n, k): A's score
        add("del", np.moveaxis(base_a[None, :, None] - cur, 0, 1))                    # (n, R, k): its fall
        arr.setdefault("del_labels", d["del_labels"])
        arr.setdefault("occl_drop_units", d["occl_drop_units"])
        arr.setdefault("occl_mean_units", d["occl_mean_units"])
        arr.setdefault("occl_chip_units", d["occl_chip_units"])
        add("seconds", dict(zip(d["seconds_names"], d["seconds"])))
    df = pd.DataFrame(rows)
    out = {}
    for k, v in arr.items():
        if k in ("del_labels", "occl_drop_units", "occl_mean_units", "occl_chip_units"):
            out[k] = list(v)
        elif k == "seconds":
            out[k] = pd.DataFrame(v)
        elif k.startswith("occl_"):
            out[k] = np.concatenate(v, 1).T                                           # (n, units)
        else:
            out[k] = np.concatenate(v, 0)
    df = df.merge(sample[["chip", "field", "class_name", "species", "sowing", "pred_species", "correct_species"]],
                  on=["chip", "field"], how="left")
    df["rival"] = [species[b] for b in df.b]
    assert (df.species == [species[a] for a in df.a]).all()
    return df, out


def unit_cols(units: list[str], kind: str) -> list[int]:
    return [i for i, u in enumerate(units) if u.split("|")[0] == kind]


def reference() -> pd.DataFrame:
    """The Estonian reference per species, the zone-21 one where Estonia has too few fields."""
    ee = pd.read_csv(HERE / "reference" / "ee_species_reference.csv").assign(source="Estonia")
    z = pd.read_csv(chain.REPO / "docs/research/xai/2026-11-16_crop_signatures/tables/species_reference.csv")
    z = z[z.zone == 21].assign(source="zone 21")
    cols = ["species", "peak", "cue_month", "cue_group", "cue_band", "cue_auc", "cue_direction", "verdict",
            "hide_window", "hide_window_drop", "hide_band_group", "hide_band_group_drop", "hide_sentinel2_drop",
            "hide_sentinel1_drop", "recall", "confused_with", "source"]
    missing = z[~z.species.isin(ee.species)]
    return pd.concat([ee[cols], missing[cols]], ignore_index=True).set_index("species")


# ------------------------------------------------------------------------------------------------------ helpers

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def months_of_window(w: str) -> list[int]:
    a, b = w.split("-")
    return list(range(MONTHS.index(a), MONTHS.index(b) + 1))


def window_relation(found: str, expected: str) -> str:
    if not isinstance(expected, str):
        return "no reference"
    fa, ea = MONTHS.index(found.split("-")[0]), MONTHS.index(expected.split("-")[0])
    return "same" if fa == ea else "neighbour" if abs(fa - ea) == 1 else f"{fa - ea:+d} months"


def group_sum(cells: np.ndarray) -> np.ndarray:
    """``(..., 12, 12)`` month x feature -> ``(..., 12, 6)`` month x band group."""
    return np.stack([cells[..., f].sum(-1) for f in BAND_GROUPS.values()], -1)


def style():
    import matplotlib as mpl

    mpl.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                         "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("dv", ["#8f2424", "#e34948", "#f0efec", "#3987e5", "#0d366b"])


def sequential():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("sq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def species_in(df: pd.DataFrame) -> list[str]:
    have = set(df.species)
    return [s for s in SPECIES_ORDER if s in have]


# ------------------------------------------------------------------------------------------------------ figures

def fig_when(df, A, ref, ok):
    """Species x month: SHAP, occlusion under the mean, occlusion under drop, IG; the expected window outlined."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    sp = species_in(df[ok])
    mu, du = A["occl_mean_units"], A["occl_drop_units"]
    panels = [("SHAP", A["shap_m0_a"]),
              ("Occlusion, mean", A["occl_mean_a"][:, unit_cols(mu, "month")]),
              ("Occlusion, drop", A["occl_drop_a"][:, unit_cols(du, "month")]),
              ("Integrated Gradients", A["ig_a"].sum(-1))]
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(sp) + 1.8), sharey=True)
    for ax, (title, v) in zip(axes, panels):
        mat = np.stack([v[ok & (df.species == s).to_numpy()].mean(0) for s in sp])
        lim = np.nanpercentile(np.abs(mat), 98) or 1
        im = ax.imshow(mat, cmap=diverging(), vmin=-lim, vmax=lim, aspect="auto")
        for i, s in enumerate(sp):
            if s in ref.index and isinstance(ref.hide_window[s], str):
                m = months_of_window(ref.hide_window[s])
                ax.add_patch(Rectangle((m[0] - 0.5, i - 0.5), len(m), 1, fill=False, ec=INK, lw=1.4))
            if s in ref.index:
                ax.plot(MONTHS.index(ref.peak[s]), i, marker="o", ms=3, color=INK)
        ax.set_xticks(range(12), [m[0] for m in MONTHS])
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(sp)), [f"{s} ({int((ok & (df.species == s)).sum())})" for s in sp])
    fig.suptitle("When: each month's contribution to the field's own species (log-probability), mean over the "
                 "fields the model got right", x=0.01, y=1.06, ha="left", fontsize=10)
    fig.text(0.01, -0.02, "Box: the three months the Estonian reference says matter most (hide_window). Dot: the "
             "species' NDVI peak. Blue = the month supports the species, red = it works against it.", fontsize=8,
             color=INK2)
    fig.savefig(FIG / "q1_when.png")
    plt.close(fig)


def fig_property(df, A, ref, ok):
    import matplotlib.pyplot as plt

    sp = species_in(df[ok])
    mu = A["occl_mean_units"]
    panels = [("SHAP", A["shap_g_a"]), ("Occlusion, mean", A["occl_mean_a"][:, unit_cols(mu, "band group")]),
              ("Integrated Gradients", group_sum(A["ig_a"]).sum(1))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.42 * len(sp) + 1.8), sharey=True)
    for ax, (title, v) in zip(axes, panels):
        mat = np.stack([v[ok & (df.species == s).to_numpy()].mean(0) for s in sp])
        lim = np.nanpercentile(np.abs(mat), 98) or 1
        im = ax.imshow(mat, cmap=diverging(), vmin=-lim, vmax=lim, aspect="auto")
        for i, s in enumerate(sp):
            if s in ref.index:
                if ref.hide_band_group[s] in GROUPS:
                    ax.plot(GROUPS.index(ref.hide_band_group[s]), i, marker="s", ms=9, mfc="none", mec=INK, mew=1.3)
                if ref.cue_group[s] in GROUPS:
                    ax.plot(GROUPS.index(ref.cue_group[s]), i, marker="o", ms=3, color=INK)
        ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=30, ha="right")
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    axes[0].set_yticks(range(len(sp)), sp)
    fig.suptitle("Which property: each band group's contribution (all twelve months)", x=0.01, ha="left",
                 fontsize=10)
    fig.text(0.01, -0.03, "Square: the band group whose hiding costs the reference's random forest most. Dot: the "
             "band group of the single best cue (AUC).", fontsize=8, color=INK2)
    fig.savefig(FIG / "q2_property.png")
    plt.close(fig)


def fig_cells(df, A, auc, ok, species: list[str]):
    """IG cell maps (month x band group) beside the reference AUC maps, for a few species."""
    import matplotlib.pyplot as plt

    species = [s for s in species if (ok & (df.species == s)).sum() >= 5 and s in set(auc.species)]
    if not species:
        return
    fig, axes = plt.subplots(2, len(species), figsize=(2.3 * len(species), 4.8), squeeze=False)
    ig = group_sum(A["ig_a"])
    for j, s in enumerate(species):
        m = ig[ok & (df.species == s).to_numpy()].mean(0).T                              # (6, 12)
        lim = np.abs(m).max() or 1
        axes[0, j].imshow(m, cmap=diverging(), vmin=-lim, vmax=lim, aspect="auto")
        a = auc[auc.species == s].pivot(index="group", columns="month", values="auc").reindex(index=GROUPS,
                                                                                                 columns=MONTHS)
        axes[1, j].imshow(a.to_numpy(), cmap=sequential(), vmin=0.5, vmax=1.0, aspect="auto")
        axes[0, j].set_title(s, loc="left")
        for ax in axes[:, j]:
            ax.set_xticks(range(12), [mm[0] for mm in MONTHS], fontsize=7)
            ax.set_yticks(range(6), GROUPS if j == 0 else [""] * 6, fontsize=7)
    axes[0, 0].set_ylabel("model (IG)")
    axes[1, 0].set_ylabel("reference (AUC)")
    fig.suptitle("Month x band group: where the model's evidence sits (top, IG, blue = supports) and what tells the "
                 "species apart in the data (bottom, AUC 0.5 to 1)", x=0.01, y=1.06, ha="left", fontsize=10)
    fig.savefig(FIG / "cells_vs_reference.png")
    plt.close(fig)


def fig_deletion(A, df):
    import matplotlib.pyplot as plt

    labels = A["del_labels"]
    d = A["del"]                                                                       # (n, R, k)
    k = d.shape[2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, op in zip(axes, ("drop", "mean")):
        for j, lab in enumerate(labels):
            name, o = lab.split("|")
            if o != op:
                continue
            y = np.r_[0, np.nanmean(d[:, j], 0)]
            if name.startswith("random"):
                ax.plot(range(k + 1), y, color=GRAY, lw=1.5, ls="--", label="random order" if name == "random0" else None)
            else:
                other = (name == "occlusion_drop") == (op == "mean")
                ax.plot(range(k + 1), y, color=METHOD_COLOR[name], lw=2.2 if other else 1.2,
                        alpha=1 if other else 0.55, marker="o" if other else None, ms=4,
                        label=METHOD_LABEL[name] + (" (other replacement)" if other else ""))
        ax.set_title(f"months removed with {'drop' if op == 'drop' else 'the mean'}", loc="left")
        ax.set_xlabel("months removed, most important first")
        ax.set_xticks(range(k + 1))
    axes[0].set_ylabel("fall of the species' log-probability")
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Deletion: removing months in each method's order (dashed grey: random orders). Thick lines use "
                 "the replacement the ranking did not use", x=0.01, y=1.06, ha="left", fontsize=10)
    fig.savefig(FIG / "check_deletion.png")
    plt.close(fig)


def fig_agreement(mat: pd.DataFrame, name: str, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.0 * len(mat) + 2, 0.8 * len(mat) + 1))
    im = ax.imshow(mat.to_numpy(), cmap=sequential(), vmin=0, vmax=1)
    for i in range(len(mat)):
        for j in range(len(mat)):
            v = mat.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(mat)), mat.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(mat)), mat.index)
    ax.set_title(title, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.05)
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_sowing(df, A, ok):
    import matplotlib.pyplot as plt

    sp = [s for s in ("wheat", "barley", "rapeseed")
          if all(((df.species == s) & (df.sowing == f) & ok).sum() >= 5 for f in ("winter", "spring"))]
    if not sp:
        return
    fig, axes = plt.subplots(1, len(sp), figsize=(3.6 * len(sp), 2.9), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], sp):
        for f, c in (("winter", "#2a78d6"), ("spring", "#eb6834")):
            sel = ok & (df.species == s).to_numpy() & (df.sowing == f).to_numpy()
            y = A["shap_m0_a"][sel].mean(0)
            ax.plot(range(12), y, color=c, lw=2, marker="o", ms=3, label=f"{f} ({sel.sum()} fields)")
        ax.legend(frameon=False, fontsize=7, loc="upper left")       # end labels collide where the lines meet
        ax.axhline(0, color="#c3c2b7", lw=0.8)
        ax.set_xticks(range(12), [m[0] for m in MONTHS])
        ax.set_title(s, loc="left")
    axes[0, 0].set_ylabel("SHAP of the month")
    fig.suptitle("Sowing form: the months that make a field 'wheat' (or barley, rapeseed), winter against spring "
                 "fields", x=0.01, y=1.06, ha="left", fontsize=10)
    fig.savefig(FIG / "q6_sowing.png")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ tables

def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """A markdown table, without pandas' optional tabulate."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        return "" if v is None else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    root = chain.results_dir()
    ap.add_argument("--explain", default=str(root / "explain"))
    ap.add_argument("--sanity", default=str(root / "sanity"))
    ap.add_argument("--step0", default=str(root / "step0"))
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    style()
    df, A = load(Path(args.explain))
    ref = reference()
    auc = pd.read_csv(HERE / "reference" / "ee_auc_map.csv")
    ok = df.correct_species.to_numpy(bool)
    n = len(df)
    T = []                                                                            # (title, markdown)

    # ---- checks
    v = Path(args.step0) / "verify.json"
    if v.exists():
        vj = json.loads(v.read_text())
        T.append(("Chain against the stored predictions", md(pd.DataFrame(
            [{"precision": p, "pixel agreement": vj[p]["pixel_agreement"], "parcel agreement": vj[p]["parcel_agreement"],
              "s per chip": vj[p]["seconds_per_chip"]} for p in ("bf16-mixed", "32")]), ".5f")))
    span = A["ig_f1"] - A["ig_f0"]
    err = np.abs(A["ig_a"].sum((1, 2)) - span[:, 0]) / np.maximum(np.abs(span[:, 0]), 1e-6)
    shap_eff = np.abs(A["shap_m0_a"].sum(1) - (A["base_a"] - A["empty_a"]))
    conv = [spearman(A["shap_m1_a"][i], A["shap_m2_a"][i]) for i in range(n)]
    chk = ~np.isnan(A["ig_check"].reshape(n, -1)).any(1)
    conv_ig = [spearman(A["ig_a"][i].sum(-1), A["ig_check"][i].sum(-1)) for i in np.flatnonzero(chk)]
    err64 = (np.abs(A["ig_check"][chk].sum((1, 2)) - (A["ig_check_f"][chk, 1, 0] - A["ig_check_f"][chk, 0, 0]))
             / np.maximum(np.abs(A["ig_check_f"][chk, 1, 0] - A["ig_check_f"][chk, 0, 0]), 1e-6))
    T.append(("Completeness and convergence", md(pd.DataFrame([
        {"check": "SHAP months: largest gap between the sum and f(x) - f(mean)", "value": float(shap_eff.max())},
        {"check": "IG, 32 steps: relative completeness error, median", "value": float(np.median(err))},
        {"check": "IG, 32 steps: relative completeness error, 90th percentile", "value": float(np.percentile(err, 90))},
        {"check": f"IG, 64 steps ({chk.sum()} fields): relative error, median",
         "value": float(np.median(err64)) if chk.any() else np.nan},
        {"check": "SHAP months, two halves of the permutations: Spearman, median", "value": float(np.nanmedian(conv))},
        {"check": "IG months, 32 against 64 steps: Spearman, median",
         "value": float(np.nanmedian(conv_ig)) if conv_ig else np.nan}]), ".4f")))

    labels = A["del_labels"]
    rows = []
    for j, lab in enumerate(labels):
        name, op = lab.split("|")
        rows.append({"order": name, "removed with": op, "mean fall over the steps (AOPC)": float(np.nanmean(A["del"][:, j])),
                     "after 1 month": float(np.nanmean(A["del"][:, j, 0])),
                     "after all steps": float(np.nanmean(A["del"][:, j, -1]))})
    dl = pd.DataFrame(rows)
    rnd = dl[dl.order.str.startswith("random")].groupby("removed with")["mean fall over the steps (AOPC)"].mean()
    dl["against random"] = dl["mean fall over the steps (AOPC)"] - dl["removed with"].map(rnd)
    T.append(("Deletion curves (A's log-probability)", md(dl, ".3f")))
    fig_deletion(A, df)

    # agreement between methods, per field
    mu, du = A["occl_mean_units"], A["occl_drop_units"]
    month_imp = {"occlusion_mean": A["occl_mean_a"][:, unit_cols(mu, "month")],
                 "occlusion_drop": A["occl_drop_a"][:, unit_cols(du, "month")],
                 "shap": A["shap_m0_a"], "ig": A["ig_a"].sum(-1)}
    group_imp = {"occlusion_mean": A["occl_mean_a"][:, unit_cols(mu, "band group")], "shap": A["shap_g_a"],
                 "ig": group_sum(A["ig_a"]).sum(1)}
    cell_imp = {"occlusion_mean": A["occl_mean_a"][:, unit_cols(mu, "cell")],
                "ig": group_sum(A["ig_a"]).reshape(n, -1)}
    for name, imp, fname in (("months", month_imp, "agreement_months.png"),
                             ("band groups", group_imp, "agreement_groups.png")):
        keys = list(imp)
        mat = pd.DataFrame([[np.nanmedian([spearman(imp[a][i], imp[b][i]) for i in np.flatnonzero(ok)])
                             for b in keys] for a in keys],
                           index=[METHOD_LABEL[k] for k in keys], columns=[METHOD_LABEL[k] for k in keys])
        fig_agreement(mat, fname, f"Agreement on the {name}: Spearman per field, median")
        T.append((f"Agreement on the {name} (Spearman per field, median over the fields the model got right)",
                  md(mat.reset_index(names=""))))
    T.append(("Agreement on the 72 cells, occlusion (mean) against IG",
              f"median Spearman {np.nanmedian([spearman(cell_imp['occlusion_mean'][i], cell_imp['ig'][i]) for i in np.flatnonzero(ok)]):.2f}"))

    # sanity
    sd = Path(args.sanity)
    if sd.is_dir() and any(sd.glob("*.npz")):
        rows = []
        idx = df.reset_index().set_index(["chip", "field"])["index"]
        wcols = unit_cols(mu, "window")
        wcols_d = unit_cols(du, "window")
        for f in sorted(sd.glob("*.npz")):
            s = dict(np.load(f))
            for part in ("head", "head+decoder"):
                for i, fid in enumerate(s["field"]):
                    j = idx.get((f.stem, int(fid)))
                    if j is None:
                        continue
                    a = df.a[j] + K
                    base = s[f"{part}|base"][i, a]
                    rows.append({"randomised": part,
                                 "windows, mean": spearman(A["occl_mean_a"][j, wcols], base - s[f"{part}|windows_mean"][:, i, a]),
                                 "windows, drop": spearman(A["occl_drop_a"][j, wcols_d], base - s[f"{part}|windows_drop"][:, i, a]),
                                 "IG months": spearman(A["ig_a"][j].sum(-1), s[f"{part}|ig"][i].sum(-1))})
        if rows:
            T.append(("Randomised network: Spearman with the trained network's explanation, median (near 0 = the "
                      "method depends on the weights)", md(pd.DataFrame(rows).groupby("randomised").median().reset_index())))

    # ---- Q1 when
    wcols = unit_cols(mu, "window")
    wnames = [u.split("|")[1] for u in np.array(mu)[wcols]]
    wcols_d = unit_cols(du, "window")
    rows = []
    for s in species_in(df[ok]):
        sel = ok & (df.species == s).to_numpy()
        r = ref.loc[s] if s in ref.index else None
        exp = r.hide_window if r is not None else None
        top_m = wnames[int(np.argmax(A["occl_mean_a"][sel][:, wcols].mean(0)))]
        top_d = wnames[int(np.argmax(A["occl_drop_a"][sel][:, wcols_d].mean(0)))]
        row = {"species": s, "fields": int(sel.sum()), "reference window": exp,
               "occlusion (mean)": f"{top_m} ({window_relation(top_m, exp)})",
               "occlusion (drop)": f"{top_d} ({window_relation(top_d, exp)})"}
        if isinstance(exp, str):
            w = months_of_window(exp)
            for key, v in (("SHAP share in window", A["shap_m0_a"][sel]), ("IG share in window", A["ig_a"][sel].sum(-1))):
                share = np.abs(v[:, w]).sum(1) / np.maximum(np.abs(v).sum(1), 1e-9)
                row[key] = float(np.median(share))
        row["NDVI peak"] = r.peak if r is not None else None
        row["reference"] = r.source if r is not None else None
        rows.append(row)
    q1 = pd.DataFrame(rows)
    T.append(("Q1 When: the costliest window and the share of attribution in the expected window (uniform = 0.25)",
              md(q1)))
    fig_when(df, A, ref, ok)

    # ---- Q2 which property
    gcols = unit_cols(mu, "band group")
    scols = unit_cols(du, "sensor")
    rows = []
    for s in species_in(df[ok]):
        sel = ok & (df.species == s).to_numpy()
        r = ref.loc[s] if s in ref.index else None
        shap_g = A["shap_g_a"][sel].mean(0)
        occ_g = A["occl_mean_a"][sel][:, gcols].mean(0)
        ig_g = group_sum(A["ig_a"][sel]).sum(1).mean(0)
        sens = A["occl_drop_a"][sel][:, scols].mean(0)
        s1_share = np.abs(A["shap_g_a"][sel][:, 4:]).sum(1) / np.maximum(np.abs(A["shap_g_a"][sel]).sum(1), 1e-9)
        rows.append({"species": s, "SHAP top": GROUPS[int(np.argmax(shap_g))],
                     "occlusion (mean) top": GROUPS[int(np.argmax(occ_g))], "IG top": GROUPS[int(np.argmax(ig_g))],
                     "reference: costliest group": r.hide_band_group if r is not None else None,
                     "reference: best cue": f"{r.cue_month} {r.cue_group} ({r.cue_auc:.2f})" if r is not None else None,
                     "S2 removed (drop)": float(sens[0]), "S1 removed (drop)": float(sens[1]),
                     "reference S2 / S1 recall lost": f"{r.hide_sentinel2_drop:.2f} / {r.hide_sentinel1_drop:.2f}"
                     if r is not None else None,
                     "SHAP share of radar": float(np.median(s1_share))})
    T.append(("Q2 Which property: the leading band group by each method, and the sensors", md(pd.DataFrame(rows))))
    fig_property(df, A, ref, ok)

    # ---- the cell maps against the reference AUC map
    rows = []
    igc = group_sum(A["ig_a"]).reshape(n, -1)
    occ_c = A["occl_mean_a"][:, unit_cols(mu, "cell")]
    for s in species_in(df[ok]):
        sel = ok & (df.species == s).to_numpy()
        a = auc[auc.species == s]
        if a.empty:
            continue
        amap = a.pivot(index="month", columns="group", values="auc").reindex(index=MONTHS, columns=GROUPS).to_numpy()
        nulls = (amap < 0.55).ravel()
        mig = np.abs(igc[sel]).mean(0)
        rows.append({"species": s, "IG cells (absolute) vs AUC": spearman(mig, amap.ravel()),
                     "occlusion cells vs AUC": spearman(occ_c[sel].mean(0), amap.ravel()),
                     "null cells (AUC < 0.55)": int(nulls.sum()),
                     "IG share on null cells": float(mig[nulls].sum() / max(mig.sum(), 1e-9)),
                     "null cells' share of the map": float(nulls.mean())})
    T.append(("The 72 cells against the reference AUC map (Spearman), and the null cells (Q7)", md(pd.DataFrame(rows))))
    fig_cells(df, A, auc, ok, ["rapeseed", "maize", "wheat", "barley", "grassland", "buckwheat"])

    # ---- Q3 against which crop
    rows = []
    pairs = pd.read_csv(HERE / "reference" / "ee_pairs.csv")
    ab = group_sum(A["ig_ab"]).reshape(n, -1)
    cell_names = [f"{m} {g}" for m in MONTHS for g in GROUPS]
    for (s, b), g in df.groupby(["species", "rival"]):
        if len(g) < 8:
            continue
        i = g.index.to_numpy()
        m = ab[i].mean(0)
        top = cell_names[int(np.argmax(m))]
        conc = float(np.median(np.abs(ab[i]).max(1) / np.maximum(np.abs(ab[i]).sum(1), 1e-9)))
        p = pairs[((pairs.a == s) & (pairs.b == b)) | ((pairs.a == b) & (pairs.b == s))]
        cue = f"{p.month.iat[0]} {p.group.iat[0]} ({p.auc.iat[0]:.2f}, {p.verdict.iat[0]})" if len(p) else None
        # where the pair's reference cue cell ranks in the model's mean A - B map (1 = the model's top cell, of 72)
        rank = (int(1 + (m > m[cell_names.index(f"{p.month.iat[0]} {p.group.iat[0]}")]).sum())
                if len(p) and f"{p.month.iat[0]} {p.group.iat[0]}" in cell_names else None)
        rows.append({"A": s, "rival B": b, "fields": len(g), "model wrong": int((~g.correct_species).sum()),
                     "IG (A - B) top cell": top, "share of the top cell": conc, "reference pair cue": cue,
                     "rank of the reference cue cell (of 72)": "" if rank is None else str(rank)})
    T.append(("Q3 Against which crop: what separates a field's species from its rival (IG of A - B)",
              md(pd.DataFrame(rows).sort_values("fields", ascending=False)) if rows
              else "no species and rival with 8 fields or more"))

    # ---- Q4 direction
    rows = []
    feat = {name: i for i, name in enumerate(chain.FEATURES)}
    for s in species_in(df[ok]):
        if s not in ref.index or not isinstance(ref.cue_band[s], str):
            continue
        r = ref.loc[s]
        band = S2_NAME.get(r.cue_band, r.cue_band)
        if band not in feat:
            continue
        sel = ok & (df.species == s).to_numpy()
        g = A["ig_grad_a"][sel][:, MONTHS.index(r.cue_month), feat[band]]
        agree = np.mean(np.sign(g) == (1 if r.cue_direction == "higher" else -1))
        rows.append({"species": s, "cue": f"{r.cue_month} {band}", "the species is": r.cue_direction,
                     "fields": int(sel.sum()), "share whose gradient agrees": float(agree)})
    T.append(("Q4 Direction: does raising the cue band move the species' score the way the data says?",
              md(pd.DataFrame(rows))))

    # ---- Q6 sowing form
    rows = []
    for s in ("wheat", "barley", "rapeseed"):
        for f in ("winter", "spring"):
            sel = ok & (df.species == s).to_numpy() & (df.sowing == f).to_numpy()
            if sel.sum() < 5:
                continue
            v = A["shap_m0_a"][sel].mean(0)
            top3 = [MONTHS[k] for k in np.argsort(-v)[:3]]
            rows.append({"species": s, "form": f, "fields": int(sel.sum()), "top months (SHAP)": ", ".join(top3),
                         "share Jan to May": float(v[:5].clip(0).sum() / max(v.clip(0).sum(), 1e-9))})
    T.append(("Q6 Sowing form: winter against spring fields of one species",
              md(pd.DataFrame(rows)) if rows else "fewer than 5 fields of a form"))
    fig_sowing(df, A, ok)

    # ---- Q7 cloud-filled months
    filled = [1, 2, 10]
    sh = np.abs(A["shap_m0_a"][ok])
    T.append(("Q7 Cloud-filled months (Feb, Mar, Nov)",
              f"share of |SHAP| on them, median over fields: {np.median(sh[:, filled].sum(1) / np.maximum(sh.sum(1), 1e-9)):.3f} "
              f"(uniform 0.25)"))

    # ---- context: chip against field scope
    cu = A["occl_chip_units"]
    rows = []
    for op in ("mean", "drop"):
        cc = [i for i, u in enumerate(cu) if u.startswith("window|") and u.endswith(f"|{op}")]
        fc = unit_cols(mu if op == "mean" else du, "window")
        a_f, a_c = A[f"occl_{op}_a"][ok][:, fc], A["occl_chip_a"][ok][:, cc]
        if a_c.size == 0:
            continue
        rows.append({"replacement": op, "largest window fall, own field only": float(np.median(a_f.max(1))),
                     "same, whole chip": float(np.median(a_c.max(1))),
                     "Spearman field vs chip, median": float(np.nanmedian([spearman(a_f[i], a_c[i]) for i in range(len(a_f))]))})
    T.append(("Context: the same windows removed from the field only or from the whole chip", md(pd.DataFrame(rows))))

    # ---- the sample and the cost
    T.insert(0, ("The fields explained", md(df.groupby("species").agg(fields=("field", "size"),
                                                                     right=("correct_species", "mean"),
                                                                     median_pixels=("n_pixels", "median")).reset_index())))
    T.append(("Minutes per chip, median", md(A["seconds"].div(60).median().rename("minutes").rename_axis("part").reset_index(), ".1f")))
    (FIG / "tables.md").write_text("\n\n".join(f"### {t}\n\n{m}" for t, m in T) + "\n")
    print(f"{n} fields from {df.chip.nunique()} chips -> {FIG}")


if __name__ == "__main__":
    main()
