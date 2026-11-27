"""Field cases: how good a cue, in the data, is what each method says drives a confident prediction?

    poetry run python experiments/2026-11-26_xai_tessera_blocks/field_cases.py

One field per species: among the fields predicted correctly with p >= 0.95 (p = the field's species probability,
geometric mean over its interior), the one with the most interior pixels. For each, every method's main window,
property and cell, and the AUC of that block in the species' map (``reference/ee_block_auc.csv``; a window's AUC is
its best property's, a property's its best window's), beside the species' main AUC cues. Above them, the same for
all correctly predicted fields. Writes ``FIELD_CASES.md`` and ``figures/fields/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import chain  # noqa: E402
import compare as C  # noqa: E402

OUT = C.FIG / "fields"
FEAT = chain.FEATURES
PANELS = [("Visible (B04, red)", "B04"), ("Red-edge (B05)", "B05"), ("NIR (B08)", "B08"), ("SWIR (B11)", "B11"),
          ("VV (dB)", "VV"), ("VH (dB)", "VH"), ("NDVI", "NDVI"), ("NDMI", "NDMI")]
PROP_OF_PANEL = ["Visible", "Red-edge", "NIR", "SWIR", "VV", "VH", "NDVI", "NDMI"]
LEVELS = {"window": ["Occlusion", "SHAP", "IG", "Drop"], "property": ["Occlusion", "SHAP (groups)", "IG"],
          "cell": ["Occlusion", "IG"]}


def panel_values(X: np.ndarray) -> np.ndarray:
    """``(n, 12, 12)`` inputs in TESSERA's order -> ``(n, 12, 8)``: one band per property, NDVI, NDMI."""
    b = lambda k: X[..., FEAT.index(k)].astype(np.float64)  # noqa: E731
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (b("B08") - b("B04")) / (b("B08") + b("B04"))
        ndmi = (b("B8A") - b("B11")) / (b("B8A") + b("B11"))
    return np.stack([b("B04"), b("B05"), b("B08"), b("B11"), b("VV"), b("VH"), ndvi, ndmi], -1)


def fig_input(path, title, field_v, sp_mean, ref, win, prop):
    import matplotlib.pyplot as plt

    med, q25, q75 = (np.nanpercentile(field_v, q, 0) for q in (50, 25, 75))
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 4.4), sharex=True)
    x = np.arange(12)
    a, b = chain.WINDOWS[C.WN.index(win)]
    for k, (ax, (name, _)) in enumerate(zip(axes.ravel(), PANELS)):
        ax.axvspan(a - 0.5, b - 0.5, color="#e7e6e1", lw=0, zorder=0)
        ax.fill_between(x, q25[:, k], q75[:, k], color="#2a78d6", alpha=0.18, lw=0)
        ax.plot(x, med[:, k], color="#2a78d6", lw=2, label="this field (median; middle half shaded)")
        ax.plot(x, sp_mean[:, k], color="#eb6834", lw=1.4, ls="--", label="its species, training fields")
        ax.plot(x, ref[:, k], color="#8a8a83", lw=1.4, ls=":", label="all species (the methods' reference)")
        star = PROP_OF_PANEL[k] == prop
        ax.set_title(name + ("  ← main property" if star else ""), loc="left", fontsize=9,
                     color="#1f1f1d" if star else "#55554f", fontweight="bold" if star else "normal")
        ax.set_xticks(x, [m[0] for m in chain.MONTHS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"{title}: the monthly input (grey band: the main window, {win})", x=0.01, ha="left", y=1.01, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    C.style()
    OUT.mkdir(parents=True, exist_ok=True)
    df, X, total, _ = C.load()
    maps = C.auc_maps()
    for lv in C.METHODS:
        X[lv]["Consensus"] = C.consensus(X, lv)
    keep = df.explained_right.to_numpy() & df.species.isin(list(maps)).to_numpy()
    units = C.LEVEL_UNITS

    def top(lv, m, i):
        x = X[lv][m][i]
        return int(np.nanargmax(np.where(np.isfinite(x), x, -np.inf)))

    # ---- all correct fields: the AUC of each method's main block, against the best and a typical block
    rows = []
    for lv, ms in LEVELS.items():
        for m in (["Consensus"] if lv == "window" else []) + ms:
            got, best, typ = [], [], []
            for i in np.flatnonzero(keep):
                r = maps[df.species.iat[i]][lv]
                ok = np.isfinite(X[lv][m][i])
                got.append(r[top(lv, m, i)])
                best.append(np.nanmax(r[ok]))
                typ.append(np.nanmedian(r[ok]))
            rows.append({"level": lv, "method": m, "AUC of the main block": np.median(got),
                         "best block": np.median(best), "typical block (median)": np.median(typ),
                         "main block above the typical": np.mean(np.array(got) > np.array(typ))})
    S = pd.DataFrame(rows)

    # ---- one field per species
    mf = np.load(chain.results_dir() / "step0" / "mean_field.npz")
    ref = panel_values(mf["baseline"][None])[0]
    tr = pd.read_parquet(chain.results_dir() / "auc_map" / "fields.parquet")
    cols = [f"{m}_{b}" for m in chain.MONTHS for b in FEAT]
    root = chain.REPO / "data/eurocrops_chips/EE_2021_rev/chips"
    sp_order = [s for s in df.species.value_counts().index if s in maps]
    cases, glance = [], []
    for s in sp_order:
        sel = keep & (df.species == s).to_numpy()
        hi = sel & (df.p.to_numpy() >= 0.95)
        pool = hi if hi.any() else sel
        i = int(np.flatnonzero(pool)[np.argmax(df.n_interior.to_numpy()[pool])])
        r = df.iloc[i]
        m_ = maps[s]
        win, prop = units["window"][top("window", "Consensus", i)], units["property"][top("property", "Occlusion", i)]
        d = chain.read_chip(root, r.chip)
        f = next(x for x in chain.chip_fields(d["mask"], d["parcels"]) if x["field"] == r.field)
        trs = tr[tr.species == s][cols].to_numpy(np.float64).reshape(-1, 12, 12)
        title = f"{s}, field {r.field} (p = {r.p:.3f})"
        stem = f"{s.replace(' ', '_')}_{r.field}"
        fig_input(OUT / f"{stem}.png", title, panel_values(d["X"][f["interior"]]), np.nanmean(panel_values(trs), 0),
                  ref, win, prop)
        trows = []
        for m in ["Occlusion", "SHAP", "IG", "Drop"]:
            row = {"method": m}
            for lv, ms in LEVELS.items():
                mm = "SHAP (groups)" if (m == "SHAP" and lv == "property") else m
                if mm not in ms:
                    row[lv] = "–"
                    continue
                j = top(lv, mm, i)
                row[lv] = f"{units[lv][j]} ({m_[lv][j]:.2f})"
            trows.append(row)
        T = pd.DataFrame(trows).rename(columns={"window": "main window (AUC)", "property": "main property (AUC)",
                                                "cell": "main cell (AUC)"})
        mw = [f"{units['window'][k]} {m_['window'][k]:.2f}" for k in np.argsort(-m_["window"])[:2]]
        mp = [f"{units['property'][k]} {m_['property'][k]:.2f}" for k in np.argsort(-m_["property"])[:2]]
        bc = int(np.argmax(m_["cell"]))
        cue = (f"windows {', '.join(mw)}; properties {', '.join(mp)}; best cell {units['cell'][bc]} {m_['cell'][bc]:.2f}")
        cw, cp = top("window", "Consensus", i), top("property", "Occlusion", i)
        glance.append({"species": s, "p": r.p, "main window (AUC; best)": f"{units['window'][cw]} ({m_['window'][cw]:.2f}; {m_['window'].max():.2f})",
                       "main property (AUC; best)": f"{units['property'][cp]} ({m_['property'][cp]:.2f}; {m_['property'].max():.2f})",
                       "main cell, occlusion (AUC; best)": f"{units['cell'][top('cell', 'Occlusion', i)]} "
                                                           f"({m_['cell'][top('cell', 'Occlusion', i)]:.2f}; {m_['cell'].max():.2f})"})
        cases.append((s, r, stem, T, cue))

    lines = [HEAD, "## All correctly predicted fields\n",
             "Median over the 438 fields of the AUC of each method's main block, beside the AUC of the species' best "
             "block and of a typical one (the median block); and the share of fields whose main block is above the "
             "typical one.\n", C.md(S), "\n", SUMMARY, "\n## The fields at a glance\n",
             "Main window: the consensus of occlusion, SHAP and IG. Main property and cell: occlusion.\n",
             C.md(pd.DataFrame(glance), ".3f"), "\n"]
    for s, r, stem, T, cue in cases:
        lines += [f"## {s.capitalize()}: field {r.field} (p = {r.p:.3f}, {r.n_interior} interior pixels)\n",
                  f"![input](figures/fields/{stem}.png)\n", C.md(T), "\n",
                  f"**Main AUC cues of {s}:** {cue}.\n", f"{READ.get(s, '')}\n"]
    (HERE / "FIELD_CASES.md").write_text("\n".join(lines))
    print(C.md(S))
    print(C.md(pd.DataFrame(glance), ".3f"))


HEAD = """# Field cases: how good a cue is what each method says drives the prediction?

From the focused run ([REPORT.md](REPORT.md)); generated by `field_cases.py`. One confidently and correctly predicted
field per species (p ≥ 0.95; oats, with none, its surest field).

**How to read.** For each method, the *main* block is the one it ranks first: the window, the property (band group
or index) and the window × property cell. Beside it, in brackets, that block's **AUC**: how well it alone separates
the species from the other crops in the training fields (0.5 = not at all, 1 = perfectly). A window's AUC is that of
its best property, a property's that of its best window. The species' **main AUC cues** are listed below each table.
IG's property and cell choices are the least reliable (its split between correlated bands is unstable).
"""

SUMMARY = """**What it says.** The model's main blocks are informative but rarely the best cue. Its main window has a
median AUC of 0.78 (the species' best window 0.85, a typical window 0.77), above a typical window for two thirds of
the fields; its main property 0.80 to 0.82 by occlusion and SHAP (best 0.85, typical 0.80); its main cell 0.74 by
occlusion (best 0.85, typical 0.66), above a typical cell for three quarters of the fields. The windows the model
cannot do without (drop) are better cues, 0.83. IG's main property is no better than a typical one.
"""

READ = {
    "rye": "Occlusion and SHAP pick Jul-Sep, a weak window (0.69); IG and drop pick May-Jul, the best (0.85). Red-edge "
           "(0.80) is close to the best property, and occlusion's main cell, May-Jul SWIR, is the best cell (0.85).",
    "buckwheat": "The model reads red-edge (0.86, near the best 0.90) from May to August (0.71 to 0.78); the best season "
                 "in the data, July to October (0.90), is picked only by drop.",
    "potatoes": "Every method's main window (Jun-Aug, 0.90) and, by occlusion and SHAP, its main property (SWIR, 0.91) "
                "are at the top of the data's cues.",
    "rapeseed": "The main window is the best (May-Jul, 0.95); red-edge (0.89) and May-Jul NDMI (0.88) are strong cues, "
                "though the data's best is radar (VH 0.95).",
    "oats": "Oats has only weak cues (best 0.75, VV). The model's window is close to the best (0.72 to 0.75), its main "
            "property, SWIR, weak (0.61).",
    "maize": "The clearest case: occlusion and SHAP name the data's best window, property and cell (Jul-Sep red-edge, "
             "0.94). IG differs (Jun-Aug, 0.71).",
    "barley": "Barley has no strong cue (all 0.78 or less). The model's window (0.74) and property (visible, 0.78) are "
              "near the top of that flat map; its cells are weak (0.52 to 0.55).",
    "forage legumes": "Moderate cues in the wrong season: the model reads May-Jul (0.75) red-edge and SWIR (0.72 to "
                      "0.76); the data's best are autumn NDVI and visible (0.80).",
    "grassland": "Good cues, not the best: May-Jul (0.83) NIR (0.86). The data's best, autumn VV (0.91), is not what "
                 "the model uses.",
    "dry pulses": "A flat map (0.74 to 0.76): the methods differ on the window, all on cues of that level; occlusion's "
                  "main property, NDMI (0.74), is one of the main AUC properties.",
    "wheat": "Red-edge (0.82) is the data's best property; the model reads it in Jun-Aug (0.79), one window after the "
             "best (Apr-Jun, 0.82).",
}


if __name__ == "__main__":
    main()
