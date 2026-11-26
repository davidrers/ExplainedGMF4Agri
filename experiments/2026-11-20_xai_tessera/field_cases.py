"""Case studies: one confidently and correctly predicted field per species, with the input as the model gets it, the
main drivers by each method, and the species' AUC map. Writes ``figures/fields/`` and ``FIELD_CASES.md``.

    poetry run python experiments/2026-11-20_xai_tessera/field_cases.py

**Choice of fields.** Per species with an Estonian AUC map: among the correctly predicted fields (``drivers.py``'s
rule) with p >= 0.95 (p = the field's species probability, geometric mean over its interior), the one with the most
interior pixels; a species without such a field gets its most confident correct field, flagged.

**Input curves.** The monthly composites exactly as they enter TESSERA (Sentinel-2 surface reflectance x 10,000,
Sentinel-1 in dB): the median over the field's interior pixels and the band between its 25th and 75th percentiles;
beside them the species' mean over the labelled pixels of the training chips, and the species-balanced training
mean, the field that occlusion (mean), SHAP and IG replace the input with.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import chain  # noqa: E402
import drivers as D  # noqa: E402
import report as R  # noqa: E402

MONTHS, GROUPS, FEAT = chain.MONTHS, list(chain.BAND_GROUPS), chain.FEATURES
OUT = R.FIG / "fields"
GROUP_OF = {f: g for g, idx in chain.BAND_GROUPS.items() for f in (FEAT[i] for i in idx)}
FIELD_C, SPECIES_C, BASE_C = "#2a78d6", "#eb6834", "#8a8a83"
METHODS = {"month": ["SHAP", "IG", "Occlusion (mean)", "Occlusion (drop)"], "band group": ["SHAP", "IG", "Occlusion (mean)"],
           "cell": ["IG", "Occlusion (mean)"]}


def pick(df, A, keep, species) -> list[tuple[str, int, bool]]:
    p = np.exp(np.asarray(A["base_a"]))
    out = []
    for s in species:
        sel = keep & (df.species == s).to_numpy()
        hi = sel & (p >= 0.95)
        if hi.any():
            i = int(np.flatnonzero(hi)[np.argmax(df.n_pixels.to_numpy()[hi])])
            out.append((s, i, True))
        elif sel.any():
            out.append((s, int(np.flatnonzero(sel)[np.argmax(p[sel])]), False))
    return out


def field_input(chip: str, field: int):
    d = chain.read_chip(chain.REPO / "data/eurocrops_chips/EE_2021_rev/chips", chip)
    f = next(x for x in chain.chip_fields(d["mask"], d["parcels"]) if x["field"] == field)
    X = d["X"][f["interior"]].astype(np.float64)
    return np.nanmedian(X, 0), np.nanpercentile(X, 25, 0), np.nanpercentile(X, 75, 0), len(f["interior"])


def fig_input(path, title, med, q25, q75, sp_mean, base):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(13, 4.6), sharex=True)
    x = np.arange(12)
    for k, (ax, feat) in enumerate(zip(axes.ravel(), FEAT)):
        ax.fill_between(x, q25[:, k], q75[:, k], color=FIELD_C, alpha=0.18, lw=0)
        ax.plot(x, med[:, k], color=FIELD_C, lw=2, label="this field (median, middle half shaded)")
        ax.plot(x, sp_mean[:, k], color=SPECIES_C, lw=1.4, ls="--", label="its species, training mean")
        ax.plot(x, base[:, k], color=BASE_C, lw=1.4, ls=":", label="all species, training mean (the methods' reference)")
        ax.set_title(f"{feat} ({GROUP_OF[feat] if k < 10 else 'Sentinel-1'})", loc="left", fontsize=9)
        ax.set_xticks(x, [m[0] for m in MONTHS], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if k in (0, 6):
            ax.set_ylabel("reflectance x 10,000", fontsize=8)
    axes[1, 4].set_ylabel("dB", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title + ": the input as the model gets it (monthly composites; Sentinel-1 in dB)", x=0.01, ha="left",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_drivers(path, title, xi, amap, top_auc):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(13, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.75, wspace=0.42)
    maps = [("IG per cell (log p; blue supports)", xi["cell"]["IG"].reshape(12, 6), "div"),
            ("Occlusion (mean) per cell: fall of log p", xi["cell"]["Occlusion (mean)"].reshape(12, 6), "div"),
            ("AUC map: the species against the rest", amap, "seq")]
    tops = set(top_auc["cell"])
    for j, (name, m, kind) in enumerate(maps):
        ax = fig.add_subplot(gs[0, j])
        if kind == "div":
            v = np.nanmax(np.abs(m)) or 1
            im = ax.imshow(m.T, cmap=R.diverging(), vmin=-v, vmax=v, aspect="auto")
            t = int(np.argmax(m))
            ax.plot(t // 6, t % 6, "o", color="#1f1f1d", ms=4)
        else:
            im = ax.imshow(m.T, cmap=R.sequential(), vmin=0.5, vmax=1, aspect="auto")
        for c in tops:
            ax.add_patch(Rectangle((c // 6 - 0.5, c % 6 - 0.5), 1, 1, fill=False, ec="#1f1f1d", lw=1.4))
        ax.set_xticks(range(12), [mm[0] for mm in MONTHS], fontsize=7)
        ax.set_yticks(range(6), GROUPS, fontsize=7)
        ax.set_title(name, loc="left", fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    # months: each method scaled by its largest absolute value, so their rankings can be compared
    ax = fig.add_subplot(gs[1, :2])
    for mm in top_auc["month"]:
        ax.axvspan(mm - 0.5, mm + 0.5, color="#e7e6e1", lw=0, zorder=0)
    for meth in METHODS["month"]:
        v = xi["month"][meth]
        ax.plot(range(12), v / (np.max(np.abs(v)) or 1), marker="o", ms=4, lw=1.6, color=D.COLOR[meth], label=meth)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(range(12), MONTHS, fontsize=8)
    ax.set_ylabel("relative to its largest", fontsize=8)
    ax.set_title("Months (grey: the species' 3 best AUC months)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.14))
    ax = fig.add_subplot(gs[1, 2])
    w = 0.26
    for g in top_auc["band group"]:
        ax.axvspan(g - 0.5, g + 0.5, color="#e7e6e1", lw=0, zorder=0)
    for k, meth in enumerate(METHODS["band group"]):
        v = xi["band group"][meth]
        ax.bar(np.arange(6) + (k - 1) * w, v / (np.max(np.abs(v)) or 1), width=w, color=D.COLOR[meth], label=meth)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.set_xticks(range(6), GROUPS, fontsize=7, rotation=30)
    ax.set_title("Band groups (grey: the 2 best AUC groups)", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(-0.05, -0.3))
    fig.text(0.01, -0.06, "Squares: the species' 5 best AUC cells. Dot: the method's top cell. Values of SHAP and IG "
             "are shares of log p (they add up to the field's score minus the reference's); occlusion is the fall "
             "of log p when the block alone is hidden.", fontsize=7.5, color="#55554f")
    fig.suptitle(title + ": main drivers by each method, against the AUC map", x=0.01, ha="left", y=0.98, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    R.style()
    OUT.mkdir(parents=True, exist_ok=True)
    df, A = R.load(chain.results_dir() / "explain")
    n = len(df)
    maps = D.auc_levels()
    auc_cells = {s: m["cell"].reshape(12, 6) for s, m in maps.items()}
    pred = np.asarray(A["base_all"])[:, R.K:].argmax(1)
    keep = df.correct_species.to_numpy(bool) & (pred == df.a.to_numpy())
    X = D.xai_levels(A, n)
    mf = np.load(chain.results_dir() / "step0" / "mean_field.npz")
    mf_species = list(mf["species"])
    base = mf["baseline"]
    species = [s for s in R.species_in(df[keep]) if s in maps]
    cases = pick(df, A, keep, species)

    md, summary = [], []
    for s, i, confident in cases:
        r = df.iloc[i]
        p = float(np.exp(A["base_a"][i]))
        rival = r.rival
        ab = float(A["base_a"][i] - A["base_b"][i])
        med, q25, q75, n_int = field_input(r.chip, int(r.field))
        stem = f"{s.replace(' ', '_')}_{r.field}"
        title = f"{s}, field {r.field} ({r.class_name.replace('_', ' ')}, p = {p:.3f})"
        fig_input(OUT / f"{stem}_input.png", title, med, q25, q75, mf["species_mean"][mf_species.index(s)], base)
        xi = {lv: {m: X[lv][m][i] for m in METHODS[lv]} for lv in METHODS}
        top_auc = {lv: [int(j) for j in np.argsort(-maps[s][lv], kind="stable")[:D.K[lv]]] for lv in D.LEVELS}
        fig_drivers(OUT / f"{stem}_drivers.png", title, xi, auc_cells[s], top_auc)

        rows = []
        for lv in D.LEVELS:
            row = {"level": f"{lv}s (AUC top {D.K[lv]} of {D.N[lv]})", "AUC main cues": ", ".join(
                f"{D.unit_name(lv, j)} {maps[s][lv][j]:.2f}" for j in top_auc[lv])}
            for meth in ["SHAP", "IG", "Occlusion (mean)", "Occlusion (drop)"]:
                if meth not in xi[lv]:
                    row[meth] = "–"
                    continue
                v = xi[lv][meth]
                j = int(np.argmax(v))
                hit = j in top_auc[lv]
                row[meth] = f"{'**' if hit else ''}{D.unit_name(lv, j)}{'**' if hit else ''} ({v[j]:.2f}; AUC {maps[s][lv][j]:.2f})"
            rows.append(row)
        tab = pd.DataFrame(rows)
        hits = {f"{lv}, {m}": int(np.argmax(xi[lv][m])) in top_auc[lv] for lv in METHODS for m in METHODS[lv]}
        def named(lv, m):
            j = int(np.argmax(xi[lv][m]))
            return f"**{D.unit_name(lv, j)}**" if j in top_auc[lv] else D.unit_name(lv, j)

        summary.append({"species": s, "field": int(r.field), "class": r.class_name.replace("_", " "), "p": round(p, 3),
                        "interior pixels": n_int, "SHAP month": named("month", "SHAP"),
                        "SHAP band group": named("band group", "SHAP"), "IG cell": named("cell", "IG"),
                        "occlusion cell": named("cell", "Occlusion (mean)"),
                        "AUC best cue": f"{D.unit_name('cell', top_auc['cell'][0])} ({maps[s]['cell'][top_auc['cell'][0]]:.2f})",
                        "main drivers that are main AUC cues": f"{sum(hits.values())} of {len(hits)}"})
        md.append((s, r, p, confident, rival, ab, n_int, stem, tab))

    S = pd.DataFrame(summary)
    text = [HEADER.format(n=len(cases)), "## The fields at a glance\n",
            "**Bold**: the main driver is one of the species' main AUC cues (top 3 months, top 2 band groups, top 5 "
            "cells).\n", R.md(S, ".3f"), "\n", SUMMARY, "\n"]
    for s, r, p, confident, rival, ab, n_int, stem, tab in md:
        text += [f"## {s.capitalize()}: field {r.field}, chip `{r.chip}`\n",
                 f"{r.class_name.replace('_', ' ')}, {n_int} interior pixels, **p = {p:.3f}**"
                 f"{'' if confident else ' (no correct field of this species reaches p = 0.95; its most confident one)'}"
                 f"; rival {rival}, {ab:.2f} in log p, so {s} is {np.exp(ab):,.0f} times more likely.\n",
                 f"![input](figures/fields/{stem}_input.png)\n", f"![drivers](figures/fields/{stem}_drivers.png)\n",
                 "Main driver of each method, with its value and its AUC; **bold** = one of the species' main AUC cues.\n",
                 R.md(tab), "\n", f"**What it tells us.** {READINGS.get(s, '')}\n"]
    (HERE / "FIELD_CASES.md").write_text("\n".join(text))
    print(S.to_string(index=False))


READINGS = {
    "grassland": "All methods agree that the model recognises this grassland by its green canopy in late spring: NIR, "
    "May and June. NIR is one of the data's two main groups (AUC 0.84), so the band group agrees; the month does not. "
    "The data's best grassland cues are August to November VV (lower backscatter, AUC 0.88 to 0.89), and the field "
    "shows that cue (October VV −10.8 dB against −9.2 dB for the reference), but no method finds the model using it.",
    "forage legumes": "The model's evidence is June red-edge (SHAP and occlusion; red-edge is the data's second group, "
    "0.71); IG puts it in May, one of the data's months. The data itself has no sharp cue for forage legumes: its best "
    "cells are VV in May and in autumn, all 0.72 to 0.74. Occlusion's months rest on falls of 0.01 and say nothing.",
    "wheat": "The clearest agreement. Every method puts the evidence in June, SHAP and occlusion on red-edge, and "
    "occlusion's top cell is June red-edge, the species' best cue (AUC 0.83). The field shows it plainly: June B05 is "
    "824 against 1,363 for the reference (a dense, chlorophyll-rich canopy absorbs more). IG names June NIR, the same "
    "canopy seen through the correlated NIR band.",
    "barley": "No main driver is one of the AUC main cues, but the misses are near ties: barley has no strong single "
    "cue (best 0.74), and the model's July (0.73) and visible (0.74) are as good in the data as the top ones. The "
    "data's August SWIR cue (ripe or harvested crop) is strongly present in the field (B12 2,654 against 1,421) and "
    "unused.",
    "rye": "The three methods agree on red-edge, one of the data's two main groups (0.79); IG's top cell, May "
    "red-edge, is the rye canopy's chlorophyll, close to the data's best cells (June SWIR 0.82, June red-edge 0.79). "
    "The month disagrees: the model looks at July, the data at May and June.",
    "oats": "The least confident case (p = 0.90). The methods find June and July, the data's months, but read them "
    "through NIR and visible, while the data's main cue is July VV (lower backscatter, 0.81), which the field shows "
    "(−12.2 dB against −10.5 dB).",
    "rapeseed": "June, the data's best month, for every method. The data's best cues are June radar (VV 0.95, VH 0.94) "
    "and the field shows them (VV −7.1 dB against −10.9 dB), but the model leads with red-edge and NIR, the canopy of "
    "May and June; SHAP does rank VH second among the band groups. The jump of the visible and SWIR bands in August is "
    "the harvest.",
    "dry pulses": "The three methods agree on red-edge in June and July (AUC 0.73). The data's main cue is June radar "
    "(VH 0.82, VV 0.79), present in the field (VH −15.5 dB against −16.9 dB) but not what the model uses.",
    "buckwheat": "The methods split. SHAP gives July red-edge, both among the data's cues, and occlusion's top cell is "
    "July red-edge too (0.89, one of the five best). IG points to June NIR, low because buckwheat is sown late and "
    "has little canopy in June (2,651 against 3,784 for the reference), itself a good cue (0.81). The data's best cue, "
    "September VH, is present in the field and unused.",
    "maize": "Red-edge, the data's main group, for all three methods. IG's top cell, June red-edge (0.87), is nearly "
    "as strong as the best (0.92): in June the maize field is still mostly bare (B05 1,952 against 1,363). The "
    "model's month, July, is the summer month where no single band tells maize apart well (0.74), so the model's July "
    "evidence comes from bands combined.",
}

SUMMARY = """**What the cases show.** The methods agree with one another on when: SHAP and IG name the same month
in 6 of the 10 fields and neighbouring months in the other 4, all between May and July. The model's evidence
is optical in every case, mostly red-edge and NIR. Against the AUC map, the band group agrees most often (SHAP: 6 of 10 fields), the
month less (SHAP: 4 of 10) and single cells rarely (occlusion: wheat and buckwheat; IG: none, though its top cells
are good cues, AUC 0.56 to 0.87). Where the data's strongest cue is radar (grassland, forage legumes, oats,
rapeseed, dry pulses, buckwheat), the cue is visible in the field's own curves, yet the model's main drivers are
optical; for barley the AUC map is flat, so its misses are near ties.
"""

HEADER = """# Field cases: what drives a confident prediction, by each method, against the AUC map

Generated by `field_cases.py` from the explanations of this experiment ([REPORT.md](REPORT.md)). One field per
species, {n} in all: among the fields the model predicts correctly with p ≥ 0.95, the one with the most interior
pixels (a species with none takes its most confident correct field).

**How to read the numbers.** Scores are the field's mean log p over its interior; p = e^score. SHAP and IG values,
and the falls of occlusion, are differences of log p, so e^value is a factor: 0.7 doubles, 2.3 multiplies by ten,
a negative value works against the species. SHAP and IG start from the species-balanced training mean (the grey
dotted curve of each input figure) and share out the difference between the field's score and that reference's;
occlusion gives the fall of the score when one block alone is hidden.

**Each field has two figures.** *The input*: the monthly composites as TESSERA gets them, per band (median over the
interior pixels, the middle half shaded), against the species' training mean and the methods' reference. *The
drivers*: IG and occlusion per month × band group cell beside the species' AUC map (squares: its five best cells;
dot: the method's top cell), then each method's months and band groups, scaled to their largest value so the
rankings can be compared (grey: the species' best AUC months and groups).
"""


if __name__ == "__main__":
    main()
