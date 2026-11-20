"""Figures and tables of the SAE study, from ``results/<name>/EE_2021_rev/eval`` and the trained runs.

    poetry run python experiments/2026-11-20_sae_tessera_store/report.py

Writes ``figures/*.png`` and ``figures/tables.md`` in this folder.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path.insert(1, str(REPO / "src"))

from run import CFG, CHIPS, EMB_SUFFIX, OUT, read_band_stack  # noqa: E402

FIG = HERE / "figures"
EVD = OUT / "eval"

# Categorical slots in fixed order (validated: scripts/validate_palette.js of the dataviz skill, light mode).
SAE, FROZEN, RAW, PCA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REP_COLOR = {"sae": SAE, "frozen": FROZEN, "raw": RAW, "pca": PCA}
REP_LABEL = {"sae": "SAE feature", "frozen": "frozen-dictionary feature", "raw": "raw dimension",
             "pca": "PCA component"}
SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "legumes_harvested_green": "green legumes", "spring_barley": "spring barley", "oats": "oats",
         "clover": "clover", "winter_rapeseed_rape": "winter rapeseed", "spring_common_soft_wheat": "spring wheat",
         "peas": "peas", "potatoes": "potatoes", "winter_barley": "winter barley", "beans": "beans",
         "fresh_vegetables": "vegetables", "rye": "rye", "alfalfa_lucerne": "alfalfa",
         "spring_rapeseed_rape": "spring rapeseed", "buckwheat": "buckwheat",
         "grain_maize_corn_popcorn": "maize", "legumes_dried_pulses_protein_crops": "dry pulses",
         "orchards_fruits": "orchards", "outside_fields": "outside fields"}
MONTHS = list("JFMAMJJASOND")

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})


def short(n: str) -> str:
    return SHORT.get(n, n.replace("_", " "))


def save(fig, name: str) -> None:
    fig.savefig(FIG / name)
    plt.close(fig)
    print("wrote", FIG / name)


def load():
    s = json.loads((EVD / "summary.json").read_text())
    a = dict(np.load(EVD / "arrays.npz"))
    return s, a


# ---------------------------------------------------------------------------------------------------- figures
def fig_reconstruction(s):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    rec = s["reconstruction"]
    for meth, col, lab in (("batchtopk", SAE, "SAE"), ("frozen", FROZEN, "frozen dictionary"),
                           ("pca", PCA, "PCA, first L0 components"), ("raw", RAW, "raw, L0 dimensions")):
        rows = [r for r in rec if r["method"] == meth and (meth not in ("batchtopk", "frozen") or r["run"].endswith("s0"))]
        rows.sort(key=lambda r: r["k"])
        x, y = [r["l0"] for r in rows], [r["fve"] for r in rows]
        ax.plot(x, y, "-o", color=col, lw=2, ms=6, label=lab)
        ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=8)
    extra = [r for r in rec if r["method"] == "batchtopk" and not r["run"].endswith("s0")]
    ax.plot([r["l0"] for r in extra], [r["fve"] for r in extra], "o", mfc="white", mec=SAE, ms=7,
            label="SAE, seeds 1 and 2")
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32], ["8", "16", "32"])
    ax.set_xlabel("active features (or kept components) per pixel, L0")
    ax.set_ylabel("variance of the embedding explained")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Reconstruction of the test embeddings", loc="left", fontsize=10)
    save(fig, "fig1_reconstruction.png")


def fig_alignment(s, a):
    concepts = [short(c) for c in s["concepts"]]
    bins = [(0, 0, "0"), (1, 1, "1"), (2, 3, "2–3"), (4, 7, "4–7"), (8, 99, "8+")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    w = 0.2
    for i, rep in enumerate(("sae", "frozen", "raw", "pca")):
        fa = a[f"auroc_{rep}"]
        live = fa.max(1) > CFG["evaluate"]["live_auroc"]
        n = (fa[live] >= CFG["evaluate"]["align_auroc"]).sum(1)
        share = [((n >= lo) & (n <= hi)).mean() for lo, hi, _ in bins]
        ax.bar(np.arange(len(bins)) + (i - 1.5) * w, share, w * 0.9, color=REP_COLOR[rep],
               label=f"{REP_LABEL[rep]} ({live.sum()} live)")
    ax.set_xticks(range(len(bins)), [b[2] for b in bins])
    ax.set_xlabel(f"concepts a unit is aligned with (folded AUROC ≥ {CFG['evaluate']['align_auroc']})")
    ax.set_ylabel("share of live units")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("How many crops one unit marks", loc="left", fontsize=10)
    for ax, rep in ((axes[1], "sae"), (axes[2], "raw")):
        fa, up = a[f"auroc_{rep}"], a[f"auroc_up_{rep}"]
        m = np.zeros((len(concepts), len(concepts)))
        for i in range(len(concepts)):
            score = np.where(up[:, i], fa[:, i], 0)
            m[i] = fa[int(np.argmax(score))]
        im = ax.imshow(m, cmap="Blues", vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(concepts)), concepts, rotation=90, fontsize=7)
        ax.set_yticks(range(len(concepts)), concepts if rep == "sae" else [], fontsize=7)
        ax.grid(False)
        ax.set_title(f"Best {REP_LABEL[rep]} per concept (row):\nits folded AUROC for every concept",
                     loc="left", fontsize=9)
    fig.subplots_adjust(wspace=0.12)
    fig.colorbar(im, ax=axes[1:], shrink=0.7, label="folded AUROC")
    p = axes[0].get_position()
    axes[0].set_position([p.x0, p.y0, p.width - 0.045, p.height])
    save(fig, "fig2_alignment.png")


def fig_single(s):
    su = s["single_unit"]
    classes = sorted(su["sae"], key=lambda c: su["sae"][c]["f1"])
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(classes) + 1.2))
    y = np.arange(len(classes))
    h = 0.2
    for i, rep in enumerate(("sae", "frozen", "raw", "pca")):
        f1 = [su[rep][c]["f1"] for c in classes]
        ax.barh(y + (1.5 - i) * h, f1, h * 0.9, color=REP_COLOR[rep],
                label=f"{REP_LABEL[rep]} (mean {np.mean(f1):.2f})")
    ax.set_yticks(y, [short(c) for c in classes])
    ax.set_xlabel("test F1 of the single best unit (chosen on validation)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("One unit as a crop detector, field interiors", loc="left", fontsize=10)
    save(fig, "fig3_single_unit.png")


def family_columns(s, n_per=3):
    fam = s["families"]["sae"]["families"]
    cls = [c for c in s["scoring_sets"]["supported"] if fam.get(c)]
    cols, owner = [], []
    for c in cls:
        for j in fam[c][:n_per]:
            cols.append(j)
            owner.append(c)
    return cls, cols, owner


def fig_heatmap(s, a):
    names = s["classes"]
    sup = s["scoring_sets"]["supported"]
    cls, cols, owner = family_columns(s)
    cm = a["class_mean_test"][[names.index(c) for c in sup]][:, cols]
    cm = cm / np.maximum(cm.max(0, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(0.2 * len(cols) + 2.5, 0.28 * len(sup) + 1.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(sup)), [short(c) for c in sup], fontsize=8)
    ax.set_xticks(range(len(cols)), [f"f{j}" for j in cols], rotation=90, fontsize=6)
    ax.grid(False)
    edges = [i for i in range(1, len(owner)) if owner[i] != owner[i - 1]]
    for e in edges:
        ax.axvline(e - 0.5, color="white", lw=2)
    ax.set_xlabel("the three strongest features of each class's family, grouped by the class")
    ax.set_title("Where each feature fires: its mean activation per class, divided by its largest",
                 loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8, label="share of the feature's largest class mean")
    save(fig, "fig4_feature_class.png")


def fig_profiles(s, a, key: str, label: str, fname: str):
    names = s["classes"]
    su = s["single_unit"]["sae"]
    cls = sorted(su, key=lambda c: -su[c]["f1"])
    ncol = 4
    nrow = int(np.ceil(len(cls) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), sharex=True, sharey=True)
    pf, pc = a[f"profile_feature_{key}"], a[f"profile_class_{key}"]
    sup_idx = [names.index(c) for c in s["scoring_sets"]["supported"]]
    allf = np.nanmean(pc[sup_idx], 0)
    comp = a["composition_test"]
    for ax, c in zip(axes.flat, cls):
        j, ci = su[c]["unit"], names.index(c)
        ax.plot(range(12), allf, color=GRID, lw=3, label="mean over all classes")
        ax.plot(range(12), pc[ci], "--", color=INK2, lw=1.5, label="all pixels of the class")
        ax.plot(range(12), pf[j], color=SAE, lw=2, label="pixels where the feature is active")
        ax.set_title(f"{short(c)}, feature f{j}\n{100 * comp[j, ci]:.0f} % of its active pixels are {short(c)}",
                     fontsize=8, loc="left")
        ax.set_xticks(range(12), MONTHS)
    for ax in list(axes.flat)[len(cls):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for ax in axes[:, 0]:
        ax.set_ylabel(label)
    fig.suptitle(f"{label} through 2021 under the best single SAE feature of each class (test field interiors)",
                 x=0.01, y=0.997, ha="left", fontsize=10)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.982), ncol=3, fontsize=8,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.968))
    save(fig, fname)


def macro(v):
    return v  # the convention is chosen in fig_head


def head_convention(s):
    logged = s["head"]["test_macro_f1_logged"]
    e = s["full_test"]["variants"]["embedding"]["all"]
    conv = min(("macro_f1_present", "macro_f1_all20"), key=lambda k: abs(e[k] - logged))
    return conv, logged


def fig_head(s):
    conv, logged = head_convention(s)
    ft = s["full_test"]["variants"]
    order = [("embedding", "the embedding itself", INK2), (CFG["main"], "SAE reconstruction (k = 16)", SAE),
             (CFG["control"], "frozen-dictionary reconstruction", FROZEN), ("pca16", "first 16 PCA components", PCA)]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for gi, scope in enumerate(("all", "interior")):
        for i, (v, lab, col) in enumerate(order):
            val = ft[v][scope][conv]
            ax.barh(gi * 5 + i, val, 0.85, color=col, label=lab if gi == 0 else None)
            ax.text(val + 0.005, gi * 5 + i, f"{val:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks([1.5, 6.5], ["all labelled\ntest pixels", "field\ninteriors"])
    ax.invert_yaxis()
    ax.set_xlim(0, max(ft["embedding"]["all"][conv], ft["embedding"]["interior"][conv]) * 1.15)
    ax.set_xlabel("test Macro-F1 of the trained crop head")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "fig5_head_on_reconstruction.png")


def fig_ablation(s):
    ab = s["ablation"]
    cls = sorted(ab, key=lambda c: ab[c]["own_drop"])
    fig, ax = plt.subplots(figsize=(6.8, 0.33 * len(cls) + 1.3))
    y = np.arange(len(cls))
    ax.barh(y, [100 * ab[c]["own_drop"] for c in cls], 0.6, color=SAE, label="its own family removed")
    nm = np.array([100 * ab[c]["null_mean"] for c in cls])
    nq = np.array([100 * ab[c]["null_q975"] for c in cls])
    ax.errorbar(nm, y, xerr=[np.zeros_like(nm), nq - nm], fmt="o", color=INK2, ms=4, capsize=2,
                label="random features of the same number (mean, 97.5th percentile)")
    ax.plot([100 * ab[c]["others_mean_drop"] for c in cls], y, "D", color=FROZEN, ms=4,
            label="mean drop of the other classes")
    ax.set_yticks(y, [f"{short(c)} ({ab[c]['family_size']})" for c in cls])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("drop in the class's test recall, percentage points")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Removing a class's features from the embedding before the crop head\n"
                 "(family size in brackets)", loc="left", fontsize=10)
    save(fig, "fig6_ablation.png")


def fig_families(s, a):
    f = s["families"]
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    rows = [("trained crop head (MLP on the embedding)", f["head"]["macro_f1"], INK2),
            ("SAE feature families", f["sae"]["macro_f1"], SAE),
            ("frozen-dictionary families", f["frozen"]["macro_f1"], FROZEN)]
    for i, (lab, v, col) in enumerate(rows):
        ax.barh(i, v, 0.6, color=col)
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    null = a["family_null_sae"]
    ax.barh(3, np.median(null), 0.6, color=GRID)
    ax.errorbar([np.median(null)], [3], xerr=[[np.median(null) - np.quantile(null, 0.025)],
                                              [np.quantile(null, 0.975) - np.median(null)]],
                fmt="none", color=INK2, capsize=3)
    ax.text(np.quantile(null, 0.975) + 0.005, 3, f"{np.median(null):.3f}", va="center", fontsize=8)
    ax.set_yticks(range(4), [r[0] for r in rows] + ["random SAE families, same sizes (median, 95 %)"])
    ax.invert_yaxis()
    ax.set_xlabel("test Macro-F1, field interiors")
    ax.set_xlim(0, 1)
    ax.set_title("A crop map from feature families alone", loc="left", fontsize=10)
    save(fig, "fig7_families.png")


def fig_stability(s, a):
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    bins = np.linspace(0, 1, 41)
    for n, col in zip(CFG["seeds"], (SAE, PCA)):
        ax.hist(a[f"match_cos_{n}"], bins=bins, histtype="step", lw=2, color=col,
                label=f"closest atom in seed {n[-1]}")
    best = s["stability"]["class_best_features"]
    for c, v in best.items():
        ax.plot([v[CFG["seeds"][0]]], [-3], "|", color=INK, ms=10)
    rng = np.random.default_rng(0)
    r1, r2 = rng.standard_normal((2, 1024, 128))
    r1 /= np.linalg.norm(r1, axis=1, keepdims=True)
    r2 /= np.linalg.norm(r2, axis=1, keepdims=True)
    chance = float(np.median((r1 @ r2.T).max(1)))
    ax.axvline(chance, color=INK2, ls=":", lw=1.2)
    ax.text(0.55, ax.get_ylim()[1] * 0.6, f"dotted line: closest of 1,024 random\ndirections (median {chance:.2f})",
            fontsize=7.5, color=INK2)
    ax.set_xlim(0.15, 1.0)
    ax.set_xlabel("cosine between a feature of seed 0 and its closest feature in another seed")
    ax.set_ylabel("features of seed 0")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Do the features come back with another seed? (ticks: each class's best feature)",
                 loc="left", fontsize=9)
    save(fig, "fig8_stability.png")


def fig_training():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for n, r in CFG["runs"].items():
        log = json.loads((OUT / "runs" / n / "log.json").read_text())
        col = SAE if r["variant"] == "batchtopk" else FROZEN
        ls = {8: ":", 16: "-", 32: "--"}[r["k"]]
        ax.plot([e["step"] for e in log], [e["val"]["fve"] for e in log], ls, color=col, lw=1.5,
                label=f"{r['variant']} k={r['k']} seed {r['seed']}")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation FVE")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Training curves", loc="left", fontsize=10)
    save(fig, "figA_training.png")


def fig_maps(s):
    import torch

    from stages import load_model, norm_stats, to_sae

    names = s["classes"]
    su = s["single_unit"]
    sc = np.load(EVD / "scoring_test.npz")
    targets = [c for c in ("winter_rapeseed_rape", "pasture_meadow_grassland_grass", "winter_common_soft_wheat")
               if c in su["sae"]]
    tidx = [names.index(c) for c in targets]
    chip_ids = sc["chip_ids"]
    score = {}
    for ch in np.unique(sc["chip"]):
        m = (sc["chip"] == ch) & (sc["kind"] == 1)
        score[ch] = sum(min(int((sc["label"][m] == t).sum()), 300) for t in tidx)
    chips = [chip_ids[c] for c, _ in sorted(score.items(), key=lambda kv: -kv[1])[:3]]
    ns = norm_stats()
    model = load_model(CFG["main"])
    raw_unit = su["raw"][targets[0]]
    fig, axes = plt.subplots(len(chips), 3 + len(targets), figsize=(2.3 * (3 + len(targets)), 2.4 * len(chips)))
    lab_colors = [SAE, RAW, PCA]
    for r, cid in enumerate(chips):
        e = read_band_stack(CHIPS / "chips" / f"{cid}{EMB_SUFFIX}").reshape(128, -1).T
        valid = np.isfinite(e).all(1)
        z = np.zeros((len(e), model.latent_dim), np.float32)
        z[valid] = model.codes(to_sae(e[valid], ns)).numpy()
        xs = np.full((len(e), 128), np.nan, np.float32)
        xs[valid] = to_sae(e[valid], ns).numpy()
        s2 = read_band_stack(CHIPS / "chips" / f"{cid}_merged.tif").astype(np.float32)
        rgb = np.stack([s2[5 * 12 + b] for b in (3, 2, 1)], -1)
        rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)
        mask = read_band_stack(CHIPS / "chips" / f"{cid}.mask.tif")[0]
        lab = np.ones(mask.shape + (3,)) * 0.93
        lab[mask >= 0] = 0.75
        for t, col in zip(tidx, lab_colors):
            lab[mask == t] = matplotlib.colors.to_rgb(col)
        panels = [("June true colour", rgb, None), ("declared crops", lab, None)]
        for c in targets:
            j = su["sae"][c]["unit"]
            panels.append((f"SAE f{j} ({short(c)})", z[:, j].reshape(224, 224), "Blues"))
        v = raw_unit["sign"] * xs[:, raw_unit["unit"]]
        panels.append((f"raw dim {raw_unit['unit']} ({short(targets[0])})", v.reshape(224, 224), "Greys"))
        for ax, (title, img, cmap) in zip(axes[r], panels):
            ax.imshow(img, cmap=cmap, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(title, fontsize=8)
        axes[r, 0].set_ylabel(cid, fontsize=7)
    handles = [matplotlib.patches.Patch(color=c, label=short(t)) for t, c in zip(targets, lab_colors)]
    handles.append(matplotlib.patches.Patch(color=(0.75,) * 3, label="other declared crops"))
    fig.tight_layout()
    fig.legend(handles=handles, loc="upper left", ncol=4, fontsize=8, bbox_to_anchor=(0.01, 0.0))
    save(fig, "fig9_maps.png")


# ------------------------------------------------------------------------------------------------------ tables
def tables(s, a) -> str:
    out = []
    t = out.append
    t("# Tables\n")
    t("## Reconstruction, test pixels (uniform sample)\n")
    t("| method | run | k | L0 | FVE | cosine | dead share |\n|---|---|---|---|---|---|---|")
    for r in sorted(s["reconstruction"], key=lambda r: (r["method"], r["k"], r.get("run", ""))):
        t(f"| {r['method']} | {r.get('run', '')} | {r['k']} | {r['l0']:.1f} | {r['fve']:.3f} | {r['cosine']:.3f} | "
          f"{r.get('dead_share', float('nan')):.3f} |")
    t("\n## Concept alignment, test (folded AUROC ≥ 0.65)\n")
    t("| representation | units | live | live aligned with ≤ 1 concept | exactly 1 | none | median concepts per live unit | "
      "concepts with an aligned unit |\n|---|---|---|---|---|---|---|---|")
    for rep, v in s["alignment"].items():
        t(f"| {REP_LABEL[rep]} | {v['units']} | {v['live']} | {v['share_live_at_most_one']:.2f} | "
          f"{v['share_live_exactly_one']:.2f} | {v['share_live_none']:.2f} | {v['median_aligned_live']:.0f} | "
          f"{v['concepts_with_an_aligned_unit']} of {len(s['concepts'])} |")
    t("\n## Best single unit per class, test field interiors (precision / recall / F1)\n")
    reps = ["sae", "frozen", "raw", "pca"]
    t("| class | test pixels | " + " | ".join(REP_LABEL[r] for r in reps) + " |\n|---|---|" + "---|" * len(reps))
    su = s["single_unit"]
    for c in sorted(su["sae"], key=lambda c: -su["sae"][c]["f1"]):
        cells = []
        for r in reps:
            v = su[r][c]
            cells.append(f"{'f' if r in ('sae', 'frozen') else 'd'}{v['unit']}: {v['precision']:.2f} / "
                         f"{v['recall']:.2f} / **{v['f1']:.2f}**")
        t(f"| {short(c)} | {s['scoring_sets']['test_pixels_per_class'][c]} | " + " | ".join(cells) + " |")
    t("\n## Feature families and the crop map from families alone, test field interiors\n")
    f = s["families"]
    t(f"Live SAE features: {f['sae']['live_features']}. Macro-F1: SAE families {f['sae']['macro_f1']:.3f} "
      f"(random families: median {f['sae']['null_median']:.3f}, 97.5th percentile {f['sae']['null_q975']:.3f}, "
      f"p = {f['sae']['null_p']:.3f}); frozen-dictionary families {f['frozen']['macro_f1']:.3f} "
      f"(random: median {f['frozen']['null_median']:.3f}); the trained head on the same pixels "
      f"{f['head']['macro_f1']:.3f}.\n")
    t("| class | SAE family size | SAE families F1 | frozen family size | frozen families F1 | head F1 |\n|---|---|---|---|---|---|")
    for c in s["scoring_sets"]["supported"]:
        t(f"| {short(c)} | {f['sae']['sizes'][c]} | {f['sae']['per_class_f1'][c]:.2f} | {f['frozen']['sizes'][c]} | "
          f"{f['frozen']['per_class_f1'][c]:.2f} | {f['head']['per_class_f1'][c]:.2f} |")
    conv, logged = head_convention(s)
    ft = s["full_test"]
    t(f"\n## The crop head on reconstructions, all {ft['chips']} test chips\n")
    t(f"Agreement of the exported head with the stored test predictions: {ft['stored_prediction_agreement']:.5f}. "
      f"Macro-F1 convention `{conv}` (the logged test Macro-F1 is {logged:.4f}).\n")
    t("| input to the head | Macro-F1, all labelled pixels | accuracy | Macro-F1, field interiors | accuracy |\n|---|---|---|---|---|")
    for v, r in ft["variants"].items():
        t(f"| {v} | {r['all'][conv]:.4f} | {r['all']['accuracy']:.4f} | {r['interior'][conv]:.4f} | "
          f"{r['interior']['accuracy']:.4f} |")
    t("\n## Ablation of each class's family, test field interiors\n")
    t("| class | family size | recall before | own drop | strongest feature only | random sets: mean / 97.5 % | p | "
      "other classes' mean drop | its pixels go to |\n|---|---|---|---|---|---|---|---|---|")
    ab = s["ablation"]
    for c in sorted(ab, key=lambda c: -ab[c]["own_drop"]):
        v = ab[c]
        goes = ", ".join(f"{short(k)} {100 * x:.0f} %" for k, x in v["goes_to"].items())
        t(f"| {short(c)} | {v['family_size']} | {v['base_recall']:.2f} | {100 * v['own_drop']:.1f} pp | "
          f"{100 * v['own_drop_strongest_only']:.1f} pp | {100 * v['null_mean']:.1f} / {100 * v['null_q975']:.1f} pp | "
          f"{v['null_p']:.2f} | {100 * v['others_mean_drop']:.1f} pp | {goes} |")
    t("\n## Stability across seeds\n")
    st = s["stability"]
    for n in CFG["seeds"]:
        t(f"- {n}: share of seed-0 features with a closest atom at cosine ≥ 0.9: {st[n]['share_ge_0.9']:.2f}; "
          f"≥ 0.7: {st[n]['share_ge_0.7']:.2f}")
    t("\n| class | best SAE feature (seed 0) | closest cosine, seed 1 | seed 2 | test F1 seed 0 / 1 / 2 |\n|---|---|---|---|---|")
    for c, v in st["class_best_features"].items():
        f1s = " / ".join(f"{su[r][c]['f1']:.2f}" for r in ("sae", *CFG["seeds"]))
        t(f"| {short(c)} | f{su['sae'][c]['unit']} | {v[CFG['seeds'][0]]:.2f} | {v[CFG['seeds'][1]]:.2f} | {f1s} |")
    t("\n## Scoring sets\n")
    t("| class | test interior pixels | test fields | validation interior pixels | scored |\n|---|---|---|---|---|")
    for c in s["classes"]:
        t(f"| {short(c)} | {s['scoring_sets']['test_pixels_per_class'][c]} | "
          f"{s['scoring_sets'].get('test_fields_per_class', {}).get(c, '')} | "
          f"{s['scoring_sets']['val_pixels_per_class'][c]} | {'yes' if c in s['scoring_sets']['supported'] else 'no'} |")
    return "\n".join(out) + "\n"


def firing_table(a) -> str:
    """Where the main SAE's features fire on the uniform test pixels (weighted), and how often they return in the
    other seeds by firing frequency."""
    import torch
    from stages import load_model, norm_stats, to_sae
    torch.set_grad_enabled(False)
    d = np.load(OUT / "samples" / "test.npz")
    u = d["kind"] == 0
    lab, inter, w = d["label"][u], d["interior"][u], d["weight"][u].astype(np.float64)
    m = load_model(CFG["main"])
    on = (m.codes(to_sae(d["emb"][u], norm_stats())).numpy() > 0).astype(np.float64)

    def share(sel):
        return on[sel].T @ w[sel] / w[sel].sum()

    tot, fi, out = share(np.ones(len(lab), bool)), share((lab >= 0) & inter), share(lab < 0)
    outside = (out > 3 * np.maximum(fi, 1e-9)) & (out >= 1e-3)
    L = ["## Where the main SAE's features fire, uniform test pixels (weighted)", "",
         f"- active on at least 1 in 1,000 field-interior pixels: {int((fi >= 1e-3).sum())}",
         f"- mainly outside fields (at least 3 times as often as on interiors, and on at least 1 in 1,000 "
         f"outside pixels): {int(outside.sum())}",
         f"- active on fewer than 1 in 1,000 of all pixels: {int((tot < 1e-3).sum())}",
         "- (the groups overlap)", "",
         "| active on this share of all pixels | features | median closest cosine, " + " / ".join(CFG["seeds"])
         + " | share at cosine ≥ 0.7 |", "|---|---|---|---|"]
    for lo, hi, lab_ in ((0, 1e-3, "below 0.1 %"), (1e-3, 1e-2, "0.1 to 1 %"), (1e-2, 1.01, "1 % or more")):
        sel = (tot >= lo) & (tot < hi)
        mc = [np.minimum(a[f"match_cos_{n}"], 1)[sel] for n in CFG["seeds"]]
        L.append(f"| {lab_} | {int(sel.sum())} | " + " / ".join(f"{np.median(x):.2f}" for x in mc) + " | "
                 + " / ".join(f"{np.mean(x >= 0.7):.2f}" for x in mc) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    FIG.mkdir(exist_ok=True)
    s, a = load()
    fig_reconstruction(s)
    fig_alignment(s, a)
    fig_single(s)
    fig_heatmap(s, a)
    fig_head(s)
    fig_ablation(s)
    fig_families(s, a)
    fig_stability(s, a)
    fig_training()
    fig_profiles(s, a, "ndvi", "NDVI", "fig10_profiles_ndvi.png")
    fig_profiles(s, a, "vh", "VH backscatter (dB)", "fig11_profiles_vh.png")
    fig_maps(s)
    (FIG / "tables.md").write_text(tables(s, a) + "\n" + firing_table(a))
    print("wrote", FIG / "tables.md")


if __name__ == "__main__":
    main()
