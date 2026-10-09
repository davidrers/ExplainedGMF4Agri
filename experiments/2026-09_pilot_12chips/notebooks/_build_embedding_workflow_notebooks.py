"""Build the embedding-workflow pilot notebooks, for TerraMind, Prithvi and THOR.

    poetry run python notebooks/terratorch/_build_embedding_workflow_notebooks.py
    poetry run python notebooks/terratorch/_build_embedding_workflow_notebooks.py thor

writes ``eurocrops_terramind_pilot_embeddings.ipynb``, ``eurocrops_prithvi_pilot_embeddings.ipynb``
and ``eurocrops_thor_pilot_embeddings.ipynb`` beside this script, or only those named. The first
two are the two-stage counterparts of ``eurocrops_terramind_pilot.ipynb`` and
``eurocrops_prithvi_pilot.ipynb``, which keep the end-to-end workflow and stay as the reference.
THOR has no end-to-end notebook; its reference fits are run by ``scripts/seg/train.py``.

A notebook is written unexecuted, so building one replaces its outputs; execute it afterwards.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SPECS = {
    "terramind": {
        "name": "TerraMind",
        "config": "experiments/2026-09_pilot_12chips/configs/terramind_v1_small_ee_pilot.yaml",
        "reference": "eurocrops_terramind_pilot.ipynb",
        "out": "eurocrops_terramind_pilot_embeddings.ipynb",
        "coords": False,
        "alternatives": "`terramind_v1_large_ee_pilot.yaml`, `terramind_v1_small_s2s1_ee_pilot.yaml` or "
                        "`terramind_v1_large_s2s1_ee_pilot.yaml`",
        "encoder_note": "TerraMind is a single-date model, so TerraTorch's temporal wrapper runs it "
                        "once per month and concatenates the twelve monthly token maps on the channel "
                        "axis: 12 x 384 = 4,608 channels per layer for the small variant.",
    },
    "prithvi": {
        "name": "Prithvi-EO-2.0",
        "config": "experiments/2026-09_pilot_12chips/configs/prithvi_eo_v2_300_tl_ee_pilot.yaml",
        "reference": "eurocrops_prithvi_pilot.ipynb",
        "out": "eurocrops_prithvi_pilot_embeddings.ipynb",
        "coords": True,
        "alternatives": "`prithvi_eo_v2_600_tl_ee_pilot.yaml`",
        "encoder_note": "Prithvi is natively multi-temporal: the twelve months enter one ViT as "
                        "12 x 196 tokens, and `ReshapeTokensToImage` folds them back into a map with "
                        "the months on the channel axis, 12 x 1,024 = 12,288 channels per layer for "
                        "the 300M variant. The TL variant also takes the date of every month and the "
                        "location of the chip, which matters for stage 1, see section 4.",
    },
    "thor": {
        "name": "THOR",
        "config": "experiments/2026-09_pilot_12chips/configs/thor_v1_large_ee_pilot.yaml",
        "reference": None,
        "out": "eurocrops_thor_pilot_embeddings.ipynb",
        "coords": False,
        "alternatives": "a copy of `thor_v1_large_ee_pilot.yaml` with `model.backbone: thor_v1_base`",
        "what": "THOR v1 large on the pilot chips, run through TerraTorch's second workflow exactly as "
                "[`eurocrops_terramind_pilot_embeddings.ipynb`](eurocrops_terramind_pilot_embeddings.ipynb) "
                "runs TerraMind. The first workflow keeps the frozen encoder inside the training loop, so "
                "every epoch of every fit re-encodes every chip although the encoder never changes; for "
                "THOR those end-to-end fits are run by `scripts/seg/train.py` with the configuration "
                "below, into `results/seg/thor_v1_large_ee_pilot/`, and are the reference of section 7.",
        "setup_note": "**THOR is not part of TerraTorch.** It comes through its own extension, "
                      "`thor_terratorch_ext`, which `scripts/env/install_thor.sh` installs outside the "
                      "Poetry lock, and the backbone registry imports it only when a THOR entry is built. "
                      "The first forward pass in a process takes about a minute, because THOR compiles its "
                      "attention-bias function with `torch.compile`.",
        "encoder_note": "THOR is a single-date model, so TerraTorch's temporal wrapper runs it once per "
                        "month, as for TerraMind. Within a month it tokenises each spectral group on its "
                        "own: the four 10 m bands (B02, B03, B04, B08) at 16 px patches and the six 20 m "
                        "bands (B05, B06, B07, B8A, B11, B12) at 8 px, so both groups give a 14 x 14 grid "
                        "of 160 m tokens and attend to each other in one sequence of 392. B01 and B09, "
                        "THOR's 60 m group, do not tile the 2,240 m chip and are left out. Its TerraTorch "
                        "wrapper concatenates the two groups' token maps, as the THOR authors do for dense "
                        "tasks, and the temporal wrapper the twelve months: 12 x 2 x 1,024 = 24,576 "
                        "channels per layer for the large variant.",
        "history": "**Augmentation in both workflows.** Both data pipelines draw the D4 symmetry from "
                   "PyTorch's generator. That is the fix for a fault this comparison found in the TerraMind "
                   "and Prithvi notebooks, where the end-to-end pipeline replayed the same few symmetries "
                   "every epoch; section 7 of "
                   "[`eurocrops_terramind_pilot_embeddings.ipynb`](eurocrops_terramind_pilot_embeddings.ipynb) "
                   "tells it. A regression test in `tests/test_segmentation_data.py` fails on the old draw.",
        "sec4_note": "The cached features agree with the in-loop ones to the precision the encoder runs "
                     "at: under 16-bit autocast a relative error of one to one and a half parts in a thousand, "
                     "growing slightly with depth, at a correlation above 0.99999 in every layer. Against "
                     "that, a different chip differs by nearly its own size.",
        "sec5_note": "Apart from the identity, which matches by construction, rotating the features is a "
                     "poor stand-in for encoding the rotated chip, although a better one than for TerraMind "
                     "or Prithvi, whose pilot caches correlate at 0.81 to 0.92 under the same test. THOR "
                     "has no positional embedding: its ALiBi bias depends only on the distance between two "
                     "tokens, which a rotation of the chip preserves. What does not rotate is the patch "
                     "embedding, a learned kernel per band that sees the pixels of each patch in one "
                     "orientation, and every token passes through it. A relative error of a third to two "
                     "fifths is still far from the precision of section 4, so training on rotated features "
                     "would show the decoder inputs it never meets at test time. Every rotation is "
                     "therefore encoded here as well, at eight times the stage 1 cost for the training chips.",
        "sec7_note": "**For THOR** the ranges overlap at every budget, and at 100 % the two workflows agree "
                     "within a hundredth. At 5 % and 20 % the two-stage mean is lower by two to four "
                     "hundredths, and at 5 % the ranges only just touch. Differences of that size, in either "
                     "direction, also appear in the TerraMind and Prithvi notebooks, where Prithvi's "
                     "two-stage fits score higher at 20 %, so the pilot cannot tell them from seed "
                     "variation. Both low budgets falling on the same side is the thing to watch when THOR "
                     "is run on the full split, where the test set is large enough to resolve it.",
        "cost_note": "* The two-stage workflow removes loading the raw chip and running the encoder from "
                     "every fit, and replaces both with reading the cached features back, plus a one-time "
                     "encoding and the disk to hold it (section 8). For THOR large the encoder is about "
                     "two fifths of the per-chip cost the end-to-end fit pays, and loading the raw chip most of "
                     "the rest. Reading the cache back costs about half of what loading the raw chip does, "
                     "although each cached chip is 38.5 MB, twice TerraMind large's, because THOR hands "
                     "over the tokens of two spectral groups. The extrapolation to Estonia counts compute "
                     "only: TerraMind large's stage 1, at half these bytes, was bound by writing to the "
                     "network file system and took 11.5 h, so THOR's 1.6 TB would take about a day.",
    },
}

#: The text of the TerraMind and Prithvi notebooks where a spec does not give its own.
WHAT = ("The same experiment as [`{ref}`]({ref}), run through TerraTorch's second workflow. That "
        "notebook keeps the frozen encoder inside the training loop, so every epoch of every fit "
        "re-encodes every chip although the encoder never changes.")
HISTORY = """**What this comparison first found.** Its first run showed the two-stage fits scoring higher at every budget, with the seed ranges not even touching at 5 %. The features and the model were identical, so the difference had to lie in how the decoder was trained, and it did: the end-to-end data pipeline drew its D4 augmentation from albumentations' private random generator, which is never reseeded in the DataLoader workers that are created afresh at every epoch, so every epoch replayed the same few symmetries. Logged over three epochs of a real fit, nine worker processes produced three distinct sequences between them. The two-stage dataset drew from a generator that is reseeded, and so augmented properly. Both now draw from PyTorch's generator, the end-to-end runs below were repeated with the fix, and the fix raised their Macro-F1 by one to four hundredths. A regression test in `tests/test_segmentation_data.py` fails on the old draw."""
SEC5_NOTE = """Apart from the identity, which matches by construction, rotating the features is a poor stand-in for encoding the rotated chip: the ViT sees different patches in a different order and its positional embeddings do not rotate with them. Training on rotated features would therefore show the decoder inputs it never meets at test time, and training on the identity alone would drop the augmentation the end-to-end fit has. Encoding all eight keeps the two workflows on the same footing, at eight times the stage 1 cost for the training chips."""
SEC4_NOTE = """The cached features agree with the in-loop ones to the precision the encoder runs at. Under 16-bit autocast that is a relative error of about one part in a thousand; the same comparison in full float32, run while this workflow was being built, agreed to four parts in ten thousand for TerraMind and exactly for Prithvi. Against that, a different chip differs by more than half its own size."""
COST_NOTE = """* The two-stage workflow removes loading the raw chip and running the encoder from every fit, and replaces both with reading the cached features back, plus a one-time encoding and the disk to hold it (section 8). In the measurements the larger part of the saving is the **loading**, not the encoder: decoding a deflate-compressed 144-band GeoTIFF and augmenting it costs several times more per chip than reading a flat float16 array, and for a small encoder more than the forward pass itself. The encoder's share of the saving grows with the backbone. Part of the same gain would therefore also be open to the end-to-end workflow, through a chip format that is cheaper to decode."""


def build(key: str) -> nbf.NotebookNode:
    s = SPECS[key]
    cells: list = []
    md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))  # noqa: E731
    code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))  # noqa: E731

    what = s.get("what") or WHAT.format(ref=s["reference"])
    stays = " The reference notebook stays as it is." if s["reference"] else ""
    setup_note = f"\n\n{s['setup_note']}" if s.get("setup_note") else ""
    md(f"""
# {s['name']} on the EuroCrops pilot chips, the embedding workflow

**Thesis.** *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions*, David Reyes, ITC, University of Twente.

**What this is.** {what} Here the encoder runs **once per chip**, through TerraTorch's `EmbeddingGenerationTask`, and only the trainable part of the model is trained from the stored features.{stays}

```
end-to-end   chip ──► frozen encoder ──► frozen necks ──► trainable necks ──► decoder ──► head      every step
two-stage    chip ──► frozen encoder ──► frozen necks ──► disk                                     once per chip
                                                   disk ──► trainable necks ──► decoder ──► head    every step
```

**The two train the same model.** The encoder is frozen, so its output for a given input is fixed, and caching it removes only the repetition. This notebook does not take that on trust: section 4 compares the cached features with the ones the encoder produces inside the training loop, and section 7 compares the metrics of the two workflows over several seeds.

**Kernel.** Python 3.11 (gfm4agri). The pilot chips must exist, see `scripts/data/build_pilot_chips.py`. The machinery lives in `src/gfm4agri/benchmark/cached.py`; this notebook calls it and shows what it does.{setup_note}

**Another backbone.** Point `CONFIG` below at {s['alternatives']} and everything follows.
""")
    code("""
import json, sys, time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.benchmark.backbones import get_backbone
from gfm4agri.benchmark.cached import (EMB_SUFFIX, IDENTITY, VARIANTS, CachedFeatureDataModule,
                                       apply_d4, embedding_model_args, fit_cached,
                                       generate_embeddings, split_at_cache_point, variant_name)
from gfm4agri.benchmark.segmentation import build_task
from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

CONFIG = REPO / """ + repr(s["config"]) + """
cfg = yaml.safe_load(CONFIG.read_text())
BACKBONE = cfg["model"]["backbone"]
ROOT = REPO / cfg["data"]["root"]
CACHE = REPO / "data" / "embeddings" / BACKBONE / ROOT.name     # stage 1 output
E2E = REPO / "results" / "seg" / cfg["run_name"]                # end-to-end runs, the reference
TWO = REPO / "results" / "seg_cached" / cfg["run_name"]         # the runs of this notebook
BUDGETS = [5, 20, 100]                                          # per cent of each class's parcels
SEEDS = [0, 1, 2]
REBUILD_CACHE = False                                           # True re-encodes an existing cache
print(BACKBONE, "|", ROOT.relative_to(REPO), "->", CACHE.relative_to(REPO))
""")

    md("""
## 1. The cache point

The end-to-end model is a chain of modules, and the chain splits in two at the first one with trainable parameters. Everything before that point is frozen and deterministic, so its output can be computed once and stored. Everything after it is what the label budget trains.
""")
    code("""
dm = EuroCropsSegDataModule(ROOT, BACKBONE, normalisation=cfg["data"]["normalisation"],
                            batch_size=1, num_workers=0, augment=False)
frozen, trainable = split_at_cache_point(get_backbone(BACKBONE).model_args(dm.bands, dm.n_timesteps))
print("stage 1, run once per chip :  encoder ->", " -> ".join(n["name"] for n in frozen))
print("stage 2, trained per budget:", " -> ".join(n["name"] for n in trainable), "-> decoder -> head")
embedding_model_args(BACKBONE, dm.bands, dm.n_timesteps)
""")
    md(f"""
{s['encoder_note']} So what is stored for every chip is, for each of the four selected encoder layers, a `(channels, 14, 14)` map: one vector per 16 x 16 pixel patch, with the months stacked on the channel axis.
""")

    md("""
## 2. Stage 1: encode every chip once

`generate_embeddings` runs TerraTorch's `EmbeddingGenerationTask` over the chips with the frozen head of the model from section 1, under the same autocast precision as the end-to-end fit, so the stored features are the ones the decoder would have seen.

**Augmentation has to be encoded, not applied afterwards.** The end-to-end fit draws one of the eight D4 symmetries of each training chip at every step. A ViT is not equivariant to rotation, so the features of a rotated chip are not the rotated features of the chip, and section 5 measures by how much. Training chips are therefore encoded in all eight variants; validation chips once, unrotated. The features are stored as float16 arrays, one file per chip, variant and layer, in TerraTorch's folder layout.
""")
    code("""
if REBUILD_CACHE or not (CACHE / "cache.json").exists():
    cache = generate_embeddings(BACKBONE, ROOT, CACHE, normalisation=cfg["data"]["normalisation"],
                                batch_size=4, num_workers=4, precision=cfg["trainer"]["precision"])
else:
    cache = json.loads((CACHE / "cache.json").read_text())
per = cache["encode_seconds"] / cache["chip_encodings"]
print(f"{cache['chip_encodings']} chip encodings in {cache['encode_seconds']:.0f} s ({per:.2f} s each, model loading included)")
print(f"{cache['bytes'] / 1e6:.0f} MB on disk, {cache['bytes'] / 1e6 / cache['chip_encodings']:.1f} MB per chip and variant, {cache['dtype']}")
print(f"layers {cache['encoder_layers']} -> channels {cache['channel_list']}, feature map {cache['feature_size']}")
pd.DataFrame(cache["runs"])
""")

    md("""
## 3. What the stored features look like

One training chip, three of its eight variants. The left column is the chip as the encoder saw it, true colour for June. The right column is the deepest cached layer, reduced to three principal components shown as colour, one pixel per 16 x 16 patch. Parcels that the chip shows in one colour tend to share a colour in the feature map, and the map turns with the chip.
""")
    code("""
import rasterio

tids = (ROOT / "training_data.txt").read_text().split()
CHIP = tids[0]
with rasterio.open(ROOT / "training_chips" / f"{CHIP}_merged.tif") as src:
    june = src.read().reshape(12, 12, 224, 224)[5][[3, 2, 1]].transpose(1, 2, 0).astype(float)
rgb = np.clip(june / np.percentile(june, 98), 0, 1)
last = len(cache["channel_list"]) - 1
feats = {v: np.load(CACHE / v / f"layer_{last:02d}" / f"{CHIP}{EMB_SUFFIX}").astype(np.float32)
         for v in ("k0", "k1", "k2f")}
flat = np.concatenate([f.reshape(f.shape[0], -1).T for f in feats.values()])
mean = flat.mean(0)
_, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
lo, hi = np.percentile((flat - mean) @ vt[:3].T, [2, 98], axis=0)

fig, axes = plt.subplots(3, 2, figsize=(6, 9))
for row, (v, f) in enumerate(feats.items()):
    k, flip = int(v[1]), v.endswith("f")
    pcs = ((f.reshape(f.shape[0], -1).T - mean) @ vt[:3].T).reshape(*f.shape[1:], 3)
    axes[row, 0].imshow(apply_d4(rgb, k, flip))
    axes[row, 1].imshow(np.clip((pcs - lo) / (hi - lo), 0, 1), interpolation="nearest")
    axes[row, 0].set_ylabel(f"variant {v}")
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 0].set_title("chip, June true colour"); axes[0, 1].set_title(f"cached layer {cache['encoder_layers'][last]}, 3 PCs")
fig.suptitle(CHIP); plt.tight_layout(); plt.show()
""")

    coords_md = ""
    coords_code = ""
    if s["coords"]:
        coords_md = """

**Prithvi TL needs one more thing.** Its encoder takes the date of every month and the location of the chip as inputs of their own. TerraTorch's stock `EmbeddingGenerationTask` calls the model with the image alone, so it would encode Prithvi without them and store features the decoder never sees in training. `cached.py` passes them through. The last cell of this section shows what dropping them would have cost."""
        coords_code = """

# What TerraTorch's stock task would have stored: the same encoder, called without the
# date and location inputs.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AUTOCAST):
    no_coords = full.neck[: len(frozen)](full.encoder(x), image_size=(224, 224))
a, c = feats_loop[-1][0].float().cpu().numpy(), no_coords[-1][0].float().cpu().numpy()
print(f"without date and location: relative error {np.linalg.norm(a - c) / np.linalg.norm(a):.2e}, "
      f"correlation {np.corrcoef(a.ravel(), c.ravel())[0, 1]:.4f} against the in-loop features")
"""
    md(f"""
## 4. Are the cached features the ones the training loop sees?

For one validation chip, the features are computed the way the end-to-end fit computes them, through the full model's encoder and frozen necks, and compared with the stored file. The measure is the relative error, the size of the difference over the size of the features, together with the correlation. For scale, the last line compares the stored chip with a *different* chip, which is what a genuine mismatch looks like.{coords_md}
""")
    code("""
AUTOCAST = "16" in cfg["trainer"]["precision"]
dm.setup("fit")
vids = [Path(f).name[: -len("_merged.tif")] for f in dm.val_dataset.image_files]

def in_loop(i):
    item = dm.val_dataset[i]
    b = dm.aug({k: (v[None] if torch.is_tensor(v) else [v]) for k, v in item.items()})
    x = dm.split_modalities(b["image"])
    x = x.cuda() if torch.is_tensor(x) else {k: v.cuda() for k, v in x.items()}
    kw = {k: b[k].cuda() for k in ("temporal_coords", "location_coords") if k in b}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AUTOCAST):
        f = full.neck[: len(frozen)](full.encoder(x, **kw), image_size=(224, 224))
    return f, x

full = build_task(BACKBONE, num_classes=len(dm.class_names), class_names=dm.class_names,
                  bands=dm.bands, n_timesteps=dm.n_timesteps).model.cuda().eval()
feats_loop, x = in_loop(0)
rows = []
for layer, f in enumerate(feats_loop):
    a = f[0].float().cpu().numpy()
    c = np.load(CACHE / "k0" / f"layer_{layer:02d}" / f"{vids[0]}{EMB_SUFFIX}").astype(np.float32)
    rows.append({"layer": cache["encoder_layers"][layer], "shape": a.shape,
                 "relative error": np.linalg.norm(a - c) / np.linalg.norm(a),
                 "correlation": np.corrcoef(a.ravel(), c.ravel())[0, 1]})
other = np.load(CACHE / "k0" / f"layer_{len(feats_loop) - 1:02d}" / f"{vids[1]}{EMB_SUFFIX}").astype(np.float32)
a = feats_loop[-1][0].float().cpu().numpy()
print(f"reference, a different chip: relative error {np.linalg.norm(a - other) / np.linalg.norm(a):.2e}, "
      f"correlation {np.corrcoef(a.ravel(), other.ravel())[0, 1]:.4f}")
pd.DataFrame(rows).style.format({"relative error": "{:.2e}", "correlation": "{:.7f}"})
""" + coords_code)
    code("""
# Encoder cost per chip, measured here for section 8: forward passes over the four validation
# chips as one batch, under the fit's autocast.
items = [dm.val_dataset[i] for i in range(len(vids))]
b = dm.aug({k: (torch.stack([it[k] for it in items]) if torch.is_tensor(items[0][k]) else [it[k] for it in items])
            for k in items[0]})
xb = dm.split_modalities(b["image"])
xb = xb.cuda() if torch.is_tensor(xb) else {k: v.cuda() for k, v in xb.items()}
kwb = {k: b[k].cuda() for k in ("temporal_coords", "location_coords") if k in b}
def encode():
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=AUTOCAST):
        return full.neck[: len(frozen)](full.encoder(xb, **kwb), image_size=(224, 224))
for _ in range(2):
    encode()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(5):
    encode()
torch.cuda.synchronize()
ENC_S_PER_CHIP = (time.time() - t0) / 5 / len(items)
print(f"encoder and frozen necks: {ENC_S_PER_CHIP * 1000:.0f} ms per chip")
""")
    md(s.get("sec4_note", SEC4_NOTE))

    md("""
## 5. Why every rotation is encoded

If the encoder were equivariant to rotation, the features of a rotated chip would equal the rotated features of the chip, and storing one variant would be enough. The cell below tests that on the cache itself: for every variant, the stored features are compared with the identity features rotated the same way.
""")
    code("""
rows = []
for k, flip in VARIANTS:
    v = variant_name(k, flip)
    stored = np.load(CACHE / v / f"layer_{last:02d}" / f"{CHIP}{EMB_SUFFIX}").astype(np.float32)
    rotated = apply_d4(feats["k0"].transpose(1, 2, 0), k, flip).transpose(2, 0, 1)
    rows.append({"variant": v, "relative error": np.linalg.norm(stored - rotated) / np.linalg.norm(stored),
                 "correlation": np.corrcoef(stored.ravel(), rotated.ravel())[0, 1]})
pd.DataFrame(rows).style.format({"relative error": "{:.3f}", "correlation": "{:.4f}"})
""")
    md(s.get("sec5_note", SEC5_NOTE))

    md("""
## 6. Stage 2: train the decoder from the cache

`fit_cached` builds the end-to-end model, keeps exactly its trainable modules, and trains them through TerraTorch's `SemanticSegmentationTask` with the configuration of the end-to-end run: the same label budget draw, optimiser, learning rate, weight decay, dropout, loss, precision, number of epochs, effective batch and validation checkpointing. It writes `results.json` with the same fields as `scripts/seg/train.py`, plus `workflow` and a summary of the cache it read. Fits that already exist are read back rather than repeated.
""")
    code("""
full = None                     # the end-to-end model of section 4 is not needed any more
torch.cuda.empty_cache()

def run(pct, seed):
    out = TWO / f"P{pct:g}_draw0_seed{seed}" / "results.json"
    if out.exists():
        return json.loads(out.read_text())
    return fit_cached(dict(cfg, seed=seed), CACHE, pct=pct, num_workers=4)

two = {(p, s): run(p, s) for s in SEEDS for p in BUDGETS}
t = two[(BUDGETS[-1], SEEDS[0])]
print(f"trainable parameters: {t['params']['trainable_total']:,}; fit at {BUDGETS[-1]} %: {t['fit_seconds']:.0f} s")
""")

    md("""
## 7. Are the metrics the same?

The two workflows compute their metrics identically: the same TerraTorch task, the same 47 metrics, the same `ignore_index`, the same test chips. Whether they reach the same *values* can only be judged against the variation of the end-to-end runs themselves: repeating an identical end-to-end configuration with another seed moves Macro-F1 by a few hundredths on these twelve chips, because the decoder is initialised and fed in a different random order and the rare classes flip on and off. Two workflows that train the same model should therefore overlap in their spread over seeds, not agree to the third decimal.

The label draw is fixed at draw 0 throughout, so the seeds vary the training only.

""" + s.get("history", HISTORY))
    code("""
def load_e2e(pct, seed):
    f = E2E / f"P{pct:g}_draw0_seed{seed}" / "results.json"
    return json.loads(f.read_text()) if f.exists() else None

rows = []
for p in BUDGETS:
    for wf, get in (("end-to-end", lambda p, s: load_e2e(p, s)), ("two-stage", lambda p, s: two.get((p, s)))):
        for s in SEEDS:
            r = get(p, s)
            if r:
                rows.append({"K %": p, "workflow": wf, "seed": s, "Macro-F1": r["test_metrics"]["test/F1_Score"],
                             "mIoU": r["test_metrics"]["test/mIoU"], "fit s": r["fit_seconds"]})
res = pd.DataFrame(rows)
summary = res.groupby(["K %", "workflow"]).agg(
    seeds=("seed", "count"), f1_mean=("Macro-F1", "mean"), f1_min=("Macro-F1", "min"), f1_max=("Macro-F1", "max"),
    miou_mean=("mIoU", "mean"), fit_s=("fit s", "mean")).round(4)
summary
""")
    code("""
fig, ax = plt.subplots(figsize=(6, 4))
for wf, marker, dx in (("end-to-end", "o", -0.6), ("two-stage", "s", 0.6)):
    d = res[res["workflow"] == wf]
    ax.scatter(d["K %"] * np.exp(dx / 20), d["Macro-F1"], marker=marker, alpha=0.6, label=f"{wf}, one per seed")
    m = d.groupby("K %")["Macro-F1"].mean()
    ax.plot(m.index * np.exp(dx / 20), m.values, marker=marker, linewidth=1)
ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels([f"{b} %" for b in BUDGETS])
ax.set_xlabel("label budget, per cent of each class's parcels"); ax.set_ylabel("Macro-F1 on the validation chips")
ax.set_title(f"{BACKBONE}: the two workflows over seeds"); ax.legend(frameon=False); plt.tight_layout(); plt.show()

overlap = []
for p in BUDGETS:
    a, b = res[(res["K %"] == p) & (res.workflow == "end-to-end")]["Macro-F1"], res[(res["K %"] == p) & (res.workflow == "two-stage")]["Macro-F1"]
    if len(a) and len(b):
        overlap.append(f"{p} %: ranges {'overlap' if max(a.min(), b.min()) <= min(a.max(), b.max()) else 'do NOT overlap'} "
                       f"(end-to-end {a.min():.3f} to {a.max():.3f}, n = {len(a)}; two-stage {b.min():.3f} to {b.max():.3f}, n = {len(b)})")
print("\\n".join(overlap))
""")
    md("""
Read the table and the plot together. Where the two ranges overlap, the difference between the workflows is inside the variation either one shows on its own, and the two cannot be told apart. A range built from a single end-to-end seed is a point, so overlap with it is a weak test; three or more seeds on each side are what make the comparison mean something.
""" + (f"\n{s['sec7_note']}" if s.get("sec7_note") else ""))

    md("""
## 8. What it costs, and what it would cost for Estonia

Stage 1 is paid once per chip, variant and backbone. Stage 2 is paid per fit, and the Phase 1 protocol calls for many fits: every label budget, several draws of every budget, every backbone. The end-to-end workflow pays the encoder inside every one of them.

The pilot's own fit times are no guide to that. Twelve chips cannot amortise the fixed cost of a fit, so dividing a pilot fit time by its chips attributes model loading, validation and checkpointing to each chip. The cell below therefore measures the three per-chip costs that scale with the data: encoding a chip, reading its cached features, and one training step of the decoder. It extrapolates from those, charging each workflow for everything it does per chip and epoch: the end-to-end fit **loads and augments the raw chip, encodes it and trains**; the two-stage fit **reads the cached features and trains**. Loading runs in DataLoader workers alongside the GPU in both, so both figures are upper bounds in the same way. Timings are indicative only, since the GPU is shared with another container that at times slows every kernel, and reads from the network file system vary.
""")
    code("""
from gfm4agri.benchmark.cached import build_cached_model

# the end-to-end fit loads and augments the raw chip every epoch
raw = EuroCropsSegDataModule(ROOT, BACKBONE, normalisation=cfg["data"]["normalisation"],
                             batch_size=1, num_workers=0, augment=True)
raw.setup("fit")
t0 = time.time()
for i in range(16):
    raw.train_dataset[i % len(raw.train_dataset)]
LOAD_S_PER_CHIP = (time.time() - t0) / 16

# the two-stage fit reads the cached features instead
cdm = CachedFeatureDataModule(ROOT, CACHE, batch_size=4, num_workers=0)
cdm.setup("fit")
t0 = time.time()
for i in range(16):
    cdm.train_dataset[i % len(cdm.train_dataset)]
READ_S_PER_CHIP = (time.time() - t0) / 16

m = build_cached_model(BACKBONE, num_classes=len(cdm.class_names), class_names=cdm.class_names,
                       bands=dm.bands, n_timesteps=dm.n_timesteps, channel_list=cache["channel_list"]).cuda()
opt, scaler = torch.optim.AdamW(m.parameters(), lr=1e-4), torch.amp.GradScaler()
batch = next(iter(cdm.train_dataloader()))
xs, ys = batch["image"].cuda(), batch["mask"].cuda()
def step():
    with torch.autocast("cuda", dtype=torch.float16, enabled=AUTOCAST):
        loss = torch.nn.functional.cross_entropy(m(xs).output, ys, ignore_index=cdm.ignore_index)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
for _ in range(3):
    step()
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20):
    step()
torch.cuda.synchronize()
DEC_S_PER_CHIP = (time.time() - t0) / 20 / xs.shape[0]
del m, opt; torch.cuda.empty_cache()

per_chip = pd.DataFrame({"seconds per chip": {
    "load and augment the raw chip (end-to-end)": LOAD_S_PER_CHIP,
    "encode, encoder and frozen necks (end-to-end)": ENC_S_PER_CHIP,
    "read cached features (two-stage)": READ_S_PER_CHIP,
    "decoder training step (both)": DEC_S_PER_CHIP}})
per_chip.style.format("{:.3f}")
""")
    code("""
# Full Estonia under the 4 x 4-chip block split blocks4_buf1600_seed0: 4,898 training chips,
# 769 validation and 1,475 test chips; every training chip in all eight D4 variants.
EE_TRAIN, EE_VAL, EE_TEST = 4898, 769, 1475
EPOCHS = cfg["trainer"]["max_epochs"]
chip_epochs = (EE_TRAIN + EE_VAL) * EPOCHS
encodings = EE_TRAIN * len(cache["train_variants"]) + EE_VAL + EE_TEST
mb = cache["bytes"] / 1e6 / cache["chip_encodings"]

stage1_h = encodings * (LOAD_S_PER_CHIP + ENC_S_PER_CHIP) / 3600   # every encoding loads its chip
two_fit_h = chip_epochs * (READ_S_PER_CHIP + DEC_S_PER_CHIP) / 3600
e2e_fit_h = chip_epochs * (LOAD_S_PER_CHIP + ENC_S_PER_CHIP + DEC_S_PER_CHIP) / 3600
print(f"stage 1, once: {encodings:,} encodings, {stage1_h:.1f} h, {encodings * mb / 1e3:.0f} GB")
print(f"one fit of {EPOCHS} epochs: two-stage {two_fit_h:.1f} h, end-to-end {e2e_fit_h:.1f} h")
saving = e2e_fit_h - two_fit_h
print(f"per fit, two-stage {'saves' if saving > 0 else 'costs'} {abs(saving):.1f} h; stage 1 pays for itself after "
      + (f"{stage1_h / saving:.1f} fits\\n" if saving > 0 else "no number of fits: reading the cache costs more than encoding\\n"))
rows = []
for fits in (3, 15, 50, 150):
    rows.append({"fits": fits, "end-to-end h": fits * e2e_fit_h, "two-stage h": stage1_h + fits * two_fit_h})
pd.DataFrame(rows).set_index("fits").style.format("{:.0f}")
""")
    md("""
## 9. What this establishes

* The cached features are the features the end-to-end fit trains on, to the precision the encoder runs at (section 4).
* Augmentation survives the move only because every rotation is encoded; rotating stored features would not reproduce it (section 5).
* The trainable model, its parameter count, the label draw, the optimisation and the metrics are identical by construction (section 6), and section 7 shows whether the resulting scores fall inside each other's seed spread.
""" + s.get("cost_note", COST_NOTE) + """
* The price is disk. The cache holds every training chip in all eight D4 variants, and its size grows with the width of the backbone's feature maps (section 8 gives the figure for Estonia), so it is the stage 1 storage, not its compute, that bounds how many backbones can be cached at once.

The caveat that carries over from the end-to-end notebooks still applies: the pilot is twelve chips with no spatial block split, and its test metrics are computed on the validation chips.
""")

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3.11 (gfm4agri)", "language": "python",
                                    "name": "gfm4agri"}
    nb["metadata"]["language_info"] = {"name": "python"}
    return nb


if __name__ == "__main__":
    import sys

    for key in sys.argv[1:] or SPECS:
        spec = SPECS[key]
        path = HERE / spec["out"]
        nbf.write(build(key), path)
        print(f"wrote {path.relative_to(HERE.parents[1])}")
