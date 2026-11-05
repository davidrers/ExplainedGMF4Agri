# Full PaPs (utae-paps) on TESSERA, frozen and fine-tuned, 5 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it. Asked by the user on
5 November 2026: "a second experiment using the full PaPs approach (https://github.com/VSainteuf/utae-paps), only in
TESSERA frozen and TESSERA fine-tuned, all of Estonia and enough epochs", to see whether the method takes us
somewhere before the other GFMs get it.

**Question.** Does the Parcels-as-Points module, trained as its authors train it, delineate the parcels of Estonia
from TESSERA's 10 m embedding, with no extent given, and does an encoder fine-tuned on crop type do better than the
frozen one? How does it compare with the boundary-head parcels of
[`../2026-11-05_parcel_extent/`](../2026-11-05_parcel_extent/README.md) in the same extents?

**What it varies.**

| Arm (`arms_run/`) | Raster (`rasters.py`) |
|---|---|
| `paps_tessera_v2_frozen` | the frozen TESSERA v2 Medium raster cache of the core |
| `paps_tessera_v2_ft` | every chip re-encoded with the TESSERA student fine-tuned on crop type in `2026-10-15_kshot_finetune` at the same K, draw and seed, then frozen (the user's choice of 5 November 2026; fine-tuning end to end through PaPs would cost about 90 GPU hours per budget) |

**The model** (`paps_model.py`, `paps_loss.py`, vendored from utae-paps, MIT). TESSERA encodes each pixel's time
series alone, so PaPs needs a spatial backbone: U-TAE's spatial encoder and decoder (widths 64-64-64-128 down,
32-32-64-128 up), without its temporal attention, which TESSERA's embedding replaces; then PaPs as published:
centerness heatmap, saliency, and per centre the class, the box size and a 16 x 16 shape, with the pseudo-NMS.

**Training** (`fit_paps.py`), utae-paps' defaults: Adam at 0.01, x 0.3 after epochs 60 and 80, **100 epochs**,
batch 4, 5 warm-up epochs on the heatmap alone, one supervised centre per parcel, D4 augmentation. Validation every
5 epochs: the pseudo-NMS confidence is tuned on the validation chips and the checkpoint is the epoch of best
validation PQ (void rule, whole chip), as the source keeps its best-PQ epoch.

**Targets without the truth** (`targets.py`). Parcels are the 4-connected pieces of the drawn in-scheme parcels,
under the label budget; centres the pixel nearest each piece's centroid; Gaussian kernels of 1/20 of the box,
zones by the argmax of the kernels, as the paper defines them. The only background is ESA WorldCover's
non-agricultural land (`data/worldcover/EE_2021_rev/`, from `../2026-11-05_parcel_extent/fetch_worldcover.py`);
WorldCover grassland and cropland outside the drawn parcels is unknown, neither parcel nor background. The declared
parcel layer is never used to say where background is, at any K.

**Scoring** (`score_paps.py`, with `panoptic_x.py` copied from the extent experiment): PaPs' parcels over the whole
chip (`chip/paps`), and the same parcels cut to WorldCover (`worldcover/paps`, `worldcover_30_40_90/paps`) and to the
declared parcels (`oracle/paps`, the upper bound), by the void rule.

**Protocol.** K = 100, 20 and 5 %, draw 0, seed 0, all of `EE_2021_rev`. One cluster job per arm and budget
(`experiment_k<K>.yaml`).

**How to run.**

    poetry run python experiments/2026-11-05_paps_tessera/check.py --chips EE_2021_rev_mini
    poetry run python experiments/2026-11-05_paps_tessera/run.py -e experiments/2026-11-05_paps_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
    for k in 100 20 5; do CPUS=16 MEM=100G bash scripts/cluster/submit.sh experiments/2026-11-05_paps_tessera/experiment_k$k.yaml EE_2021_rev; done   # cluster
    bash scripts/cluster/pull_results.sh 2026-11-05_paps_tessera EE_2021_rev
    poetry run python experiments/2026-11-05_paps_tessera/run.py -e experiments/2026-11-05_paps_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --summarise

**Results.** `results/2026-11-05_paps_tessera/<set>/<arm>/P<k>_draw0_seed0/`: `results.json` (with
`panoptic_metrics` per extent and the training history), `metrics.csv`, `predictions_test.npz`, `best-pq.pt`.

**Outcome.** Pending.
