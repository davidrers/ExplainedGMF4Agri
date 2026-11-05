# Parcels without the oracle extent: WorldCover and the whole chip, TESSERA frozen and fine-tuned, 5 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it. It follows
[`../2026-10-18_parcel_delineation/NEXT_non_oracle_extent.md`](../2026-10-18_parcel_delineation/NEXT_non_oracle_extent.md),
as asked by the user on 5 November 2026: "following the approach of the md file using the ESA WorldCover; it does not
make sense to induce a bias by including a ground-truth mask", "only in TESSERA frozen and TESSERA fine-tuned", "all
of Estonia and enough epochs".

**Question.** The delineation experiment built every parcel inside **E**, the pixels of the declared in-scheme
parcels: an oracle first stage that gives away every outer edge, every isolated parcel and the absence of false
parcels outside the fields. Without it, built inside ESA WorldCover's grassland and cropland, or inside the whole
chip, how well do the boundary-head parcels of TESSERA delineate, how much does fine-tuning the encoder help, and
what does the boundary head add over each extent's own floor?

**What it varies.**

| Arm | Encoder | Fit |
|---|---|---|
| `tessera_v2_medium` | frozen | none: the delineation experiment's cells (`results/2026-10-18_parcel_delineation/<set>/tessera_v2_medium/`), rescored |
| `tessera_v2_medium_ft_bnd` (`arms_run/`) | fine-tuned at 1e-5 on the labelled pixels, as `2026-10-15_kshot_finetune` | `run.py`: the per-pixel MLP with the boundary head and joint loss of the delineation experiment |

Both: K = 100 % only (no budget sweep at this stage: the question is whether the method takes us anywhere; user,
5 November 2026), draw 0, seed 0, 45 epochs, checkpoint of lowest validation loss (crop-type plus boundary loss),
all of `EE_2021_rev`. The fine-tuned fits are new; the frozen ones are reused, so the two differ in the
encoder and in the route (labelled pixels instead of whole chips, which changes only the MLP's batch-norm statistics;
in `kshot_finetune` that moved validation Macro-F1 by less than 0.005).

**Extents** (`panoptic_x.py`), each with the variants `extent` (its 4-connected regions, its floor), `semantic`,
`boundary@tau` and `combined@tau` for tau in 0.3, 0.5, 0.7:

| Extent | Taken from | Reads as |
|---|---|---|
| `oracle` | E, the declared in-scheme parcels | upper bound, the delineation experiment's |
| `worldcover` | ESA WorldCover 2021 v200, grassland (30) and cropland (40) | what a first stage could give |
| `worldcover_30_40_90` | the same and herbaceous wetland (90) | sensitivity: 90 holds 3 % of our grassland pixels |
| `chip` | the whole chip | lower bound |

**Scoring** uses the truth only after the parcels are built: true pieces, void (declared parcels of a class outside
the scheme) and background (no declared parcel), with the void rule of the handoff note: void pixels leave every
predicted parcel, a predicted parcel more than half void is dropped, every other unmatched one is an FP. With X = E
the scorer reproduces the stored scores of all 18 delineation cells (`check.py --regression`). The boundary pixel F1
stays inside E.

**WorldCover** (`fetch_worldcover.py`): the three v200 tiles over Estonia from the Planetary Computer, nearest
neighbour onto every chip's grid, in `data/worldcover/EE_2021_rev/` (outside the chip set). On the test chips,
95.8 % of the parcel pixels lie in WorldCover 30 or 40, but 57 % of WorldCover grassland lies on no declared parcel
(`worldcover_stats.py`): the undeclared-grassland leak is large and is reported.

**How to run.**

    poetry run python experiments/2026-11-05_parcel_extent/fetch_worldcover.py --chips EE_2021_rev
    poetry run python experiments/2026-11-05_parcel_extent/worldcover_stats.py --chips EE_2021_rev
    poetry run python experiments/2026-11-05_parcel_extent/check.py --regression EE_2021_rev
    poetry run python experiments/2026-11-05_parcel_extent/run.py -e experiments/2026-11-05_parcel_extent/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
    bash scripts/cluster/submit.sh experiments/2026-11-05_parcel_extent/experiment_k100.yaml EE_2021_rev       # cluster
    bash scripts/cluster/pull_results.sh 2026-11-05_parcel_extent EE_2021_rev
    poetry run python experiments/2026-11-05_parcel_extent/score.py --chips EE_2021_rev       # every cell, every extent
    poetry run python experiments/2026-11-05_parcel_extent/report.py --chips EE_2021_rev

**Results.** `results/2026-11-05_parcel_extent/<set>/`: the fine-tuned cells, and for both arms
`extent_scores.json` per cell.

**Outcome.** Pending.
