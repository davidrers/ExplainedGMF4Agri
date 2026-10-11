# How many D4 variants does TerraMind's decoder need? Half of Estonia, 9 October 2026

**Question.** Does the D4 augmentation add to the results? The two-stage fit encodes every training chip in all
eight D4 symmetries and the decoder draws one per sample, which costs eight times the training-chip cache. The
ablation of 8 October (`../2026-10-08_d4_ablation`, full Estonia, K = 5 % only, three seeds) gave test Macro-F1
0.304 +- 0.030 (identity), 0.335 +- 0.049 (four quarter turns) and 0.358 +- 0.003 (all eight): ordered, but with
spreads that overlap. This study repeats it at every budget of the main workflow, through the K-shot workflow.

**What it varies.** Only the variants the decoder trains on. Encoder, cache, decoder, optimiser, batch and epochs
are those of the core arm; validation and test always read the identity.

| Arm | Variants drawn in training | Where |
|---|---|---|
| `terramind_v1_large_d4x1` | `k0`: the identity, no augmentation | `arms/` |
| `terramind_v1_large_d4x4` | `k0 k1 k2 k3`: the four quarter turns, no flip | `arms/` |
| `terramind_v1_large` | all eight of D4, the main workflow | core arm, `configs/arms/` |

The core runner never passes `train_variants` to the cached fit, so `run.py` wraps the runner's `compose` to copy
the arm's `data.train_variants` into the cell's configuration, and the cached fit to read it from there. Each
`results.json` records it under `config.data.train_variants` and `feature_cache.train_variants_drawn`. All three
arms read the same cache, which holds all eight variants.

**Chip set.** `EE_2021_half`, cut from `EE_2021` with `scripts/hub/build_pilot_chipset.py --train-blocks 224
--val-blocks 63 --test-blocks 127` (seed 0): half of the 448 training blocks holding a trainable parcel, and every
validation and test block, so the test set is the full grid's 1,475 chips. That is 2,388 training, 769
validation and 1,475 test chips; 49 % of Estonia's 4,898 training chips. All 20 classes are in the training pool:
42,700 trainable parcels at K = 100 % and 2,142 at K = 5 %, where the rarest class (dried pulses and protein crops)
has 8. With eight variants per training chip and the identity for validation and test, the cache holds 21,348
chip encodings, about 0.41 TB.

**Protocol.** K = 100, 20 and 5 % of the parcels of each class, draw 0, seeds 0, 1 and 2, 15 epochs, checkpoint
of lowest validation loss (`experiment.yaml`). 27 fits, one cluster job per arm, each computing its own cache on
the node's local NVMe. Models are compared on the same draw and seeds, so the differences can be read in pairs.

**Checks on the hub, 9 October 2026.** `pytest` passes; the dry run lists all 27 cells and the chip-set check
passes; with the overrides applied, the cached fit receives `[k0]`, `[k0, k1, k2, k3]` and `None` (all eight) for
the three arms.

**How to run.**

    poetry run python experiments/2026-10-09_terramind_d4_count/run.py -e experiments/2026-10-09_terramind_d4_count/experiment.yaml --chips data/eurocrops_chips/EE_2021_half   # hub
    bash scripts/cluster/submit.sh experiments/2026-10-09_terramind_d4_count/experiment.yaml EE_2021_half                                                                    # cluster

**Results.** `results/2026-10-09_terramind_d4_count/EE_2021_half/`, read by `kshot_results.ipynb` in this folder: a copy of
`notebooks/pipeline/kshot_results.ipynb` with the two arms in its colour table and a table of paired differences.

**Run.** UT HPC, 9 October 2026, jobs 618042 (d4x1, node29, 1 h 37 min), 618044 (core arm, node30, 1 h 39 min)
and 618043 (d4x4, node30), at commit `389cf61`. 618043 failed after two cells: a DataLoader worker could not
allocate shared memory (`/dev/shm`) while node30 also ran the THOR 80 m and 40 m token-size jobs. Its partial
third cell was moved to `logs/failed_618043_P100_draw0_seed2/`, since a refit into the same folder would have
saved `best-loss-v1.ckpt` beside the stale `best-loss.ckpt` that the prediction step reads. The arm was resubmitted
with `--exclude=hpc-node30` as job 618104 (node29, 1 h 14 min), which re-encoded its cache and finished the other
seven cells at commit `0bedeeb`; that commit changed only `../2026-10-09_thor_token_size/`.

**Outcome.** Test Macro-F1 on the 1,475 test chips, mean of seeds 0 to 2, and the paired difference per seed, mean
+- standard deviation with the seeds won (`kshot_results.ipynb`):

| K | 1 variant | 4 variants | 8 variants | 8 - 1 | 4 - 1 | 8 - 4 |
|---|---|---|---|---|---|---|
| 100 % | 0.518 | 0.543 | 0.551 | +0.033 +- 0.010 (3/3) | +0.025 +- 0.011 (3/3) | +0.008 +- 0.006 (3/3) |
| 20 % | 0.368 | 0.407 | 0.438 | +0.070 +- 0.012 (3/3) | +0.040 +- 0.061 (2/3) | +0.031 +- 0.051 (2/3) |
| 5 % | 0.258 | 0.282 | 0.278 | +0.020 +- 0.070 (1/3) | +0.024 +- 0.047 (2/3) | -0.004 +- 0.061 (2/3) |

* **The augmentation adds at K = 100 and 20 %.** All eight beat the identity in every seed, by 0.033 and 0.070
  Macro-F1, several times the seed spread of the difference. At 100 % the ordering 1 < 4 < 8 holds in every seed,
  and the four quarter turns give three quarters of the gain.
* **At K = 5 % three seeds do not decide it.** Single fits range over 0.07 to 0.09 within an arm, which is larger
  than any difference between the means. The 8 October ablation on full Estonia at 5 % was noisy too (seed
  standard deviation up to 0.049).
* **Without augmentation the decoder overfits sooner.** The checkpoint of lowest validation loss is at epoch 3 to 6
  for the identity, against 7 to 10 with all eight at K = 100 and 20 %. At K = 20 % the four-variant seeds follow
  their checkpoint epoch: 0.376 and 0.396 at epochs 5 and 6, 0.450 at epoch 10. So part of the low-K spread comes
  from the checkpoint criterion, which is the open choice of pipeline.md section 10.
* Against the full-Estonia grid of 2 October, with all eight variants, half the training blocks cost 0.035, 0.041
  and 0.084 Macro-F1 at K = 100, 20 and 5 % (0.551 against 0.586, 0.438 against 0.479, 0.278 against 0.362).

The main workflow keeps all eight variants. Halving the training cache with four would cost about 0.01 at 100 % and
an unresolved amount at lower K.
