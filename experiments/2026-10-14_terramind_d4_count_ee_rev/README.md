# How many D4 variants does TerraMind's decoder need? Full Estonia, EE_2021_rev, 14 October 2026

**Question.** Does the D4 augmentation add to the results on the main workflow's chip set? The study of
`../2026-10-09_terramind_d4_count` answered it on half of Estonia's training blocks, cut from `EE_2021` (the
`export_2021` screening): eight variants beat the identity by 0.033 test Macro-F1 at K = 100 % and 0.070 at 20 %
in every seed, three seeds could not decide K = 5 %, and most of the gain came through the checkpoint of lowest
validation loss, which stops the unaugmented decoder at epoch 3 to 6. This repeats it on all of Estonia under the
revised Sentinel-2 screening, where the decoder sees twice the training chips and should memorise them more
slowly, so the gain may be smaller.

**What it varies.** Only the variants the decoder trains on, as in the earlier study. Encoder, cache, decoder,
optimiser, batch and epochs are those of the core arm; validation and test always read the identity.

| Arm | Variants drawn in training | Where |
|---|---|---|
| `terramind_v1_large_d4x1` | `k0`: the identity, no augmentation | `arms/` |
| `terramind_v1_large_d4x4` | `k0 k1 k2 k3`: the four quarter turns, no flip | `arms/` |
| `terramind_v1_large` | all eight of D4, the main workflow | core arm, `configs/arms/` |

The arm files and `run.py` are copies of the earlier study's: `run.py` wraps the runner's `compose` to copy the
arm's `data.train_variants` into the cell's configuration, and the cached fit to read it from there. All three
arms read the same cache, which holds all eight variants.

**Chip set.** `EE_2021_rev`, the main workflow's: split `blocks4_buf1600_seed0__6b0eb4cb`, 4,898 training, 769
validation and 1,475 test chips; the test chips are those of the earlier study and the 2 October grid. With eight
variants per training chip and the identity for validation and test, each job encodes 41,428 chips, about
0.8 TB, into its node's `/local`.

**Protocol.** K = 100, 20 and 5 % of the parcels of each class, draw 0, seeds 0, 1 and 2, 15 epochs, checkpoint
of lowest validation loss (`experiment.yaml`): the earlier study's. 27 fits, one cluster job per arm. The arms
share draw and seeds, so the differences are read in pairs. The core arm's nine fits are also the main workflow's
TerraMind cells on `EE_2021_rev`, with three seeds.

**Checks on the hub, 14 October 2026.** `pytest` passes; the dry run lists all 27 cells and the chip-set check passes; with the
overrides applied, the cached fit receives `[k0]`, `[k0, k1, k2, k3]` and `None` (all eight) for the three arms.

**How to run.**

    poetry run python experiments/2026-10-14_terramind_d4_count_ee_rev/run.py -e experiments/2026-10-14_terramind_d4_count_ee_rev/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev   # hub
    TIME=0-12:00:00 bash scripts/cluster/submit.sh experiments/2026-10-14_terramind_d4_count_ee_rev/experiment.yaml EE_2021_rev                                                   # cluster

**Results.** `results/2026-10-14_terramind_d4_count_ee_rev/EE_2021_rev/`. `report.py --chips EE_2021_rev` (the
earlier study's, same `ARMS` and `PAIRS`) writes the figures and tables of `REPORT.md` into `figures/`.

**Run.** UT HPC, 14 October 2026, at commit `53f7783`: jobs 621979 (identity, node30, 3 h 36 min), 621980 (four
rotations, node29, 3 h 23 min) and 621981 (all eight, node29, 3 h 23 min), each encoding its own 0.8 TB cache in
26 min. No failure.

**Outcome.** See [`REPORT.md`](REPORT.md). The augmentation adds at every budget on full Estonia: eight variants
beat the identity by 0.047, 0.061 and 0.070 test Macro-F1 at K = 100, 20 and 5 %, in every seed. As on half of
Estonia, most of it comes through the checkpoint of lowest validation loss, which tests the unaugmented decoder at
epoch 2 to 7; at each fit's best validation epoch the gap is a quarter to a third as large. At 100 %, half of the
gain is potatoes, learnt from epoch 5 to 8 on and so missed by most identity checkpoints. The main workflow keeps
all eight variants.
