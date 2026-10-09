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

**Results.** `results/2026-10-09_terramind_d4_count/EE_2021_half/`.

**Outcome.** Filled in when the results are in.
