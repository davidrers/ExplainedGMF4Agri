# THOR token size, 160, 80 and 40 m, on a sample of Estonia, 9 October 2026

**Question.** How does the size of THOR's tokens change crop-type segmentation under label scarcity? THOR v1
large is run frozen on 160, 80 and 40 m tokens with the same decoder, budgets and seeds. 160 m is the main
workflow's arm, which matches TerraMind's token grid. 80 m beat 160 m on the 12-chip pilot and on `EE_2021_mini`,
but on 9 and 57 chips. 40 m goes below what THOR was pretrained with: 4 px patches at the least, and 1,296
tokens per image.

**What it varies.** Only the token size. Every arm has the same encoder layers (5, 11, 17, 23), the same
`ChannelBottleneck` to 768 channels, the same UNet decoder, a batch of 2 accumulated twice, fp16 mixed precision,
and the same draws and seeds.

| Arm | Patch, 10 m bands | Patch, 20 m bands | Grid | Tokens per month | Cache per chip encoding |
|---|---|---|---|---|---|
| `thor_v1_large` (core arm, `configs/arms/`) | 16 px | 8 px | 14 x 14 | 392 | about 38 MB |
| `thor_v1_large_80m` (`arms/`) | 8 px | 4 px | 28 x 28 | 1,568 | about 154 MB |
| `thor_v1_large_40m` (`arms_run/`, backbone registered by `run.py`) | 4 px | 2 px | 56 x 56 | 6,272 | about 616 MB |

`thor_v1_large_80m` here differs from the arm of `../2026-10-08_thor_80m` only in its batch: 2 accumulated
twice, as in the core arm, instead of 1 accumulated four times. The 40 m arm sits in `arms_run/`, which `run.py`
puts first on the arm search path, because the core tests check every arm in `arms/` against the core registry,
which does not hold the 40 m backbone. At 2 px, THOR resizes each 20 m band's 16 x 16 kernel
further than pretraining ever did, so the 40 m arm measures how far the frozen weights extrapolate.

**Chip set.** `EE_2021_sample`, cut from `EE_2021` with `scripts/hub/build_pilot_chipset.py --train-blocks 36
--val-blocks 8 --test-blocks 18` (seed 0): whole 4 x 4 chip blocks of the full split. That is 413 training, 88
validation and 207 test chips, about 8 % of Estonia's training blocks and 14 % of its test blocks. All 20 classes
are present in every partition; the rarest has 43 training parcels. The sample has 3,599 chip encodings (8 D4
variants per training chip), so the 40 m cache is about 2.2 TB on a cluster node's local NVMe.

**Protocol.** K = 100, 20 and 5 % of the parcels of each class, draw 0, seeds 0, 1 and 2, 15 epochs, checkpoint
of lowest validation loss (`experiment.yaml`).

**Checks on the hub, 9 October 2026.** One validation chip, one month, under fp16 autocast. All three sizes build,
give finite features (largest |x| at layer 23 about 163, 196 and 200) and logits at 224 x 224. At 40 m the encoder
takes 0.51 s and 5.5 GiB per month image on the RTX A4000. A chip's 12 months go through the encoder as one batch,
so the 40 m arm encodes one chip at a time on the cluster.

**How to run.**

    poetry run python experiments/2026-10-09_thor_token_size/run.py -e experiments/2026-10-09_thor_token_size/experiment.yaml --chips data/eurocrops_chips/EE_2021_sample   # hub
    bash scripts/cluster/submit.sh experiments/2026-10-09_thor_token_size/experiment.yaml EE_2021_sample                                                                    # cluster

**Runs on the cluster.** Commit `11fe4da`: 160 m (job 617993, 32 min) and 80 m (job 617994, 1 h 40 min)
finished. 40 m (job 617995) encoded all 3,599 chip encodings in 2 h 25 min, then failed two minutes into its
first fit: a DataLoader worker could not allocate shared memory. A 40 m sample is 1.23 GB in float32, so 16
workers each prefetching two batches of 2 held about 79 GB, beyond the job's 120 GB. The job's exit trap
removed the cache from the node's local disk. `run.py` now caps the cluster's fit workers at 4, which hold what
80 m's sixteen did, and 40 m was submitted again. The worker count changes throughput and which worker draws a
sample's D4 variant, nothing else.

**Results.** `results/2026-10-09_thor_token_size/EE_2021_sample/`.

**Outcome.** Filled in when the results are in.
