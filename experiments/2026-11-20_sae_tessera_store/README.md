# Sparse autoencoders on the precomputed TESSERA embeddings of Estonia 2021, 20 November 2026

**Status: in progress.** The extraction was started on 20 November 2026 (`run.py extract`, log in
`results/2026-11-20_sae_tessera_store/EE_2021_rev/logs/`). `stages.py` (train, evaluate) is written but not yet
run. `report.py` and `REPORT.md` are still to be written.

**Question.** Does a sparse autoencoder (SAE) on the precomputed TESSERA embeddings (the v1.0 store, 128
dimensions, `<chip>_tessera.tif`) of all of Estonia 2021 find features that each mark one crop? Do they do so more
cleanly than the raw dimensions or PCA, and better than an SAE with a frozen random dictionary? And does the
trained crop head rely on those features?

**Method.** This follows geosae (Benson et al., arXiv 2611.11398). `sae.py` is adapted from it under the MIT
licence, and its header lists what differs.

- **SAE:** BatchTopK with M = 1,024 features, k = 16, 3 seeds; k = 8 and 32 for the reconstruction curve; the
  frozen-decoder control.
- **Training data:** about 4.9 M pixels from the training chips, z-scored with the training-chip statistics and
  scaled to unit mean norm.
- **Selection and scoring:** choices (best unit per class, thresholds, feature families) are made on the
  validation chips. Every score is computed on the test chips, on field interiors (fields shrunk by one pixel,
  for the store's 0.7 px offset).
- **Tests:**
  1. Reconstruction at matched L0.
  2. Folded AUROC alignment per feature and class.
  3. The best single feature per class.
  4. Feature families that classify crops on their own, against random families.
  5. The crop head run on the SAE reconstruction. Then each class's family is removed from the embedding,
     against random sets of the same size (Young and Keshav, arXiv 2610.32617).
  6. Monthly NDVI, NDMI, VV and VH of the pixels where each feature is active.
  7. Stability across seeds.

**Crop head.** `results/2026-10-14_unet_baseline/EE_2021_rev/tessera_v1/P100_draw0_seed0`: the frozen store
embedding with a per-pixel MLP, K = 100 %, test pixel Macro-F1 0.623.

**How to run** (hub, CPU; the shared GPU was full):

    poetry run python experiments/2026-11-20_sae_tessera_store/run.py extract
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py head
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py train --run batchtopk_k16_s0   # one process per run
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py evaluate

**Results.** `results/2026-11-20_sae_tessera_store/EE_2021_rev/`.

**Outcome.** Not yet.
