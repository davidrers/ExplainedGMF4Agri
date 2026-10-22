# Phase 1 pipeline: crop types from frozen foundation models {#phase1-pipeline}

Six arms map crop types on all of Estonia 2021 from the same chips, labels and test blocks. **Five frozen foundation models feed trained decoders, against a U-Net trained from scratch.**

- Every arm reads one chip set, draws the same parcels at each label budget K and is scored on the same 1,475 test chips.[](source:pipeline)
- Each frozen encoder runs once per chip set; only the decoders train, 45 epochs at every K.
- Six experiments back the main design choices, each varying one factor on one encoder.
- First results: frozen TerraMind leads per pixel at K = 100 %, frozen TESSERA v2 at 20 and 5 %.[](source:estonia)

@answer scope="Estonia 2021 · main workflow kshot · 45-epoch grid of 16 October 2026 · one draw and one seed per cell"
@meta locale=en style=swiss
@source pipeline title="docs/phase1/pipeline.md" url=https://github.com/davidrers/ExplainedGMF4Agri/blob/main/docs/phase1/pipeline.md
@source estonia title="First results for Estonia 2021" url=https://github.com/davidrers/ExplainedGMF4Agri/blob/main/docs/phase1/results_estonia/REPORT.md

## 1. Experimental workflow {#workflow}

@diagram workflow id=experimental-workflow source=experimental.workflow.json title="One run of the main workflow"

Grey is frozen or saved to disk, orange is trained for every K, draw and seed, white is data and evaluation.

@steps id=workflow-steps
| s-parcels | EuroCrops parcels | Estonia 2021 declarations from EuroCrops v11, mapped to 20 crop classes. | experimental-workflow | parcels |
| s-imagery | Satellite inputs | Twelve monthly Sentinel-2 L2A and Sentinel-1 RTC composites of 2021, and the AlphaEarth annual embedding. | experimental-workflow | imagery |
| s-chips | Chip set | A fixed 2,240 m grid in EPSG:3035: 7,402 chips, each with a crop mask and a parcel-id raster. | experimental-workflow | chips |
| s-split | Spatial split | Whole blocks of 4 × 4 chips with a 1,600 m buffer: 4,898 training, 769 validation and 1,475 test chips. | experimental-workflow | split |
| s-budget | Label budget | K = 100, 20 and 5 % of the training parcels of each class, applied when the mask loads. | experimental-workflow | budget |
| s-encode | Encode once | Each frozen encoder runs once per chip set, training chips in all 8 D4 copies. The U-Net skips this step. | experimental-workflow | encode |
| s-train | Train decoder | 45 epochs at every K, then the checkpoint of lowest validation loss is tested. | experimental-workflow | train |
| s-test | Score test blocks | Macro-F1 per pixel, the primary metric, and per parcel by majority vote of its pixels. | experimental-workflow | test |
| s-report | results.json | One file per fit with its settings, the drawn parcels and every metric. | experimental-workflow | report |

| Fairness rule | Status in the main workflow |
| --- | --- |
| Same input: identical chips and masks for every arm | One chip set, `EE_2021_rev` {good:Met} |
| Same labels: identical parcels at each K and draw | The draw depends on class, K and draw only {good:Met} |
| Same test: one fixed, spatially separated test set | The same 1,475 test chips for every fit {good:Met} |
| Same effort: equal training for every arm and K | 45 epochs for all; no tuning trials yet {caution:Partly} |
| No leakage: buffer, and no validation labels at low K | 1,600 m buffer; the checkpoint reads validation labels at every K {caution:Partly} |
| Like with like: headline comparisons within a resolution group | Groups declared; the U-Net belongs to neither {caution:Open} |
| Repeated and paired: several draws, the same draws for every arm | Every arm on the same draw, but one draw and one seed {bad:Not yet} |

## 2. Pipeline architecture {#architecture}

@diagram architecture id=pipeline-architecture source=pipeline.architecture.json title="The six arms, from chips to crop map"

Grey is frozen or saved to disk, orange is trained for every K, draw and seed, white is data and evaluation. The token-grid backbones work as follows.

@image id=img-backbones path=backbones.png alt="Left: TemporalWrapper, twelve monthly ViT passes concatenated into 12 × D channels. Right: Prithvi, twelve months entering one ViT encoder." caption="Left: TerraMind and THOR, one frozen ViT pass per month. Right: Prithvi, all 12 months in one frozen ViT."

| Arm | Input | Encoder | Decoder | Trained parameters | Learning rate |
| --- | --- | --- | --- | --- | --- |
| TerraMind v1 large | S2 12 bands + S1 VV, VH | frozen, one pass per month; S2 and S1 tokens averaged | bottleneck + UNet | 53.3 M | 1e-4 |
| THOR v1 large | S2 10 bands + S1 VV, VH | frozen, one pass per month; 3 band groups averaged | bottleneck + UNet | 53.3 M | 1e-4 |
| Prithvi-EO-2.0 600M TL | S2 6 HLS bands + date, location | frozen, 12 months in one ViT | bottleneck + UNet | 62.7 M | 1e-4 |
| TESSERA v2 Medium | S2 10 bands + S1 VV, VH | frozen, per pixel, run on our composites | per-pixel MLP | 0.20 M | 1e-3 |
| AlphaEarth v1 | 64-d annual embedding | precomputed by Google | per-pixel MLP | 0.17 M | 1e-3 |
| U-Net, the baseline | S2 + S1 on 168 channels | none: ResNet-18 from scratch | U-Net | 14.8 M | 1e-3 |

@facts id=shared-settings
| Epochs | 45 at every K |
| Optimiser | AdamW, weight decay 0.05 |
| Effective batch | 4 chips |
| Loss | cross-entropy on labelled pixels |
| Augmentation | D4: 8 flips and rotations |
| Checkpoint | lowest validation loss |

## 3. Experiments that justify the pipeline {#experiments}

Each experiment varies one factor on one encoder and compares the variants on shared draws and seeds. The relative result backs the decision, so none is rerun on the current arms.

| Experiment | Result | Decision it backs | Report |
| --- | --- | --- | --- |
| 1. THOR tokens: 160, 80 or 40 m | 40 m beats 160 m by about 0.05 Macro-F1 at every K, for 16 times the cache | 160 m kept, for parity with TerraMind and a cache that fits {caution:Trade-off} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-09_thor_token_size/REPORT.md) |
| 2. Rotations: 1, 4 or 8 D4 copies | 8 copies beat 1, no augmentation, by 0.05 to 0.07; most of it through the checkpoint | All 8 copies cached {good:Kept} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-14_terramind_d4_count_ee_rev/REPORT.md) |
| 3. 12 monthly or 24 half-month frames | +0.015, +0.035 and +0.001 for 24 frames, within the seed spread, at twice the cost | 12 monthly composites {good:Kept} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-12_terramind_12v24/REPORT.md) |
| 4. TESSERA: encoder on composites or store embeddings | Equal in the probe (0.371 against 0.373); 0.068 lower at 5 % in the main workflow | TESSERA v2 frozen on our composites, without the store's offset {caution:Trade-off} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-14_tessera_composites/REPORT.md) |
| 5. TerraMind on S2, S1 or both | S2 + S1 beats S2 by 0.025 to 0.054 and wins every fit; S1 alone is below S2 | Every model reads every sensor it was pretrained on {good:Kept} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-13_terramind_s1_s2/REPORT.md) |
| 6. Fine-tuned or frozen encoder | TESSERA gains 0.04 to 0.08; the token-grid models about 0.01 in validation | Encoders stay frozen; fine-tuning is reported as an insight {good:Kept} | [REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/experiments/2026-10-15_kshot_finetune/REPORT.md) |

The main workflow's own results on all of Estonia, frozen and fine-tuned, per pixel and per parcel, are in [results_estonia/REPORT.md](https://github.com/davidrers/ExplainedGMF4Agri/blob/main/docs/phase1/results_estonia/REPORT.md).

@callout level=note title="Older arms in experiments 1 to 3"
| Experiments 1 to 3 ran on TerraMind or THOR with Sentinel-2 alone, at 15 epochs. |
| The frozen token-grid arms gave the same cells at 15 and 45 epochs in the main workflow. |
