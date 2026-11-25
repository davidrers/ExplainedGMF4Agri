### The fields explained

| species | fields | right | median_pixels |
|---|---|---|---|
| barley | 85 | 0.86 | 458.00 |
| buckwheat | 40 | 1.00 | 433.00 |
| dry pulses | 126 | 0.72 | 445.50 |
| forage legumes | 124 | 0.53 | 290.50 |
| grassland | 57 | 1.00 | 276.00 |
| maize | 40 | 1.00 | 554.50 |
| oats | 40 | 0.72 | 325.00 |
| orchards | 16 | 0.00 | 86.50 |
| potatoes | 40 | 0.95 | 215.50 |
| rapeseed | 85 | 0.78 | 392.00 |
| rye | 40 | 0.68 | 383.50 |
| vegetables | 7 | 0.00 | 83.00 |
| wheat | 80 | 0.76 | 387.50 |

### Chain against the stored predictions

| precision | pixel agreement | parcel agreement | s per chip |
|---|---|---|---|
| bf16-mixed | 0.99905 | 0.99974 | 2.46777 |
| 32 | 0.99653 | 0.99908 | 2.60851 |

### Completeness and convergence

| check | value |
|---|---|
| SHAP months: largest gap between the sum and f(x) - f(mean) | 0.0000 |
| IG, 32 steps: relative completeness error, median | 0.0027 |
| IG, 32 steps: relative completeness error, 90th percentile | 0.0130 |
| IG, 64 steps (102 fields): relative error, median | 0.0013 |
| SHAP months, two halves of the permutations: Spearman, median | 0.9231 |
| IG months, 32 against 64 steps: Spearman, median | 1.0000 |

### Deletion curves (A's log-probability)

| order | removed with | mean fall over the steps (AOPC) | after 1 month | after all steps | against random |
|---|---|---|---|---|---|
| occlusion_mean | drop | 1.012 | 0.343 | 1.544 | 0.772 |
| occlusion_mean | mean | 1.879 | 0.808 | 2.386 | 1.405 |
| occlusion_drop | drop | 1.252 | 0.490 | 1.736 | 1.012 |
| occlusion_drop | mean | 1.419 | 0.571 | 1.918 | 0.945 |
| shap | drop | 1.036 | 0.303 | 1.561 | 0.796 |
| shap | mean | 2.447 | 0.698 | 3.492 | 1.974 |
| ig | drop | 0.914 | 0.285 | 1.440 | 0.673 |
| ig | mean | 2.104 | 0.656 | 2.962 | 1.630 |
| random0 | drop | 0.235 | 0.038 | 0.516 | -0.006 |
| random0 | mean | 0.478 | 0.065 | 0.971 | 0.004 |
| random1 | drop | 0.246 | 0.046 | 0.491 | 0.006 |
| random1 | mean | 0.470 | 0.067 | 0.943 | -0.004 |

### Agreement on the months (Spearman per field, median over the fields the model got right)

|  | Occlusion (mean) | Occlusion (drop) | SHAP | Integrated Gradients |
|---|---|---|---|---|
| Occlusion (mean) | 1.00 | 0.46 | 0.56 | 0.57 |
| Occlusion (drop) | 0.46 | 1.00 | 0.30 | 0.32 |
| SHAP | 0.56 | 0.30 | 1.00 | 0.77 |
| Integrated Gradients | 0.57 | 0.32 | 0.77 | 1.00 |

### Agreement on the band groups (Spearman per field, median over the fields the model got right)

|  | Occlusion (mean) | SHAP | Integrated Gradients |
|---|---|---|---|
| Occlusion (mean) | 1.00 | 0.60 | 0.40 |
| SHAP | 0.60 | 1.00 | 0.54 |
| Integrated Gradients | 0.40 | 0.54 | 1.00 |

### Agreement on the 72 cells, occlusion (mean) against IG

median Spearman 0.28

### Randomised network: Spearman with the trained network's explanation, median (near 0 = the method depends on the weights)

| randomised | windows, mean | windows, drop | IG months |
|---|---|---|---|
| head | -0.07 | 0.02 | 0.05 |
| head+decoder | -0.12 | -0.02 | 0.01 |

### Q1 When: the costliest window and the share of attribution in the expected window (uniform = 0.25)

| species | fields | reference window | occlusion (mean) | occlusion (drop) | SHAP share in window | IG share in window | NDVI peak | reference |
|---|---|---|---|---|---|---|---|---|
| grassland | 57 | Apr-Jun | May-Jul (neighbour) | May-Jul (neighbour) | 0.41 | 0.34 | Jun | Estonia |
| forage legumes | 66 | May-Jul | May-Jul (same) | May-Jul (same) | 0.67 | 0.59 | Jun | Estonia |
| wheat | 61 | May-Jul | Jun-Aug (neighbour) | Jun-Aug (neighbour) | 0.54 | 0.63 | Jun | Estonia |
| barley | 73 | May-Jul | May-Jul (same) | Jun-Aug (neighbour) | 0.62 | 0.55 | Jun | Estonia |
| rye | 27 | May-Jul | Jul-Sep (+2 months) | May-Jul (same) | 0.66 | 0.56 | Jun | Estonia |
| oats | 29 | May-Jul | Apr-Jun (neighbour) | Jun-Aug (neighbour) | 0.59 | 0.48 | Jun | Estonia |
| rapeseed | 66 | May-Jul | May-Jul (same) | May-Jul (same) | 0.71 | 0.78 | Jun | Estonia |
| dry pulses | 91 | May-Jul | Jun-Aug (neighbour) | Jun-Aug (neighbour) | 0.68 | 0.67 | Jun | Estonia |
| buckwheat | 40 | Jun-Aug | Jun-Aug (same) | Jul-Sep (neighbour) | 0.56 | 0.55 | Jul | Estonia |
| potatoes | 38 | Jul-Sep | Jun-Aug (neighbour) | Jul-Sep (same) | 0.55 | 0.47 | Jul | zone 21 |
| maize | 40 | Jun-Aug | Jul-Sep (neighbour) | Jul-Sep (neighbour) | 0.54 | 0.57 | Sep | Estonia |

### Q2 Which property: the leading band group by each method, and the sensors

| species | SHAP top | occlusion (mean) top | IG top | reference: costliest group | reference: best cue | S2 removed (drop) | S1 removed (drop) | reference S2 / S1 recall lost | SHAP share of radar |
|---|---|---|---|---|---|---|---|---|---|
| grassland | NIR | NIR | NIR | NIR | Oct VV (0.89) | 0.32 | 0.52 | 0.61 / 0.22 | 0.17 |
| forage legumes | SWIR | SWIR | Red-edge | SWIR | May VV (0.74) | 1.53 | -0.03 | 0.48 / 0.05 | 0.09 |
| wheat | Red-edge | Red-edge | NIR | Red-edge | Jun Red-edge (0.83) | 2.25 | 0.09 | 0.57 / 0.15 | 0.13 |
| barley | Red-edge | Red-edge | NIR | Visible | Aug SWIR (0.74) | 2.70 | 0.28 | 0.50 / 0.23 | 0.16 |
| rye | Red-edge | Red-edge | Red-edge | SWIR | Jun SWIR (0.82) | 3.16 | 1.74 | 0.63 / 0.30 | 0.06 |
| oats | NIR | NIR | Visible | VV | Jul VV (0.81) | 2.15 | 0.10 | 0.49 / 0.30 | 0.24 |
| rapeseed | VH | Red-edge | Red-edge | VH | Jun VV (0.95) | 1.74 | 1.07 | 0.24 / 0.15 | 0.35 |
| dry pulses | Visible | NIR | Red-edge | VV | Jun VH (0.82) | 1.70 | 0.27 | 0.47 / 0.54 | 0.10 |
| buckwheat | Red-edge | Red-edge | Red-edge | VH | Sep VH (0.90) | 4.92 | 0.29 | 0.77 / 0.38 | 0.03 |
| potatoes | SWIR | SWIR | Red-edge | VV | Jul VV (0.87) | 3.75 | 1.36 | 0.66 / 0.41 | 0.13 |
| maize | Red-edge | Red-edge | Red-edge | Red-edge | Sep Red-edge (0.92) | 2.36 | 1.00 | 0.85 / 0.49 | 0.04 |

### The 72 cells against the reference AUC map (Spearman), and the null cells (Q7)

| species | IG cells (absolute) vs AUC | occlusion cells vs AUC | null cells (AUC < 0.55) | IG share on null cells | null cells' share of the map |
|---|---|---|---|---|---|
| grassland | 0.04 | -0.15 | 1 | 0.00 | 0.01 |
| forage legumes | 0.09 | -0.07 | 8 | 0.10 | 0.11 |
| wheat | 0.45 | 0.18 | 8 | 0.06 | 0.11 |
| barley | 0.18 | 0.13 | 14 | 0.13 | 0.19 |
| rye | 0.58 | 0.16 | 12 | 0.02 | 0.17 |
| oats | 0.25 | 0.46 | 24 | 0.21 | 0.33 |
| rapeseed | 0.26 | 0.22 | 1 | 0.00 | 0.01 |
| dry pulses | 0.35 | 0.26 | 11 | 0.05 | 0.15 |
| buckwheat | 0.36 | 0.18 | 18 | 0.06 | 0.25 |
| maize | 0.51 | 0.22 | 2 | 0.00 | 0.03 |

### Q3 Against which crop: what separates a field's species from its rival (IG of A - B)

| A | rival B | fields | model wrong | IG (A - B) top cell | share of the top cell | reference pair cue | rank of the reference cue cell (of 72) |
|---|---|---|---|---|---|---|---|
| forage legumes | grassland | 99 | 46 | May Red-edge | 0.12 | Feb Red-edge (0.74, no single cue) | 12 |
| grassland | forage legumes | 57 | 0 | May NIR | 0.10 | Feb Red-edge (0.74, no single cue) | 63 |
| barley | wheat | 53 | 6 | Jun NIR | 0.13 | May Visible (0.75, partial) | 52 |
| dry pulses | forage legumes | 40 | 24 | Aug Red-edge | 0.13 | Jun VV (0.92, clear) | 20 |
| dry pulses | rapeseed | 38 | 0 | May NIR | 0.11 | May SWIR (0.91, clear) | 27 |
| rye | wheat | 37 | 10 | May Red-edge | 0.12 | Jul VH (0.88, partial) | 18 |
| buckwheat | rapeseed | 35 | 0 | Jul Red-edge | 0.16 | Jun VH (0.96, clear) | 22 |
| rapeseed | wheat | 35 | 0 | Jun Red-edge | 0.17 | Jun VV (0.99, clear) | 16 |
| wheat | barley | 31 | 10 | Jun Red-edge | 0.12 | May Visible (0.75, partial) | 11 |
| potatoes | dry pulses | 28 | 1 | Jun Red-edge | 0.12 |  |  |
| barley | oats | 27 | 3 | May Red-edge | 0.11 | Jul Red-edge (0.78, partial) | 13 |
| wheat | oats | 23 | 8 | Jun NIR | 0.14 | Jun Red-edge (0.84, partial) | 63 |
| rapeseed | dry pulses | 19 | 2 | May Red-edge | 0.11 | May SWIR (0.91, clear) | 28 |
| oats | wheat | 17 | 1 | May Visible | 0.15 | Jun Red-edge (0.84, partial) | 70 |
| maize | wheat | 17 | 0 | Aug Red-edge | 0.14 | Jun SWIR (0.98, clear) | 68 |
| orchards | grassland | 15 | 15 | May Red-edge | 0.12 | Aug VV (0.85, partial) | 49 |
| wheat | rye | 15 | 0 | Jun Visible | 0.10 | Jul VH (0.88, partial) | 7 |
| forage legumes | dry pulses | 14 | 5 | Jun Red-edge | 0.11 | Jun VV (0.92, clear) | 50 |
| rapeseed | forage legumes | 14 | 5 | Jun Visible | 0.13 | Jun VV (0.98, clear) | 20 |
| oats | barley | 13 | 5 | Jun Visible | 0.09 | Jul Red-edge (0.78, partial) | 70 |
| dry pulses | barley | 13 | 1 | Jun Red-edge | 0.22 | Jun VV (0.95, clear) | 7 |
| dry pulses | wheat | 13 | 0 | Jun Red-edge | 0.13 | Jun VV (0.97, clear) | 22 |
| rapeseed | grassland | 12 | 8 | Jun NIR | 0.13 | Jun VV (0.98, clear) | 44 |
| dry pulses | grassland | 10 | 10 | Jun Red-edge | 0.10 | May NIR (0.95, clear) | 69 |
| dry pulses | oats | 10 | 0 | May Red-edge | 0.12 | Jun VH (0.94, clear) | 16 |
| maize | dry pulses | 10 | 0 | Jun Red-edge | 0.12 | Jun VH (0.96, clear) | 42 |
| wheat | grassland | 8 | 0 | Jun Red-edge | 0.17 | Sep NIR (0.94, clear) | 12 |

### Q4 Direction: does raising the cue band move the species' score the way the data says?

| species | cue | the species is | fields | share whose gradient agrees |
|---|---|---|---|---|
| grassland | Oct VV | lower | 57 | 0.96 |
| forage legumes | May VV | lower | 66 | 0.39 |
| wheat | Jun B05 | lower | 61 | 1.00 |
| barley | Aug B12 | higher | 73 | 0.96 |
| rye | Jun B11 | lower | 27 | 0.93 |
| oats | Jul VV | lower | 29 | 1.00 |
| rapeseed | Jun VV | higher | 66 | 1.00 |
| dry pulses | Jun VH | higher | 91 | 0.99 |
| buckwheat | Sep VH | higher | 40 | 0.35 |
| potatoes | Jul VV | higher | 38 | 1.00 |
| maize | Sep B05 | lower | 40 | 1.00 |

### Q6 Sowing form: winter against spring fields of one species

| species | form | fields | top months (SHAP) | share Jan to May |
|---|---|---|---|---|
| wheat | winter | 40 | Jun, Aug, Jul | 0.05 |
| wheat | spring | 21 | Jul, Jun, Sep | 0.02 |
| barley | winter | 35 | Jun, Jul, Aug | 0.21 |
| barley | spring | 38 | May, Jun, Jul | 0.52 |
| rapeseed | winter | 45 | Jun, Jul, May | 0.20 |
| rapeseed | spring | 21 | Jul, Aug, Jun | 0.00 |

### Q7 Cloud-filled months (Feb, Mar, Nov)

share of |SHAP| on them, median over fields: 0.094 (uniform 0.25)

### Context: the same windows removed from the field only or from the whole chip

| replacement | largest window fall, own field only | same, whole chip | Spearman field vs chip, median |
|---|---|---|---|
| mean | 2.65 | 2.75 | 0.99 |
| drop | 1.62 | 1.69 | 0.99 |

### Minutes per chip, median

| part | minutes |
|---|---|
| setup | 0.3 |
| occlusion | 5.2 |
| shap | 7.3 |
| ig | 2.3 |
| deletion | 1.1 |
| ig_check | 12.1 |
