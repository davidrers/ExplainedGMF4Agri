Main drivers against the AUC map. 544 correctly predicted fields of 10 species; K = {'cell': 5, 'month': 3, 'band group': 2}.

### Inference level: every correctly predicted field, pooled over species (hit and overlap: shares; chance = K / N)

| level | method | chance (hit, overlap) | fields | hit | hit, other species' maps | overlap | overlap, other species' maps | AUC gap, median | AUC gap, other species' maps | gap ≤ 0.05 | Spearman, median | Spearman, other species' maps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cell | IG | 0.07 | 544.00 | 0.20 | 0.14 | 0.14 | 0.11 | 0.10 | 0.14 | 0.17 | 0.06 | 0.03 |
| cell | Occlusion (mean) | 0.07 | 544.00 | 0.20 | 0.17 | 0.16 | 0.13 | 0.12 | 0.13 | 0.17 | 0.13 | 0.06 |
| month | SHAP | 0.25 | 544.00 | 0.50 | 0.54 | 0.49 | 0.46 | 0.05 | 0.06 | 0.46 | 0.37 | 0.22 |
| month | IG | 0.25 | 544.00 | 0.54 | 0.53 | 0.48 | 0.44 | 0.02 | 0.06 | 0.60 | 0.33 | 0.18 |
| month | Occlusion (mean) | 0.25 | 544.00 | 0.50 | 0.52 | 0.43 | 0.44 | 0.03 | 0.07 | 0.55 | 0.17 | 0.20 |
| month | Occlusion (drop) | 0.25 | 544.00 | 0.56 | 0.51 | 0.45 | 0.41 | 0.02 | 0.07 | 0.61 | 0.17 | 0.15 |
| band group | SHAP | 0.33 | 544.00 | 0.37 | 0.30 | 0.35 | 0.29 | 0.04 | 0.05 | 0.56 | 0.14 | -0.03 |
| band group | IG | 0.33 | 544.00 | 0.34 | 0.31 | 0.29 | 0.27 | 0.05 | 0.05 | 0.44 | -0.03 | 0.05 |
| band group | Occlusion (mean) | 0.33 | 544.00 | 0.36 | 0.32 | 0.30 | 0.28 | 0.04 | 0.05 | 0.51 | 0.03 | -0.09 |

### Model level: each species' mean explanation over its correct fields, against the same maps

| level | method | species | hit | hit, other species' maps | overlap | overlap, other species' maps | AUC gap, median | Spearman, median | Spearman, other species' maps |
|---|---|---|---|---|---|---|---|---|---|
| cell | IG | 10 | 0.60 | 0.14 | 0.22 | 0.12 | 0.06 | 0.11 | 0.02 |
| cell | Occlusion (mean) | 10 | 0.20 | 0.21 | 0.12 | 0.14 | 0.14 | 0.18 | 0.13 |
| month | SHAP | 10 | 0.60 | 0.52 | 0.50 | 0.49 | 0.05 | 0.55 | 0.28 |
| month | IG | 10 | 0.50 | 0.53 | 0.53 | 0.50 | 0.03 | 0.35 | 0.18 |
| month | Occlusion (mean) | 10 | 0.50 | 0.51 | 0.53 | 0.50 | 0.05 | 0.47 | 0.31 |
| month | Occlusion (drop) | 10 | 0.60 | 0.52 | 0.57 | 0.49 | 0.01 | 0.45 | 0.28 |
| band group | SHAP | 10 | 0.60 | 0.30 | 0.35 | 0.23 | 0.03 | 0.17 | -0.04 |
| band group | IG | 10 | 0.60 | 0.34 | 0.35 | 0.27 | 0.04 | 0.14 | 0.12 |
| band group | Occlusion (mean) | 10 | 0.50 | 0.34 | 0.35 | 0.31 | 0.04 | -0.06 | -0.09 |

### Per species, inference level

| level | method | species | fields | hit | hit, other species' maps | overlap | AUC gap, median | AUC gap, other species' maps | Spearman, median |
|---|---|---|---|---|---|---|---|---|---|
| cell | IG | rapeseed | 64.00 | 0.25 | 0.15 | 0.20 | 0.14 | 0.13 | 0.22 |
| cell | Occlusion (mean) | rapeseed | 64.00 | 0.25 | 0.20 | 0.35 | 0.11 | 0.13 | 0.17 |
| month | SHAP | rapeseed | 64.00 | 0.98 | 0.54 | 0.80 | 0.00 | 0.05 | 0.74 |
| month | IG | rapeseed | 64.00 | 1.00 | 0.58 | 0.64 | 0.00 | 0.05 | 0.40 |
| month | Occlusion (mean) | rapeseed | 64.00 | 0.95 | 0.56 | 0.60 | 0.00 | 0.05 | 0.22 |
| month | Occlusion (drop) | rapeseed | 64.00 | 0.95 | 0.54 | 0.60 | 0.00 | 0.05 | 0.10 |
| band group | SHAP | rapeseed | 64.00 | 0.38 | 0.35 | 0.45 | 0.06 | 0.05 | 0.26 |
| band group | IG | rapeseed | 64.00 | 0.00 | 0.36 | 0.15 | 0.10 | 0.05 | 0.03 |
| band group | Occlusion (mean) | rapeseed | 64.00 | 0.00 | 0.44 | 0.02 | 0.06 | 0.04 | 0.03 |
| cell | IG | forage legumes | 66.00 | 0.00 | 0.19 | 0.00 | 0.10 | 0.13 | 0.07 |
| cell | Occlusion (mean) | forage legumes | 66.00 | 0.00 | 0.16 | 0.00 | 0.10 | 0.13 | 0.02 |
| month | SHAP | forage legumes | 66.00 | 0.09 | 0.61 | 0.29 | 0.04 | 0.08 | -0.23 |
| month | IG | forage legumes | 66.00 | 0.30 | 0.51 | 0.36 | 0.04 | 0.08 | 0.08 |
| month | Occlusion (mean) | forage legumes | 66.00 | 0.21 | 0.43 | 0.23 | 0.04 | 0.08 | -0.08 |
| month | Occlusion (drop) | forage legumes | 66.00 | 0.38 | 0.43 | 0.33 | 0.04 | 0.09 | 0.07 |
| band group | SHAP | forage legumes | 66.00 | 0.23 | 0.32 | 0.35 | 0.06 | 0.06 | 0.03 |
| band group | IG | forage legumes | 66.00 | 0.55 | 0.36 | 0.36 | 0.03 | 0.04 | 0.03 |
| band group | Occlusion (mean) | forage legumes | 66.00 | 0.44 | 0.34 | 0.34 | 0.03 | 0.06 | 0.09 |
| cell | IG | grassland | 57.00 | 0.00 | 0.09 | 0.13 | 0.10 | 0.14 | -0.04 |
| cell | Occlusion (mean) | grassland | 57.00 | 0.02 | 0.10 | 0.02 | 0.19 | 0.16 | 0.12 |
| month | SHAP | grassland | 57.00 | 0.05 | 0.59 | 0.22 | 0.09 | 0.07 | -0.17 |
| month | IG | grassland | 57.00 | 0.16 | 0.48 | 0.22 | 0.09 | 0.07 | -0.03 |
| month | Occlusion (mean) | grassland | 57.00 | 0.02 | 0.64 | 0.03 | 0.07 | 0.08 | -0.49 |
| month | Occlusion (drop) | grassland | 57.00 | 0.05 | 0.58 | 0.14 | 0.09 | 0.07 | -0.31 |
| band group | SHAP | grassland | 57.00 | 0.51 | 0.18 | 0.43 | 0.05 | 0.07 | 0.37 |
| band group | IG | grassland | 57.00 | 0.79 | 0.06 | 0.46 | 0.05 | 0.07 | 0.14 |
| band group | Occlusion (mean) | grassland | 57.00 | 0.58 | 0.15 | 0.43 | 0.05 | 0.07 | -0.03 |
| cell | IG | wheat | 60.00 | 0.60 | 0.16 | 0.18 | 0.00 | 0.10 | 0.08 |
| cell | Occlusion (mean) | wheat | 60.00 | 0.58 | 0.17 | 0.19 | 0.00 | 0.10 | 0.14 |
| month | SHAP | wheat | 60.00 | 0.92 | 0.55 | 0.61 | 0.00 | 0.05 | 0.52 |
| month | IG | wheat | 60.00 | 0.98 | 0.63 | 0.67 | 0.00 | 0.05 | 0.47 |
| month | Occlusion (mean) | wheat | 60.00 | 0.85 | 0.54 | 0.51 | 0.00 | 0.05 | 0.31 |
| month | Occlusion (drop) | wheat | 60.00 | 0.95 | 0.60 | 0.63 | 0.00 | 0.05 | 0.37 |
| band group | SHAP | wheat | 60.00 | 0.78 | 0.39 | 0.67 | 0.00 | 0.04 | 0.54 |
| band group | IG | wheat | 60.00 | 0.50 | 0.28 | 0.45 | 0.04 | 0.06 | 0.03 |
| band group | Occlusion (mean) | wheat | 60.00 | 0.73 | 0.37 | 0.48 | 0.00 | 0.04 | 0.03 |
| cell | IG | barley | 72.00 | 0.00 | 0.09 | 0.03 | 0.07 | 0.15 | -0.00 |
| cell | Occlusion (mean) | barley | 72.00 | 0.00 | 0.21 | 0.03 | 0.15 | 0.12 | 0.09 |
| month | SHAP | barley | 72.00 | 0.40 | 0.57 | 0.40 | 0.02 | 0.08 | 0.42 |
| month | IG | barley | 72.00 | 0.31 | 0.52 | 0.41 | 0.02 | 0.08 | 0.49 |
| month | Occlusion (mean) | barley | 72.00 | 0.24 | 0.49 | 0.36 | 0.02 | 0.08 | 0.35 |
| month | Occlusion (drop) | barley | 72.00 | 0.29 | 0.48 | 0.34 | 0.02 | 0.08 | 0.17 |
| band group | SHAP | barley | 72.00 | 0.00 | 0.34 | 0.22 | 0.03 | 0.04 | 0.03 |
| band group | IG | barley | 72.00 | 0.03 | 0.32 | 0.14 | 0.03 | 0.06 | -0.03 |
| band group | Occlusion (mean) | barley | 72.00 | 0.06 | 0.33 | 0.14 | 0.03 | 0.04 | -0.09 |
| cell | IG | buckwheat | 40.00 | 0.88 | 0.01 | 0.34 | 0.01 | 0.15 | 0.17 |
| cell | Occlusion (mean) | buckwheat | 40.00 | 0.38 | 0.13 | 0.21 | 0.06 | 0.14 | 0.21 |
| month | SHAP | buckwheat | 40.00 | 0.55 | 0.53 | 0.62 | 0.01 | 0.08 | 0.78 |
| month | IG | buckwheat | 40.00 | 0.35 | 0.61 | 0.59 | 0.05 | 0.04 | 0.72 |
| month | Occlusion (mean) | buckwheat | 40.00 | 0.53 | 0.51 | 0.58 | 0.01 | 0.08 | 0.80 |
| month | Occlusion (drop) | buckwheat | 40.00 | 0.50 | 0.53 | 0.53 | 0.03 | 0.08 | 0.59 |
| band group | SHAP | buckwheat | 40.00 | 0.72 | 0.38 | 0.42 | 0.01 | 0.04 | 0.03 |
| band group | IG | buckwheat | 40.00 | 0.68 | 0.36 | 0.40 | 0.01 | 0.04 | -0.09 |
| band group | Occlusion (mean) | buckwheat | 40.00 | 0.55 | 0.36 | 0.40 | 0.01 | 0.04 | 0.14 |
| cell | IG | oats | 28.00 | 0.29 | 0.13 | 0.29 | 0.18 | 0.15 | 0.12 |
| cell | Occlusion (mean) | oats | 28.00 | 0.29 | 0.12 | 0.28 | 0.20 | 0.13 | 0.22 |
| month | SHAP | oats | 28.00 | 1.00 | 0.47 | 0.81 | 0.06 | 0.08 | 0.60 |
| month | IG | oats | 28.00 | 0.89 | 0.43 | 0.67 | 0.06 | 0.08 | 0.42 |
| month | Occlusion (mean) | oats | 28.00 | 0.71 | 0.37 | 0.50 | 0.06 | 0.08 | 0.16 |
| month | Occlusion (drop) | oats | 28.00 | 0.89 | 0.40 | 0.56 | 0.00 | 0.08 | 0.12 |
| band group | SHAP | oats | 28.00 | 0.14 | 0.23 | 0.20 | 0.06 | 0.05 | -0.06 |
| band group | IG | oats | 28.00 | 0.00 | 0.07 | 0.09 | 0.09 | 0.05 | -0.09 |
| band group | Occlusion (mean) | oats | 28.00 | 0.07 | 0.32 | 0.09 | 0.06 | 0.05 | -0.37 |
| cell | IG | dry pulses | 90.00 | 0.00 | 0.19 | 0.04 | 0.14 | 0.11 | 0.03 |
| cell | Occlusion (mean) | dry pulses | 90.00 | 0.01 | 0.23 | 0.08 | 0.14 | 0.11 | 0.20 |
| month | SHAP | dry pulses | 90.00 | 0.47 | 0.54 | 0.43 | 0.16 | 0.06 | 0.27 |
| month | IG | dry pulses | 90.00 | 0.57 | 0.55 | 0.39 | 0.00 | 0.05 | 0.08 |
| month | Occlusion (mean) | dry pulses | 90.00 | 0.56 | 0.55 | 0.51 | 0.02 | 0.05 | 0.13 |
| month | Occlusion (drop) | dry pulses | 90.00 | 0.64 | 0.55 | 0.46 | 0.00 | 0.05 | 0.20 |
| band group | SHAP | dry pulses | 90.00 | 0.00 | 0.19 | 0.01 | 0.08 | 0.06 | -0.66 |
| band group | IG | dry pulses | 90.00 | 0.00 | 0.43 | 0.06 | 0.09 | 0.03 | -0.20 |
| band group | Occlusion (mean) | dry pulses | 90.00 | 0.04 | 0.19 | 0.13 | 0.08 | 0.07 | -0.54 |
| cell | IG | maize | 40.00 | 0.23 | 0.15 | 0.33 | 0.05 | 0.14 | 0.12 |
| cell | Occlusion (mean) | maize | 40.00 | 0.42 | 0.10 | 0.39 | 0.09 | 0.14 | 0.11 |
| month | SHAP | maize | 40.00 | 0.50 | 0.46 | 0.60 | 0.10 | 0.06 | 0.55 |
| month | IG | maize | 40.00 | 0.65 | 0.45 | 0.70 | 0.01 | 0.09 | 0.39 |
| month | Occlusion (mean) | maize | 40.00 | 0.65 | 0.48 | 0.62 | 0.01 | 0.06 | 0.40 |
| month | Occlusion (drop) | maize | 40.00 | 0.72 | 0.44 | 0.64 | 0.01 | 0.09 | 0.43 |
| band group | SHAP | maize | 40.00 | 1.00 | 0.44 | 0.51 | 0.00 | 0.04 | 0.26 |
| band group | IG | maize | 40.00 | 0.60 | 0.31 | 0.47 | 0.00 | 0.04 | 0.09 |
| band group | Occlusion (mean) | maize | 40.00 | 1.00 | 0.43 | 0.72 | 0.00 | 0.04 | 0.54 |
| cell | IG | rye | 27.00 | 0.19 | 0.13 | 0.26 | 0.05 | 0.11 | -0.00 |
| cell | Occlusion (mean) | rye | 27.00 | 0.56 | 0.14 | 0.33 | 0.00 | 0.14 | 0.14 |
| month | SHAP | rye | 27.00 | 0.15 | 0.41 | 0.38 | 0.12 | 0.07 | 0.03 |
| month | IG | rye | 27.00 | 0.19 | 0.42 | 0.37 | 0.12 | 0.07 | 0.00 |
| month | Occlusion (mean) | rye | 27.00 | 0.37 | 0.47 | 0.51 | 0.12 | 0.07 | 0.31 |
| month | Occlusion (drop) | rye | 27.00 | 0.26 | 0.48 | 0.33 | 0.12 | 0.07 | 0.27 |
| band group | SHAP | rye | 27.00 | 0.56 | 0.19 | 0.59 | 0.03 | 0.06 | 0.26 |
| band group | IG | rye | 27.00 | 0.74 | 0.33 | 0.59 | 0.03 | 0.04 | 0.37 |
| band group | Occlusion (mean) | rye | 27.00 | 0.70 | 0.33 | 0.63 | 0.03 | 0.04 | 0.43 |

### The species' AUC main drivers (AUC in brackets)

| species | AUC top 5 cells | AUC top 3 months | AUC top 2 band groups |
|---|---|---|---|
| grassland | Oct VV (0.89), Aug VV (0.89), Nov VV (0.89), Sep VV (0.88), Apr NIR (0.84) | Oct (0.89), Aug (0.89), Nov (0.89) | VV (0.89), NIR (0.84) |
| forage legumes | May VV (0.74), Oct VV (0.73), Sep VV (0.73), Nov VV (0.73), Aug VV (0.72) | May (0.74), Oct (0.73), Sep (0.73) | VV (0.74), Red-edge (0.71) |
| wheat | Jun Red-edge (0.83), Jun VV (0.82), Jun Visible (0.80), Jul VH (0.78), Jul VV (0.77) | Jun (0.83), Jul (0.78), Sep (0.75) | Red-edge (0.83), VV (0.82) |
| barley | Aug SWIR (0.74), Jun VV (0.74), Aug Visible (0.74), Sep SWIR (0.73), Sep Visible (0.73) | Aug (0.74), Jun (0.74), Sep (0.73) | SWIR (0.74), VV (0.74) |
| rye | Jun SWIR (0.82), May SWIR (0.81), Jun Red-edge (0.79), May Visible (0.77), Jun Visible (0.77) | Jun (0.82), May (0.81), Apr (0.75) | SWIR (0.82), Red-edge (0.79) |
| oats | Jul VV (0.81), Jul VH (0.76), May NIR (0.74), May Red-edge (0.74), May Visible (0.70) | Jul (0.81), May (0.74), Jun (0.70) | VV (0.81), VH (0.76) |
| rapeseed | Jun VV (0.95), Jun VH (0.94), Jul VH (0.90), Jun Red-edge (0.89), Jul VV (0.89) | Jun (0.95), Jul (0.90), May (0.86) | VV (0.95), VH (0.94) |
| dry pulses | Jun VH (0.82), Jun VV (0.79), May SWIR (0.78), May Visible (0.78), Sep NIR (0.74) | Jun (0.82), May (0.78), Sep (0.74) | VH (0.82), VV (0.79) |
| buckwheat | Sep VH (0.90), Aug VH (0.89), Jul Red-edge (0.89), Jul NIR (0.86), Jun SWIR (0.85) | Sep (0.90), Aug (0.89), Jul (0.89) | VH (0.90), Red-edge (0.89) |
| maize | Sep Red-edge (0.92), Jun SWIR (0.92), Aug Red-edge (0.91), Jun Visible (0.91), Sep SWIR (0.88) | Sep (0.92), Jun (0.92), Aug (0.91) | Red-edge (0.92), SWIR (0.92) |

### The fields' most frequent main drivers

| species | level | method | most frequent main drivers (share of fields) |
|---|---|---|---|
| grassland | cell | IG | May NIR 60%, Sep NIR 11%, Nov Red-edge 11% |
| grassland | month | SHAP | Jun 39%, May 33%, Jul 19% |
| grassland | band group | SHAP | NIR 51%, SWIR 40%, Red-edge 9% |
| forage legumes | cell | IG | Jun Red-edge 42%, Jul Red-edge 18%, Jul NIR 12% |
| forage legumes | month | SHAP | Jun 48%, Jul 41%, May 9% |
| forage legumes | band group | SHAP | SWIR 62%, Red-edge 23%, NIR 9% |
| wheat | cell | IG | Jun Red-edge 58%, Jul Red-edge 10%, Jun NIR 8% |
| wheat | month | SHAP | Jun 63%, Jul 28%, Aug 5% |
| wheat | band group | SHAP | Red-edge 77%, NIR 17%, SWIR 5% |
| barley | cell | IG | May NIR 28%, Jun NIR 21%, May Red-edge 17% |
| barley | month | SHAP | Jun 40%, May 33%, Jul 26% |
| barley | band group | SHAP | Red-edge 60%, Visible 35%, NIR 4% |
| rye | cell | IG | May Red-edge 44%, Jul Red-edge 19%, May NIR 11% |
| rye | month | SHAP | Jul 70%, Nov 15%, May 7% |
| rye | band group | SHAP | Visible 37%, SWIR 30%, Red-edge 26% |
| oats | cell | IG | Jun NIR 32%, May NIR 21%, Jun Visible 18% |
| oats | month | SHAP | May 82%, Jun 14%, Jul 4% |
| oats | band group | SHAP | NIR 39%, Visible 29%, VV 14% |
| rapeseed | cell | IG | May NIR 47%, Jun Red-edge 25%, May Red-edge 23% |
| rapeseed | month | SHAP | Jun 61%, Jul 38%, Aug 2% |
| rapeseed | band group | SHAP | VH 38%, SWIR 34%, Red-edge 17% |
| dry pulses | cell | IG | Jun Red-edge 39%, Jul Red-edge 29%, Jul NIR 14% |
| dry pulses | month | SHAP | Jul 53%, Jun 43%, May 3% |
| dry pulses | band group | SHAP | NIR 44%, Visible 30%, Red-edge 23% |
| buckwheat | cell | IG | Jul Red-edge 88%, Jun NIR 5%, May Red-edge 5% |
| buckwheat | month | SHAP | Jul 55%, Jun 45% |
| buckwheat | band group | SHAP | Red-edge 72%, NIR 18%, SWIR 10% |
| maize | cell | IG | Jun Red-edge 30%, Jul NIR 28%, Aug Red-edge 22% |
| maize | month | SHAP | Jul 50%, Sep 28%, Aug 20% |
| maize | band group | SHAP | Red-edge 100% |

### Model level per species

| species | level | method | main driver | hit | overlap | gap | rho |
|---|---|---|---|---|---|---|---|
| grassland | cell | IG | May NIR | 0.00 | 0.20 | 0.10 | -0.06 |
| grassland | cell | Occlusion (mean) | Jun SWIR | 0.00 | 0.00 | 0.25 | -0.15 |
| grassland | month | SHAP | Jun | 0.00 | 0.00 | 0.15 | -0.17 |
| grassland | month | IG | Jul | 0.00 | 0.00 | 0.09 | -0.16 |
| grassland | month | Occlusion (mean) | May | 0.00 | 0.00 | 0.07 | -0.44 |
| grassland | month | Occlusion (drop) | Jun | 0.00 | 0.33 | 0.15 | -0.32 |
| grassland | band group | SHAP | NIR | 1.00 | 0.50 | 0.05 | 0.37 |
| grassland | band group | IG | NIR | 1.00 | 0.50 | 0.05 | 0.20 |
| grassland | band group | Occlusion (mean) | NIR | 1.00 | 0.50 | 0.05 | -0.03 |
| forage legumes | cell | IG | Jun Red-edge | 0.00 | 0.00 | 0.10 | 0.13 |
| forage legumes | cell | Occlusion (mean) | Jul SWIR | 0.00 | 0.00 | 0.06 | -0.07 |
| forage legumes | month | SHAP | Jul | 0.00 | 0.33 | 0.04 | -0.30 |
| forage legumes | month | IG | Jul | 0.00 | 0.33 | 0.04 | 0.17 |
| forage legumes | month | Occlusion (mean) | Jul | 0.00 | 0.00 | 0.04 | 0.14 |
| forage legumes | month | Occlusion (drop) | May | 1.00 | 0.33 | 0.00 | 0.47 |
| forage legumes | band group | SHAP | SWIR | 0.00 | 0.50 | 0.06 | 0.09 |
| forage legumes | band group | IG | Red-edge | 1.00 | 0.50 | 0.03 | 0.20 |
| forage legumes | band group | Occlusion (mean) | SWIR | 0.00 | 0.50 | 0.06 | 0.09 |
| wheat | cell | IG | Jun Red-edge | 1.00 | 0.20 | 0.00 | 0.08 |
| wheat | cell | Occlusion (mean) | Jun Red-edge | 1.00 | 0.20 | 0.00 | 0.18 |
| wheat | month | SHAP | Jun | 1.00 | 0.67 | 0.00 | 0.58 |
| wheat | month | IG | Jun | 1.00 | 0.67 | 0.00 | 0.58 |
| wheat | month | Occlusion (mean) | Jun | 1.00 | 1.00 | 0.00 | 0.55 |
| wheat | month | Occlusion (drop) | Jun | 1.00 | 0.67 | 0.00 | 0.57 |
| wheat | band group | SHAP | Red-edge | 1.00 | 0.50 | 0.00 | 0.37 |
| wheat | band group | IG | Red-edge | 1.00 | 0.50 | 0.00 | 0.26 |
| wheat | band group | Occlusion (mean) | Red-edge | 1.00 | 0.50 | 0.00 | -0.09 |
| barley | cell | IG | Jun NIR | 0.00 | 0.00 | 0.17 | 0.02 |
| barley | cell | Occlusion (mean) | Jun Visible | 0.00 | 0.00 | 0.15 | 0.10 |
| barley | month | SHAP | Jun | 1.00 | 0.33 | 0.00 | 0.58 |
| barley | month | IG | Jul | 0.00 | 0.33 | 0.02 | 0.60 |
| barley | month | Occlusion (mean) | Jul | 0.00 | 0.33 | 0.02 | 0.78 |
| barley | month | Occlusion (drop) | Jul | 0.00 | 0.33 | 0.02 | 0.19 |
| barley | band group | SHAP | Red-edge | 0.00 | 0.00 | 0.03 | -0.09 |
| barley | band group | IG | NIR | 0.00 | 0.00 | 0.04 | -0.14 |
| barley | band group | Occlusion (mean) | Red-edge | 0.00 | 0.00 | 0.03 | -0.09 |
| rye | cell | IG | May Visible | 1.00 | 0.40 | 0.04 | -0.03 |
| rye | cell | Occlusion (mean) | Jun SWIR | 1.00 | 0.20 | 0.00 | 0.16 |
| rye | month | SHAP | Jul | 0.00 | 0.33 | 0.12 | 0.01 |
| rye | month | IG | Jul | 0.00 | 0.67 | 0.12 | 0.04 |
| rye | month | Occlusion (mean) | Jul | 0.00 | 0.67 | 0.12 | 0.38 |
| rye | month | Occlusion (drop) | Jul | 0.00 | 0.33 | 0.12 | 0.55 |
| rye | band group | SHAP | Red-edge | 1.00 | 0.50 | 0.03 | 0.66 |
| rye | band group | IG | Red-edge | 1.00 | 1.00 | 0.03 | 0.77 |
| rye | band group | Occlusion (mean) | Red-edge | 1.00 | 0.50 | 0.03 | 0.31 |
| oats | cell | IG | May NIR | 1.00 | 0.40 | 0.06 | 0.25 |
| oats | cell | Occlusion (mean) | Jun Visible | 0.00 | 0.20 | 0.24 | 0.46 |
| oats | month | SHAP | May | 1.00 | 1.00 | 0.06 | 0.95 |
| oats | month | IG | May | 1.00 | 1.00 | 0.06 | 0.76 |
| oats | month | Occlusion (mean) | May | 1.00 | 0.67 | 0.06 | 0.85 |
| oats | month | Occlusion (drop) | Jul | 1.00 | 1.00 | 0.00 | 0.24 |
| oats | band group | SHAP | Visible | 0.00 | 0.00 | 0.10 | -0.09 |
| oats | band group | IG | Visible | 0.00 | 0.00 | 0.10 | -0.09 |
| oats | band group | Occlusion (mean) | NIR | 0.00 | 0.00 | 0.06 | -0.37 |
| rapeseed | cell | IG | Jun Red-edge | 1.00 | 0.20 | 0.06 | 0.24 |
| rapeseed | cell | Occlusion (mean) | Jul SWIR | 0.00 | 0.20 | 0.15 | 0.26 |
| rapeseed | month | SHAP | Jul | 1.00 | 0.67 | 0.05 | 0.71 |
| rapeseed | month | IG | Jun | 1.00 | 0.67 | 0.00 | 0.37 |
| rapeseed | month | Occlusion (mean) | Jul | 1.00 | 0.67 | 0.05 | 0.73 |
| rapeseed | month | Occlusion (drop) | Jul | 1.00 | 0.67 | 0.05 | 0.71 |
| rapeseed | band group | SHAP | VH | 1.00 | 0.50 | 0.02 | 0.26 |
| rapeseed | band group | IG | Red-edge | 0.00 | 0.00 | 0.06 | 0.31 |
| rapeseed | band group | Occlusion (mean) | Red-edge | 0.00 | 0.00 | 0.06 | -0.20 |
| dry pulses | cell | IG | Jun Red-edge | 0.00 | 0.00 | 0.14 | 0.03 |
| dry pulses | cell | Occlusion (mean) | Jun Red-edge | 0.00 | 0.00 | 0.14 | 0.31 |
| dry pulses | month | SHAP | Jun | 1.00 | 0.33 | 0.00 | 0.52 |
| dry pulses | month | IG | Jun | 1.00 | 0.33 | 0.00 | 0.22 |
| dry pulses | month | Occlusion (mean) | Jun | 1.00 | 0.67 | 0.00 | 0.29 |
| dry pulses | month | Occlusion (drop) | Jun | 1.00 | 0.67 | 0.00 | 0.44 |
| dry pulses | band group | SHAP | Visible | 0.00 | 0.00 | 0.04 | -0.60 |
| dry pulses | band group | IG | Red-edge | 0.00 | 0.00 | 0.09 | -0.20 |
| dry pulses | band group | Occlusion (mean) | NIR | 0.00 | 0.00 | 0.08 | -0.89 |
| buckwheat | cell | IG | Jul Red-edge | 1.00 | 0.40 | 0.01 | 0.25 |
| buckwheat | cell | Occlusion (mean) | Sep SWIR | 0.00 | 0.20 | 0.06 | 0.18 |
| buckwheat | month | SHAP | Jul | 1.00 | 0.67 | 0.01 | 0.91 |
| buckwheat | month | IG | Jun | 0.00 | 0.67 | 0.05 | 0.85 |
| buckwheat | month | Occlusion (mean) | Jun | 0.00 | 0.67 | 0.05 | 0.90 |
| buckwheat | month | Occlusion (drop) | Jun | 0.00 | 0.67 | 0.05 | 0.90 |
| buckwheat | band group | SHAP | Red-edge | 1.00 | 0.50 | 0.01 | 0.09 |
| buckwheat | band group | IG | Red-edge | 1.00 | 0.50 | 0.01 | -0.09 |
| buckwheat | band group | Occlusion (mean) | Red-edge | 1.00 | 0.50 | 0.01 | 0.26 |
| maize | cell | IG | Aug Red-edge | 1.00 | 0.40 | 0.01 | 0.14 |
| maize | cell | Occlusion (mean) | Jul NIR | 0.00 | 0.20 | 0.37 | 0.22 |
| maize | month | SHAP | Jul | 0.00 | 0.67 | 0.18 | 0.51 |
| maize | month | IG | Aug | 1.00 | 0.67 | 0.01 | 0.34 |
| maize | month | Occlusion (mean) | Aug | 1.00 | 0.67 | 0.01 | 0.29 |
| maize | month | Occlusion (drop) | Aug | 1.00 | 0.67 | 0.01 | 0.38 |
| maize | band group | SHAP | Red-edge | 1.00 | 0.50 | 0.00 | 0.26 |
| maize | band group | IG | Red-edge | 1.00 | 0.50 | 0.00 | 0.09 |
| maize | band group | Occlusion (mean) | Red-edge | 1.00 | 1.00 | 0.00 | 0.71 |

### Confidence: only the most confident correct predictions, pooled (p = the field's species probability, geometric mean over its interior pixels)

| fields | n | species (5 fields or more) | level | method | hit | hit, other species' maps | chance | AUC gap, median |
|---|---|---|---|---|---|---|---|---|
| all correct | 544 | 10 | cell | IG | 0.20 | 0.14 | 0.07 | 0.10 |
| all correct | 544 | 10 | month | SHAP | 0.50 | 0.54 | 0.25 | 0.05 |
| all correct | 544 | 10 | band group | SHAP | 0.37 | 0.30 | 0.33 | 0.04 |
| p >= 0.9 | 305 | 8 | cell | IG | 0.22 | 0.14 | 0.07 | 0.10 |
| p >= 0.9 | 305 | 8 | month | SHAP | 0.49 | 0.56 | 0.25 | 0.04 |
| p >= 0.9 | 305 | 8 | band group | SHAP | 0.42 | 0.30 | 0.33 | 0.04 |
| p >= 0.95 | 249 | 8 | cell | IG | 0.23 | 0.14 | 0.07 | 0.10 |
| p >= 0.95 | 249 | 8 | month | SHAP | 0.50 | 0.56 | 0.25 | 0.04 |
| p >= 0.95 | 249 | 8 | band group | SHAP | 0.43 | 0.31 | 0.33 | 0.04 |
| p >= 0.99 | 118 | 6 | cell | IG | 0.28 | 0.14 | 0.07 | 0.10 |
| p >= 0.99 | 118 | 6 | month | SHAP | 0.65 | 0.58 | 0.25 | 0.00 |
| p >= 0.99 | 118 | 6 | band group | SHAP | 0.42 | 0.35 | 0.33 | 0.03 |
| top 5 % (p >= 0.998) | 28 | 3 | cell | IG | 0.36 | 0.11 | 0.07 | 0.07 |
| top 5 % (p >= 0.998) | 28 | 3 | month | SHAP | 0.43 | 0.53 | 0.25 | 0.02 |
| top 5 % (p >= 0.998) | 28 | 3 | band group | SHAP | 0.68 | 0.41 | 0.33 | 0.00 |

### Confidence per species: all correct, p >= 0.95 (5 fields or more), and the most against the least confident half of each species' correct fields

| species | level | method | fields | hit, all correct | fields p >= 0.95 | hit, p >= 0.95 | hit, other maps, p >= 0.95 | hit, most confident half | hit, least confident half |
|---|---|---|---|---|---|---|---|---|---|
| grassland | cell | IG | 57 | 0.00 | 39 | 0.00 | 0.09 | 0.00 | 0.00 |
| grassland | month | SHAP | 57 | 0.05 | 39 | 0.08 | 0.61 | 0.07 | 0.04 |
| grassland | band group | SHAP | 57 | 0.51 | 39 | 0.51 | 0.18 | 0.55 | 0.46 |
| forage legumes | cell | IG | 66 | 0.00 | 15 | 0.00 | 0.27 | 0.00 | 0.00 |
| forage legumes | month | SHAP | 66 | 0.09 | 15 | 0.00 | 0.60 | 0.09 | 0.09 |
| forage legumes | band group | SHAP | 66 | 0.23 | 15 | 0.13 | 0.33 | 0.15 | 0.30 |
| wheat | cell | IG | 60 | 0.60 | 26 | 0.92 | 0.21 | 0.87 | 0.33 |
| wheat | month | SHAP | 60 | 0.92 | 26 | 0.92 | 0.59 | 0.93 | 0.90 |
| wheat | band group | SHAP | 60 | 0.78 | 26 | 0.88 | 0.41 | 0.90 | 0.67 |
| barley | cell | IG | 72 | 0.00 | 40 | 0.00 | 0.09 | 0.00 | 0.00 |
| barley | month | SHAP | 72 | 0.40 | 40 | 0.38 | 0.56 | 0.36 | 0.44 |
| barley | band group | SHAP | 72 | 0.00 | 40 | 0.00 | 0.27 | 0.00 | 0.00 |
| rye | cell | IG | 27 | 0.19 | 1 |  |  | 0.21 | 0.15 |
| rye | month | SHAP | 27 | 0.15 | 1 |  |  | 0.07 | 0.23 |
| rye | band group | SHAP | 27 | 0.56 | 1 |  |  | 0.57 | 0.54 |
| oats | cell | IG | 28 | 0.29 | 0 |  |  | 0.29 | 0.29 |
| oats | month | SHAP | 28 | 1.00 | 0 |  |  | 1.00 | 1.00 |
| oats | band group | SHAP | 28 | 0.14 | 0 |  |  | 0.14 | 0.14 |
| rapeseed | cell | IG | 64 | 0.25 | 45 | 0.24 | 0.14 | 0.22 | 0.28 |
| rapeseed | month | SHAP | 64 | 0.98 | 45 | 1.00 | 0.59 | 1.00 | 0.97 |
| rapeseed | band group | SHAP | 64 | 0.38 | 45 | 0.42 | 0.35 | 0.41 | 0.34 |
| dry pulses | cell | IG | 90 | 0.00 | 40 | 0.00 | 0.23 | 0.00 | 0.00 |
| dry pulses | month | SHAP | 90 | 0.47 | 40 | 0.45 | 0.53 | 0.42 | 0.51 |
| dry pulses | band group | SHAP | 90 | 0.00 | 40 | 0.00 | 0.20 | 0.00 | 0.00 |
| buckwheat | cell | IG | 40 | 0.88 | 20 | 0.90 | 0.01 | 0.90 | 0.85 |
| buckwheat | month | SHAP | 40 | 0.55 | 20 | 0.50 | 0.56 | 0.50 | 0.60 |
| buckwheat | band group | SHAP | 40 | 0.72 | 20 | 0.90 | 0.42 | 0.90 | 0.55 |
| maize | cell | IG | 40 | 0.23 | 23 | 0.17 | 0.17 | 0.20 | 0.25 |
| maize | month | SHAP | 40 | 0.50 | 23 | 0.43 | 0.44 | 0.40 | 0.60 |
| maize | band group | SHAP | 40 | 1.00 | 23 | 1.00 | 0.44 | 1.00 | 1.00 |

### Confidence, the mean over species (each species weighs the same)

| level | method | hit, all correct | hit, p >= 0.95 | hit, most confident half | hit, least confident half |
|---|---|---|---|---|---|
| cell | IG | 0.24 | 0.28 | 0.27 | 0.22 |
| month | SHAP | 0.51 | 0.47 | 0.48 | 0.54 |
| band group | SHAP | 0.43 | 0.48 | 0.46 | 0.40 |
