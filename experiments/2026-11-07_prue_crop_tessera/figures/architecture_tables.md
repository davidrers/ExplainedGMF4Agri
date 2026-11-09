# Architecture note tables: TESSERA frozen, crop loss weight 0.01

## Test-set totals, recomputed field by field

| | recomputed | results.json |
|---|---|---|
| declared parcels (pieces) | 55,575 | 55,575 |
| fields drawn and scored | 50,680 | 50,680 |
| fields matching a parcel (IoU > 0.5) | 23,948 | 23,948 |
| of which with the right crop | 19,711 (82.3%) | 19,711 |
| of which with a wrong crop | 4,237 (17.7%) | |
| fields matching no parcel | 26,732 | 26,732 |
| fields mostly on a parcel outside the scheme, not scored | 1,389 | 1,389 |
| declared parcels found with the right crop | 35.5% | |

## Crop of the matched fields, per declared crop

| Declared crop | matched fields | right crop | most frequent wrong crop |
|---|---|---|---|
| grassland | 9,630 | 96% | legumes, green harvest (4%) |
| winter wheat | 2,772 | 97% | spring barley (1%) |
| legumes, green harvest | 2,085 | 46% | grassland (46%) |
| spring barley | 2,023 | 87% | oats (6%) |
| winter rapeseed | 1,187 | 98% | winter wheat (1%) |
| oats | 1,167 | 64% | spring barley (20%) |
| spring wheat | 1,150 | 54% | spring barley (27%) |
| clover | 1,127 | 42% | legumes, green harvest (26%) |
| peas | 669 | 91% | spring barley (3%) |
| winter barley | 488 | 84% | winter wheat (14%) |
| beans | 338 | 86% | peas (5%) |
| alfalfa | 252 | 7% | legumes, green harvest (54%) |
| rye | 250 | 62% | winter wheat (25%) |
| spring rapeseed | 209 | 78% | peas (9%) |
| buckwheat | 201 | 87% | oats (5%) |
| grain maize | 166 | 97% | beans (1%) |
| potatoes | 133 | 77% | grassland (9%) |
| dried pulses | 82 | 0% | grassland (39%) |
| orchards | 10 | 0% | grassland (90%) |
| vegetables | 9 | 0% | grassland (44%) |

## The most frequent wrong crops

| Declared | Voted | fields | share of all wrong |
|---|---|---|---|
| legumes, green harvest | grassland | 952 | 22% |
| grassland | legumes, green harvest | 358 | 8% |
| spring wheat | spring barley | 307 | 7% |
| clover | legumes, green harvest | 289 | 7% |
| clover | grassland | 285 | 7% |
| oats | spring barley | 232 | 5% |
| spring wheat | oats | 154 | 4% |
| alfalfa | legumes, green harvest | 137 | 3% |
| spring barley | oats | 118 | 3% |
| legumes, green harvest | clover | 110 | 3% |

## The chips of the examples

| Chip | why | parcels | fields | matched | right crop | wrong crop | no match | missed |
|---|---|---|---|---|---|---|---|---|
| EE_02300_01806 | good: 90th percentile | 38 | 28 | 24 | 22 | 2 | 4 | 14 |
| EE_02298_01819 | typical: median | 33 | 38 | 16 | 12 | 4 | 20 | 17 |
| EE_02242_01775 | poor: 10th percentile | 93 | 80 | 28 | 17 | 11 | 52 | 65 |
| EE_02383_01779 | most fields with a wrong crop | 87 | 101 | 47 | 25 | 22 | 48 | 40 |
