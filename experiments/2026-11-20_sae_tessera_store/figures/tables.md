# Tables

## Reconstruction, test pixels (uniform sample)

| method | run | k | L0 | FVE | cosine | dead share |
|---|---|---|---|---|---|---|
| batchtopk | batchtopk_k8_s0 | 8 | 8.0 | 0.844 | 0.911 | 0.000 |
| batchtopk | batchtopk_k16_s0 | 16 | 16.1 | 0.902 | 0.945 | 0.000 |
| batchtopk | batchtopk_k16_s1 | 16 | 16.1 | 0.902 | 0.946 | 0.000 |
| batchtopk | batchtopk_k16_s2 | 16 | 16.1 | 0.903 | 0.946 | 0.000 |
| batchtopk | batchtopk_k32_s0 | 32 | 32.0 | 0.951 | 0.973 | 0.000 |
| frozen | frozen_k8_s0 | 8 | 8.1 | 0.390 | 0.567 | 0.000 |
| frozen | frozen_k16_s0 | 16 | 16.1 | 0.550 | 0.706 | 0.000 |
| frozen | frozen_k32_s0 | 32 | 32.3 | 0.715 | 0.827 | 0.000 |
| pca |  | 8 | 8.0 | 0.575 | 0.750 | nan |
| pca |  | 16 | 16.0 | 0.733 | 0.851 | nan |
| pca |  | 32 | 32.0 | 0.857 | 0.924 | nan |
| raw |  | 8 | 8.0 | 0.062 | 0.226 | nan |
| raw |  | 16 | 16.0 | 0.125 | 0.334 | nan |
| raw |  | 32 | 32.0 | 0.250 | 0.484 | nan |

## Concept alignment, test (folded AUROC ≥ 0.65)

| representation | units | live | live aligned with ≤ 1 concept | exactly 1 | none | median concepts per live unit | concepts with an aligned unit |
|---|---|---|---|---|---|---|---|
| SAE feature | 1024 | 313 | 0.93 | 0.10 | 0.83 | 0 | 21 of 21 |
| frozen-dictionary feature | 1024 | 534 | 0.91 | 0.11 | 0.79 | 0 | 20 of 21 |
| raw dimension | 128 | 128 | 0.00 | 0.00 | 0.00 | 9 | 21 of 21 |
| PCA component | 128 | 128 | 0.29 | 0.21 | 0.08 | 3 | 21 of 21 |

## Best single unit per class, test field interiors (precision / recall / F1)

| class | test pixels | SAE feature | frozen-dictionary feature | raw dimension | PCA component |
|---|---|---|---|---|---|
| winter rapeseed | 5000 | f184: 0.98 / 0.90 / **0.93** | f648: 0.97 / 0.90 / **0.93** | d100: 0.85 / 0.86 / **0.85** | d10: 0.94 / 0.90 / **0.92** |
| grassland | 5000 | f497: 0.75 / 0.80 / **0.77** | f312: 0.72 / 0.76 / **0.74** | d107: 0.69 / 0.84 / **0.76** | d2: 0.66 / 0.80 / **0.72** |
| maize | 5000 | f769: 0.60 / 0.95 / **0.74** | f455: 0.89 / 0.82 / **0.85** | d100: 0.40 / 0.60 / **0.48** | d0: 0.19 / 0.68 / **0.29** |
| winter barley | 5000 | f284: 0.53 / 0.85 / **0.65** | f211: 0.45 / 0.45 / **0.45** | d30: 0.30 / 0.57 / **0.39** | d26: 0.20 / 0.25 / **0.22** |
| winter wheat | 5000 | f569: 0.76 / 0.52 / **0.62** | f790: 0.71 / 0.77 / **0.74** | d7: 0.52 / 0.74 / **0.61** | d6: 0.66 / 0.79 / **0.72** |
| buckwheat | 5000 | f844: 0.38 / 0.90 / **0.54** | f1002: 0.19 / 0.51 / **0.28** | d24: 0.11 / 0.20 / **0.14** | d126: 0.12 / 0.17 / **0.14** |
| spring barley | 5000 | f291: 0.34 / 0.65 / **0.45** | f152: 0.47 / 0.64 / **0.54** | d105: 0.34 / 0.69 / **0.45** | d9: 0.28 / 0.84 / **0.42** |
| beans | 5000 | f282: 0.26 / 0.78 / **0.40** | f487: 0.13 / 0.44 / **0.20** | d3: 0.19 / 0.38 / **0.25** | d6: 0.20 / 0.46 / **0.28** |
| green legumes | 5000 | f131: 0.30 / 0.45 / **0.36** | f222: 0.25 / 0.34 / **0.29** | d71: 0.23 / 0.63 / **0.34** | d5: 0.18 / 0.78 / **0.30** |
| spring wheat | 5000 | f494: 0.23 / 0.68 / **0.34** | f127: 0.30 / 0.57 / **0.39** | d78: 0.20 / 0.36 / **0.25** | d10: 0.27 / 0.35 / **0.30** |
| potatoes | 5000 | f256: 0.26 / 0.31 / **0.28** | f158: 0.13 / 0.29 / **0.18** | d31: 0.23 / 0.48 / **0.31** | d9: 0.19 / 0.32 / **0.24** |
| rye | 5000 | f748: 0.18 / 0.57 / **0.27** | f432: 0.14 / 0.42 / **0.21** | d58: 0.17 / 0.31 / **0.22** | d51: 0.18 / 0.25 / **0.21** |
| spring rapeseed | 5000 | f661: 0.19 / 0.44 / **0.26** | f631: 0.20 / 0.39 / **0.26** | d73: 0.11 / 0.33 / **0.16** | d11: 0.24 / 0.22 / **0.23** |
| peas | 5000 | f282: 0.20 / 0.37 / **0.26** | f644: 0.35 / 0.64 / **0.45** | d103: 0.22 / 0.42 / **0.29** | d6: 0.17 / 0.74 / **0.28** |
| clover | 5000 | f561: 0.23 / 0.27 / **0.25** | f10: 0.22 / 0.17 / **0.19** | d26: 0.10 / 0.23 / **0.14** | d28: 0.13 / 0.21 / **0.16** |
| alfalfa | 5000 | f561: 0.13 / 0.43 / **0.20** | f264: 0.19 / 0.12 / **0.15** | d47: 0.06 / 0.22 / **0.09** | d18: 0.09 / 0.16 / **0.11** |
| oats | 5000 | f373: 0.15 / 0.29 / **0.19** | f99: 0.17 / 0.26 / **0.21** | d78: 0.12 / 0.30 / **0.17** | d9: 0.11 / 0.70 / **0.18** |
| dry pulses | 5000 | f912: 0.04 / 0.33 / **0.07** | f402: 0.03 / 0.12 / **0.05** | d93: 0.06 / 0.24 / **0.10** | d12: 0.06 / 0.24 / **0.09** |
| orchards | 1404 | f896: 0.00 / 0.00 / **0.00** | f505: 0.00 / 0.01 / **0.00** | d17: 0.00 / 0.06 / **0.00** | d12: 0.00 / 0.04 / **0.00** |
| vegetables | 697 | f224: 0.00 / 0.01 / **0.00** | f254: 0.00 / 0.00 / **0.00** | d47: 0.00 / 0.02 / **0.00** | d67: 0.00 / 0.01 / **0.00** |

## Feature families and the crop map from families alone, test field interiors

Live SAE features: 331. Macro-F1: SAE families 0.206 (random families: median 0.023, 97.5th percentile 0.064, p = 0.000); frozen-dictionary families 0.335 (random: median 0.024); the trained head on the same pixels 0.650.

| class | SAE family size | SAE families F1 | frozen family size | frozen families F1 | head F1 |
|---|---|---|---|---|---|
| grassland | 52 | 0.08 | 62 | 0.66 | 0.86 |
| winter wheat | 7 | 0.63 | 32 | 0.77 | 0.95 |
| green legumes | 6 | 0.13 | 14 | 0.25 | 0.52 |
| spring barley | 6 | 0.22 | 16 | 0.50 | 0.82 |
| oats | 5 | 0.14 | 12 | 0.18 | 0.61 |
| clover | 13 | 0.01 | 10 | 0.17 | 0.53 |
| winter rapeseed | 12 | 0.93 | 46 | 0.94 | 0.98 |
| spring wheat | 5 | 0.15 | 18 | 0.23 | 0.70 |
| peas | 4 | 0.03 | 15 | 0.39 | 0.89 |
| potatoes | 22 | 0.12 | 25 | 0.10 | 0.86 |
| winter barley | 8 | 0.02 | 27 | 0.48 | 0.93 |
| beans | 9 | 0.18 | 12 | 0.20 | 0.87 |
| vegetables | 20 | 0.00 | 23 | 0.00 | 0.00 |
| rye | 17 | 0.19 | 15 | 0.13 | 0.75 |
| alfalfa | 13 | 0.07 | 15 | 0.14 | 0.10 |
| spring rapeseed | 13 | 0.21 | 12 | 0.27 | 0.77 |
| buckwheat | 13 | 0.24 | 24 | 0.46 | 0.89 |
| maize | 21 | 0.77 | 31 | 0.80 | 0.96 |
| dry pulses | 14 | 0.02 | 20 | 0.00 | 0.01 |
| orchards | 60 | 0.00 | 77 | 0.00 | 0.00 |

## The crop head on reconstructions, all 1475 test chips

Agreement of the exported head with the stored test predictions: 0.99983. Macro-F1 convention `macro_f1_present` (the logged test Macro-F1 is 0.6228).

| input to the head | Macro-F1, all labelled pixels | accuracy | Macro-F1, field interiors | accuracy |
|---|---|---|---|---|
| embedding | 0.6229 | 0.7949 | 0.6506 | 0.8131 |
| batchtopk_k16_s0 | 0.5579 | 0.7512 | 0.5875 | 0.7694 |
| frozen_k16_s0 | 0.3528 | 0.6098 | 0.3745 | 0.6312 |
| pca16 | 0.2279 | 0.6293 | 0.2339 | 0.6323 |

## Ablation of each class's family, test field interiors

| class | family size | recall before | own drop | strongest feature only | random sets: mean / 97.5 % | p | other classes' mean drop | its pixels go to |
|---|---|---|---|---|---|---|---|---|
| winter rapeseed | 12 | 0.98 | 95.7 pp | 95.7 pp | -0.0 / 0.2 pp | 0.00 | 4.9 pp | spring barley 43 %, winter wheat 18 %, grassland 15 % |
| maize | 21 | 0.95 | 95.3 pp | 45.1 pp | 0.2 / 1.3 pp | 0.00 | 5.6 pp | spring barley 57 %, spring wheat 19 %, oats 16 % |
| potatoes | 22 | 0.80 | 78.6 pp | 22.3 pp | 7.8 / 70.4 pp | 0.00 | 11.5 pp | grassland 38 %, clover 19 %, maize 16 % |
| winter barley | 8 | 0.91 | 73.1 pp | 75.2 pp | 2.6 / 44.4 pp | 0.00 | 0.1 pp | winter wheat 79 %, winter barley 18 %, spring barley 2 % |
| buckwheat | 13 | 0.85 | 68.4 pp | 65.7 pp | 0.4 / 5.0 pp | 0.00 | 0.6 pp | clover 21 %, peas 20 %, buckwheat 16 % |
| rye | 17 | 0.65 | 61.8 pp | 5.8 pp | -0.7 / 2.4 pp | 0.00 | 1.1 pp | winter wheat 71 %, winter barley 7 %, grassland 7 % |
| winter wheat | 7 | 0.95 | 60.5 pp | 56.9 pp | 0.0 / 0.6 pp | 0.00 | 2.5 pp | winter wheat 35 %, spring barley 24 %, grassland 16 % |
| grassland | 52 | 0.92 | 19.4 pp | 7.9 pp | -0.4 / 1.6 pp | 0.00 | -1.1 pp | grassland 73 %, green legumes 18 %, clover 7 % |
| green legumes | 6 | 0.51 | 15.9 pp | 13.4 pp | 0.3 / 7.1 pp | 0.00 | -0.3 pp | grassland 54 %, green legumes 35 %, clover 9 % |
| beans | 9 | 0.82 | 14.0 pp | 1.3 pp | 1.7 / 24.8 pp | 0.07 | 0.7 pp | beans 68 %, peas 9 %, oats 7 % |
| spring barley | 6 | 0.84 | 12.1 pp | 11.6 pp | 1.6 / 27.3 pp | 0.06 | -0.1 pp | spring barley 72 %, spring wheat 17 %, oats 6 % |
| spring rapeseed | 13 | 0.71 | 9.9 pp | 0.5 pp | 2.0 / 31.7 pp | 0.07 | -0.1 pp | spring rapeseed 62 %, peas 10 %, grassland 8 % |
| spring wheat | 5 | 0.68 | 6.3 pp | 3.7 pp | 3.3 / 41.0 pp | 0.09 | 0.2 pp | spring wheat 62 %, spring barley 25 %, oats 7 % |
| alfalfa | 13 | 0.05 | 5.3 pp | -0.8 pp | 0.0 / 1.0 pp | 0.00 | 1.3 pp | grassland 40 %, green legumes 33 %, spring barley 9 % |
| clover | 13 | 0.48 | 4.9 pp | 1.4 pp | 0.5 / 5.1 pp | 0.03 | 0.2 pp | clover 43 %, green legumes 29 %, grassland 22 % |
| oats | 5 | 0.57 | 4.9 pp | 4.1 pp | 0.3 / 4.1 pp | 0.02 | -0.1 pp | oats 53 %, spring barley 21 %, spring wheat 11 % |
| vegetables | 20 | 0.00 | 0.0 pp | 0.0 pp | 0.0 / 0.0 pp | 1.00 | 0.2 pp | grassland 48 %, clover 17 %, spring rapeseed 11 % |
| orchards | 60 | 0.00 | 0.0 pp | 0.0 pp | 0.0 / 0.0 pp | 1.00 | 0.8 pp | grassland 77 %, clover 7 %, green legumes 5 % |
| dry pulses | 14 | 0.00 | -0.1 pp | -0.1 pp | 0.1 / 0.4 pp | 0.97 | 0.2 pp | clover 34 %, grassland 33 %, green legumes 17 % |
| peas | 4 | 0.88 | -0.2 pp | -0.3 pp | 1.5 / 22.8 pp | 0.89 | -0.0 pp | peas 88 %, spring barley 4 %, clover 2 % |

## Stability across seeds

- batchtopk_k16_s1: share of seed-0 features with a closest atom at cosine ≥ 0.9: 0.03; ≥ 0.7: 0.10
- batchtopk_k16_s2: share of seed-0 features with a closest atom at cosine ≥ 0.9: 0.03; ≥ 0.7: 0.12

| class | best SAE feature (seed 0) | closest cosine, seed 1 | seed 2 | test F1 seed 0 / 1 / 2 |
|---|---|---|---|---|
| grassland | f497 | 0.89 | 0.71 | 0.77 / 0.72 / 0.67 |
| winter wheat | f569 | 0.83 | 0.84 | 0.62 / 0.67 / 0.71 |
| green legumes | f131 | 0.61 | 0.63 | 0.36 / 0.41 / 0.43 |
| spring barley | f291 | 0.82 | 0.74 | 0.45 / 0.45 / 0.52 |
| oats | f373 | 0.84 | 0.95 | 0.19 / 0.21 / 0.20 |
| clover | f561 | 0.72 | 0.70 | 0.25 / 0.24 / 0.22 |
| winter rapeseed | f184 | 0.84 | 0.87 | 0.93 / 0.91 / 0.91 |
| spring wheat | f494 | 0.76 | 0.87 | 0.34 / 0.37 / 0.37 |
| peas | f282 | 0.67 | 0.49 | 0.26 / 0.31 / 0.28 |
| potatoes | f256 | 0.72 | 0.75 | 0.28 / 0.19 / 0.32 |
| winter barley | f284 | 0.93 | 0.83 | 0.65 / 0.66 / 0.77 |
| beans | f282 | 0.67 | 0.49 | 0.40 / 0.39 / 0.68 |
| vegetables | f224 | 0.49 | 0.52 | 0.00 / 0.00 / 0.00 |
| rye | f748 | 0.53 | 0.75 | 0.27 / 0.16 / 0.18 |
| alfalfa | f561 | 0.72 | 0.70 | 0.20 / 0.13 / 0.23 |
| spring rapeseed | f661 | 0.45 | 0.56 | 0.26 / 0.35 / 0.19 |
| buckwheat | f844 | 0.75 | 0.84 | 0.54 / 0.55 / 0.56 |
| maize | f769 | 0.92 | 0.89 | 0.74 / 0.65 / 0.75 |
| dry pulses | f912 | 0.40 | 0.46 | 0.07 / 0.00 / 0.01 |
| orchards | f896 | 0.35 | 0.30 | 0.00 / 0.01 / 0.00 |

## Scoring sets

| class | test interior pixels | test fields | validation interior pixels | scored |
|---|---|---|---|---|
| grassland | 5000 | 3177 | 5000 | yes |
| winter wheat | 5000 | 1679 | 5000 | yes |
| green legumes | 5000 | 1495 | 5000 | yes |
| spring barley | 5000 | 1418 | 5000 | yes |
| oats | 5000 | 994 | 5000 | yes |
| clover | 5000 | 928 | 5000 | yes |
| winter rapeseed | 5000 | 946 | 5000 | yes |
| spring wheat | 5000 | 949 | 5000 | yes |
| peas | 5000 | 638 | 5000 | yes |
| potatoes | 5000 | 166 | 5000 | yes |
| winter barley | 5000 | 439 | 5000 | yes |
| beans | 5000 | 340 | 5000 | yes |
| vegetables | 697 | 62 | 627 | yes |
| rye | 5000 | 254 | 5000 | yes |
| alfalfa | 5000 | 268 | 5000 | yes |
| spring rapeseed | 5000 | 225 | 5000 | yes |
| buckwheat | 5000 | 220 | 5000 | yes |
| maize | 5000 | 184 | 5000 | yes |
| dry pulses | 5000 | 106 | 5000 | yes |
| orchards | 1404 | 88 | 615 | yes |

## Where the main SAE's features fire, uniform test pixels (weighted)

- active on at least 1 in 1,000 field-interior pixels: 358
- mainly outside fields (at least 3 times as often as on interiors, and on at least 1 in 1,000 outside pixels): 226
- active on fewer than 1 in 1,000 of all pixels: 553
- (the groups overlap)

| active on this share of all pixels | features | median closest cosine, batchtopk_k16_s1 / batchtopk_k16_s2 | share at cosine ≥ 0.7 |
|---|---|---|---|
| below 0.1 % | 553 | 0.32 / 0.32 | 0.00 / 0.00 |
| 0.1 to 1 % | 214 | 0.44 / 0.42 | 0.13 / 0.15 |
| 1 % or more | 257 | 0.57 / 0.57 | 0.31 / 0.34 |
