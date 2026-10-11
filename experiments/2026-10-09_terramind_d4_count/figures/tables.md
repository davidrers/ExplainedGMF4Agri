**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.518 +- 0.010 (3) | 0.368 +- 0.025 (3) | 0.258 +- 0.033 (3) |
| 4 variants: rotations | 0.543 +- 0.006 (3) | 0.407 +- 0.038 (3) | 0.282 +- 0.045 (3) |
| 8 variants: D4 | 0.551 +- 0.010 (3) | 0.438 +- 0.014 (3) | 0.278 +- 0.037 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.412 +- 0.011 (3) | 0.275 +- 0.019 (3) | 0.187 +- 0.022 (3) |
| 4 variants: rotations | 0.437 +- 0.007 (3) | 0.311 +- 0.033 (3) | 0.206 +- 0.030 (3) |
| 8 variants: D4 | 0.444 +- 0.008 (3) | 0.336 +- 0.011 (3) | 0.205 +- 0.026 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 8 minus 1 variant | +0.033 +- 0.010 (3/3) | +0.070 +- 0.012 (3/3) | +0.020 +- 0.070 (1/3) |
| 4 minus 1 variant | +0.025 +- 0.011 (3/3) | +0.040 +- 0.061 (2/3) | +0.024 +- 0.047 (2/3) |
| 8 minus 4 variants | +0.008 +- 0.006 (3/3) | +0.031 +- 0.051 (2/3) | -0.004 +- 0.061 (2/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 4, 6, 4 / 13, 11, 12 | 5, 3, 4 / 13, 12, 12 | 5, 7, 4 / 11, 9, 13 |
| 4 variants: rotations | 7, 7, 7 / 13, 13, 13 | 5, 10, 6 / 13, 14, 14 | 5, 11, 8 / 13, 11, 13 |
| 8 variants: D4 | 7, 7, 7 / 13, 11, 13 | 10, 9, 8 / 10, 14, 14 | 6, 5, 11 / 11, 14, 13 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.512 / 0.575 / 0.542 | 0.365 / 0.446 / 0.420 | 0.261 / 0.308 / 0.293 |
| 4 variants: rotations | 0.543 / 0.592 / 0.575 | 0.400 / 0.464 / 0.461 | 0.287 / 0.334 / 0.330 |
| 8 variants: D4 | 0.550 / 0.585 / 0.565 | 0.432 / 0.463 / 0.462 | 0.278 / 0.330 / 0.319 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 2388 training, 769 validation, 1475 test chips.
* K = 100 %: 2388 training chips, 42,700 parcels (draw 0).
* K = 20 %: 2020 training chips, 8,547 parcels (draw 0).
* K = 5 %: 1303 training chips, 2,142 parcels (draw 0).
* 27 fits, 3.7 GPU hours of fitting; machines cluster; commits 0bedeeb4, 389cf61b.
