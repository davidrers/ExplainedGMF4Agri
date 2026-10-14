**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.581 +- 0.017 (3) | 0.449 +- 0.049 (3) | 0.324 +- 0.036 (3) |
| 4 variants: rotations | 0.602 +- 0.016 (3) | 0.500 +- 0.009 (3) | 0.376 +- 0.027 (3) |
| 8 variants: D4 | 0.628 +- 0.009 (3) | 0.511 +- 0.019 (3) | 0.394 +- 0.021 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.481 +- 0.013 (3) | 0.348 +- 0.038 (3) | 0.245 +- 0.025 (3) |
| 4 variants: rotations | 0.501 +- 0.012 (3) | 0.397 +- 0.009 (3) | 0.285 +- 0.018 (3) |
| 8 variants: D4 | 0.526 +- 0.007 (3) | 0.407 +- 0.021 (3) | 0.297 +- 0.017 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 8 minus 1 variant | +0.047 +- 0.013 (3/3) | +0.061 +- 0.066 (3/3) | +0.070 +- 0.045 (3/3) |
| 4 minus 1 variant | +0.021 +- 0.032 (2/3) | +0.051 +- 0.052 (2/3) | +0.052 +- 0.034 (3/3) |
| 8 minus 4 variants | +0.026 +- 0.021 (3/3) | +0.011 +- 0.021 (2/3) | +0.018 +- 0.048 (1/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 5, 6, 4 / 13, 13, 12 | 3, 2, 6 / 9, 14, 11 | 3, 3, 7 / 12, 13, 14 |
| 4 variants: rotations | 7, 6, 7 / 14, 10, 12 | 9, 5, 6 / 14, 13, 13 | 9, 5, 9 / 14, 14, 14 |
| 8 variants: D4 | 11, 10, 7 / 11, 12, 12 | 6, 11, 7 / 13, 14, 14 | 9, 10, 9 / 13, 13, 14 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 1 variant: identity | 0.581 / 0.626 / 0.611 | 0.446 / 0.523 / 0.515 | 0.327 / 0.404 / 0.395 |
| 4 variants: rotations | 0.601 / 0.635 / 0.629 | 0.497 / 0.532 / 0.528 | 0.371 / 0.433 / 0.433 |
| 8 variants: D4 | 0.629 / 0.641 / 0.632 | 0.505 / 0.538 / 0.531 | 0.394 / 0.421 / 0.416 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 27 fits, 8.2 GPU hours of fitting; machines cluster; commits 53f7783c.
