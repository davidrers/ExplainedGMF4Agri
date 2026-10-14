**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| S2 | 0.628 +- 0.009 (3) | 0.511 +- 0.019 (3) | 0.394 +- 0.021 (3) |
| S1 | 0.557 +- 0.016 (3) | 0.482 +- 0.003 (3) | 0.370 +- 0.024 (3) |
| S2 + S1 | 0.654 +- 0.006 (3) | 0.565 +- 0.018 (3) | 0.419 +- 0.022 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| S2 | 0.526 +- 0.007 (3) | 0.407 +- 0.021 (3) | 0.297 +- 0.017 (3) |
| S1 | 0.448 +- 0.015 (3) | 0.376 +- 0.003 (3) | 0.275 +- 0.019 (3) |
| S2 + S1 | 0.560 +- 0.009 (3) | 0.460 +- 0.015 (3) | 0.324 +- 0.018 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| S2 + S1 minus S2 | +0.026 +- 0.006 (3/3) | +0.054 +- 0.036 (3/3) | +0.025 +- 0.028 (3/3) |
| S1 minus S2 | -0.072 +- 0.016 (0/3) | -0.029 +- 0.021 (0/3) | -0.024 +- 0.005 (0/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| S2 | 11, 10, 7 / 11, 12, 12 | 6, 11, 7 / 13, 14, 14 | 9, 10, 9 / 13, 13, 14 |
| S1 | 14, 10, 13 / 14, 13, 13 | 13, 12, 14 / 13, 12, 14 | 11, 14, 12 / 14, 14, 14 |
| S2 + S1 | 13, 11, 9 / 13, 14, 11 | 12, 8, 14 / 14, 14, 14 | 12, 10, 9 / 14, 14, 14 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| S2 | 0.629 / 0.641 / 0.632 | 0.505 / 0.538 / 0.531 | 0.394 / 0.421 / 0.416 |
| S1 | 0.552 / 0.557 / 0.553 | 0.479 / 0.479 / 0.474 | 0.368 / 0.374 / 0.374 |
| S2 + S1 | 0.655 / 0.662 / 0.653 | 0.566 / 0.577 / 0.577 | 0.414 / 0.438 / 0.438 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 27 fits, 6.6 GPU hours of fitting; machines cluster; commits ac3abbce, bd4cd1fc.
