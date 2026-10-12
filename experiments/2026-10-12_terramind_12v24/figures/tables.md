**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 12 monthly frames | 0.419 +- 0.011 (3) | 0.247 +- 0.031 (3) | 0.159 +- 0.019 (3) |
| 24 half-month frames | 0.434 +- 0.024 (3) | 0.282 +- 0.036 (3) | 0.160 +- 0.021 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 12 monthly frames | 0.319 +- 0.010 (3) | 0.178 +- 0.022 (3) | 0.110 +- 0.013 (3) |
| 24 half-month frames | 0.336 +- 0.021 (3) | 0.203 +- 0.033 (3) | 0.109 +- 0.013 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 24 minus 12 frames | +0.015 +- 0.018 (2/3) | +0.035 +- 0.048 (2/3) | +0.001 +- 0.018 (1/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 12 monthly frames | 7, 12, 12 / 14, 13, 13 | 7, 9, 6 / 14, 13, 13 | 5, 7, 3 / 7, 7, 11 |
| 24 half-month frames | 8, 14, 10 / 13, 14, 10 | 10, 12, 8 / 14, 14, 14 | 5, 5, 3 / 12, 14, 14 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 12 monthly frames | 0.389 / 0.422 / 0.397 | 0.237 / 0.286 / 0.281 | 0.173 / 0.198 / 0.175 |
| 24 half-month frames | 0.420 / 0.438 / 0.432 | 0.271 / 0.300 / 0.300 | 0.166 / 0.199 / 0.185 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 413 training, 88 validation, 207 test chips.
* K = 100 %: 413 training chips, 7,179 parcels (draw 0).
* K = 20 %: 352 training chips, 1,444 parcels (draw 0).
* K = 5 %: 227 training chips, 369 parcels (draw 0).
* 18 fits, 0.7 GPU hours of fitting; machines cluster; commits d120f692.
