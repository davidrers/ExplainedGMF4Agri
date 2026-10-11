**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 160 m tokens | 0.365 +- 0.013 (3) | 0.210 +- 0.006 (3) | 0.170 +- 0.016 (3) |
| 80 m tokens | 0.376 +- 0.013 (3) | 0.249 +- 0.023 (3) | 0.175 +- 0.026 (3) |
| 40 m tokens | 0.411 +- 0.035 (3) | 0.262 +- 0.012 (3) | 0.221 +- 0.010 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 160 m tokens | 0.266 +- 0.009 (3) | 0.147 +- 0.005 (3) | 0.114 +- 0.011 (3) |
| 80 m tokens | 0.281 +- 0.007 (3) | 0.182 +- 0.017 (3) | 0.123 +- 0.017 (3) |
| 40 m tokens | 0.310 +- 0.031 (3) | 0.191 +- 0.008 (3) | 0.156 +- 0.005 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 80 m minus 160 m | +0.011 +- 0.027 (2/3) | +0.038 +- 0.019 (3/3) | +0.005 +- 0.038 (2/3) |
| 40 m minus 160 m | +0.046 +- 0.023 (3/3) | +0.051 +- 0.016 (3/3) | +0.050 +- 0.026 (3/3) |
| 40 m minus 80 m | +0.035 +- 0.048 (3/3) | +0.013 +- 0.020 (2/3) | +0.045 +- 0.018 (3/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| 160 m tokens | 9, 12, 8 / 14, 12, 8 | 7, 9, 8 / 14, 12, 13 | 14, 6, 7 / 14, 6, 10 |
| 80 m tokens | 9, 6, 7 / 14, 14, 12 | 7, 10, 8 / 10, 13, 8 | 2, 7, 4 / 10, 14, 12 |
| 40 m tokens | 9, 13, 7 / 14, 14, 13 | 8, 6, 8 / 9, 14, 13 | 8, 13, 6 / 10, 13, 13 |

**Cost**, on the cluster

| Arm | Grid | Per chip encoding | Encoding | Cache | Cache, all Estonia | Fit at K = 100 % | Peak GPU, fit |
|---|---|---|---|---|---|---|---|
| 160 m tokens | 14 x 14 | 39 MB | 5 min | 0.14 TB | 1.6 TB | 3 min | 3.2 GB |
| 80 m tokens | 28 x 28 | 154 MB | 22 min | 0.55 TB | 6.4 TB | 9 min | 4.8 GB |
| 40 m tokens | 56 x 56 | 617 MB | 140 min | 2.22 TB | 25.5 TB | 45 min | 11.3 GB |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 413 training, 88 validation, 207 test chips.
* K = 100 %: 413 training chips, 7,179 parcels (draw 0).
* K = 20 %: 352 training chips, 1,444 parcels (draw 0).
* K = 5 %: 227 training chips, 369 parcels (draw 0).
* 27 fits, 7.2 GPU hours of fitting; machines cluster; commits 0bedeeb4, 11fe4da2.
