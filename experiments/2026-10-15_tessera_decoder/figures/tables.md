**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.573 +- 0.005 (3) | 0.532 +- 0.008 (3) | 0.451 +- 0.029 (3) |
| U-Net, 15 epochs | 0.516 +- 0.025 (3) | 0.444 +- 0.021 (3) | 0.259 +- 0.014 (3) |
| U-Net, 45 epochs | 0.589 +- 0.013 (3) | 0.530 +- 0.009 (3) | 0.358 +- 0.025 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.467 +- 0.006 (3) | 0.419 +- 0.008 (3) | 0.346 +- 0.025 (3) |
| U-Net, 15 epochs | 0.424 +- 0.022 (3) | 0.348 +- 0.024 (3) | 0.207 +- 0.011 (3) |
| U-Net, 45 epochs | 0.482 +- 0.010 (3) | 0.421 +- 0.007 (3) | 0.277 +- 0.019 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net minus MLP, both 15 epochs | -0.057 +- 0.027 (0/3) | -0.088 +- 0.013 (0/3) | -0.192 +- 0.041 (0/3) |
| U-Net 45 epochs minus MLP | +0.016 +- 0.009 (3/3) | -0.002 +- 0.009 (1/3) | -0.093 +- 0.004 (0/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 12, 9, 13 / 12, 13, 10 | 12, 6, 6 / 7, 6, 12 | 12, 5, 4 / 11, 10, 12 |
| U-Net, 15 epochs | 13, 10, 14 / 10, 13, 14 | 14, 13, 13 / 14, 13, 14 | 13, 9, 13 / 14, 14, 12 |
| U-Net, 45 epochs | 34, 36, 25 / 34, 25, 39 | 34, 33, 39 / 36, 44, 43 | 24, 23, 19 / 44, 42, 38 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.567 / 0.569 / 0.556 | 0.534 / 0.543 / 0.512 | 0.445 / 0.468 / 0.440 |
| U-Net, 15 epochs | 0.518 / 0.527 / 0.520 | 0.460 / 0.466 / 0.465 | 0.276 / 0.297 / 0.295 |
| U-Net, 45 epochs | 0.581 / 0.585 / 0.565 | 0.532 / 0.543 / 0.515 | 0.361 / 0.422 / 0.403 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 1190 training, 169 validation, 411 test chips.
* K = 100 %: 1190 training chips, 20,963 parcels (draw 0).
* K = 20 %: 1018 training chips, 4,202 parcels (draw 0).
* K = 5 %: 641 training chips, 1,058 parcels (draw 0).
* 27 fits, 4.1 GPU hours of fitting; machines cluster; commits 0ab085d6.

**Test Macro-F1 on parcel interiors**, mean +- standard deviation over seeds

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.596 +- 0.006 | 0.553 +- 0.009 | 0.469 +- 0.029 |
| U-Net, 15 epochs | 0.532 +- 0.025 | 0.459 +- 0.021 | 0.264 +- 0.014 |
| U-Net, 45 epochs | 0.608 +- 0.012 | 0.548 +- 0.011 | 0.368 +- 0.026 |

**Test Macro-F1 on parcel edges**, mean +- standard deviation over seeds

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.465 +- 0.003 | 0.428 +- 0.011 | 0.356 +- 0.033 |
| U-Net, 15 epochs | 0.438 +- 0.027 | 0.371 +- 0.024 | 0.232 +- 0.014 |
| U-Net, 45 epochs | 0.499 +- 0.015 | 0.441 +- 0.006 | 0.302 +- 0.019 |

**Share of a test parcel's pixels in its majority predicted class**, mean +- standard deviation over seeds

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| per-pixel MLP | 0.885 +- 0.005 | 0.872 +- 0.008 | 0.865 +- 0.017 |
| U-Net, 15 epochs | 0.907 +- 0.003 | 0.899 +- 0.003 | 0.907 +- 0.010 |
| U-Net, 45 epochs | 0.910 +- 0.004 | 0.898 +- 0.005 | 0.889 +- 0.007 |

Edge pixels are 21.2% of the labelled test pixels.
