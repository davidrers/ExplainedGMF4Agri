**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.498 +- 0.018 (3) | 0.415 +- 0.015 (3) | 0.342 +- 0.009 (3) |
| U-Net, 45 epochs | 0.544 +- 0.015 (3) | 0.477 +- 0.002 (3) | 0.413 +- 0.032 (3) |
| TESSERA + MLP | 0.625 +- 0.004 (3) | 0.597 +- 0.003 (3) | 0.570 +- 0.008 (3) |
| TerraMind v1 large | 0.628 +- 0.009 (3) | 0.511 +- 0.019 (3) | 0.394 +- 0.021 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.394 +- 0.018 (3) | 0.319 +- 0.010 (3) | 0.257 +- 0.007 (3) |
| U-Net, 45 epochs | 0.437 +- 0.014 (3) | 0.372 +- 0.002 (3) | 0.313 +- 0.023 (3) |
| TESSERA + MLP | 0.524 +- 0.002 (3) | 0.491 +- 0.003 (3) | 0.458 +- 0.007 (3) |
| TerraMind v1 large | 0.526 +- 0.007 (3) | 0.407 +- 0.021 (3) | 0.297 +- 0.017 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 45 minus 15 epochs | +0.046 +- 0.033 (3/3) | +0.062 +- 0.015 (3/3) | +0.071 +- 0.040 (3/3) |
| U-Net 15 ep. minus TerraMind | -0.130 +- 0.017 (0/3) | -0.095 +- 0.030 (0/3) | -0.052 +- 0.013 (0/3) |
| U-Net 45 ep. minus TerraMind | -0.084 +- 0.022 (0/3) | -0.034 +- 0.020 (0/3) | +0.019 +- 0.053 (2/3) |
| U-Net 15 ep. minus TESSERA | -0.127 +- 0.014 (0/3) | -0.182 +- 0.012 (0/3) | -0.228 +- 0.017 (0/3) |
| U-Net 45 ep. minus TESSERA | -0.081 +- 0.020 (0/3) | -0.120 +- 0.003 (0/3) | -0.157 +- 0.024 (0/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 12, 13, 12 / 14, 13, 12 | 12, 14, 14 / 12, 14, 12 | 14, 14, 14 / 14, 14, 13 |
| U-Net, 45 epochs | 43, 26, 36 / 39, 39, 36 | 38, 38, 35 / 38, 44, 35 | 36, 27, 44 / 38, 44, 44 |
| TESSERA + MLP | 12, 9, 14 / 9, 9, 14 | 11, 14, 13 / 11, 13, 13 | 13, 10, 14 / 13, 10, 14 |
| TerraMind v1 large | 11, 10, 7 / 11, 12, 12 | 6, 11, 7 / 13, 14, 14 | 9, 10, 9 / 13, 13, 14 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.502 / 0.504 / 0.491 | 0.417 / 0.419 / 0.416 | 0.341 / 0.348 / 0.341 |
| U-Net, 45 epochs | 0.543 / 0.555 / 0.537 | 0.479 / 0.482 / 0.458 | 0.411 / 0.426 / 0.411 |
| TESSERA + MLP | 0.626 / 0.627 / 0.617 | 0.595 / 0.595 / 0.577 | 0.570 / 0.570 / 0.555 |
| TerraMind v1 large | 0.629 / 0.641 / 0.632 | 0.505 / 0.538 / 0.531 | 0.394 / 0.421 / 0.416 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 36 fits, 16.3 GPU hours of fitting; machines cluster; commits 53f7783c, ac3abbce, bd4cd1fc.
