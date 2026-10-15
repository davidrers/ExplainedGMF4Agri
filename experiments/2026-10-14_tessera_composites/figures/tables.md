**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 0.630 +- 0.006 (3) | 0.594 +- 0.002 (3) | 0.553 +- 0.003 (3) |
| v2 Medium, fine-tuned | 0.667 +- 0.007 (3) | 0.641 +- 0.004 (3) | 0.589 +- 0.002 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 0.522 +- 0.006 (3) | 0.484 +- 0.003 (3) | 0.441 +- 0.003 (3) |
| v2 Medium, fine-tuned | 0.569 +- 0.007 (3) | 0.538 +- 0.005 (3) | 0.481 +- 0.001 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| fine-tuned minus frozen | +0.037 +- 0.007 (3/3) | +0.047 +- 0.003 (3/3) | +0.037 +- 0.005 (3/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 5, 1, 8 / 9, 14, 6 | 1, 0, 1 / 4, 14, 3 | 1, 1, 1 / 5, 7, 9 |
| v2 Medium, fine-tuned | 0, 0, 1 / 5, 3, 3 | 1, 1, 1 / 8, 3, 3 | 1, 1, 1 / 3, 8, 11 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 0.585 / 0.588 / 0.585 | 0.549 / 0.558 / 0.552 | 0.519 / 0.533 / 0.528 |
| v2 Medium, fine-tuned | 0.621 / 0.639 / 0.632 | 0.593 / 0.603 / 0.597 | 0.554 / 0.574 / 0.569 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 18 fits, 3.9 GPU hours of fitting; machines cluster; commits fb8654fa.

**Test Macro-F1 beside other arms on the same chip set, draw and seeds**, mean +- sd (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 0.630 +- 0.006 (3) | 0.594 +- 0.002 (3) | 0.553 +- 0.003 (3) |
| v2 Medium, fine-tuned | 0.667 +- 0.007 (3) | 0.641 +- 0.004 (3) | 0.589 +- 0.002 (3) |
| TESSERA v1 store (current arm) | 0.625 +- 0.004 (3) | 0.597 +- 0.003 (3) | 0.570 +- 0.008 (3) |
| TerraMind v1 large, S2 + S1 | 0.654 +- 0.006 (3) | 0.565 +- 0.018 (3) | 0.419 +- 0.022 (3) |
| U-Net from scratch, 45 epochs | 0.544 +- 0.015 (3) | 0.477 +- 0.002 (3) | 0.413 +- 0.032 (3) |
