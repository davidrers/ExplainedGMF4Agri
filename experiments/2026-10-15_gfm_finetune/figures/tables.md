**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind, frozen | 0.133 +- 0.010 (3) | 0.089 +- 0.012 (3) | 0.077 +- 0.029 (3) |
| TerraMind, fine-tuned | 0.147 +- 0.004 (3) | 0.078 +- 0.017 (3) | 0.072 +- 0.016 (3) |
| Prithvi 600M, frozen | 0.114 +- 0.025 (3) | 0.100 +- 0.022 (3) | 0.100 +- 0.030 (3) |
| Prithvi 600M, fine-tuned | 0.159 +- 0.020 (3) | 0.110 +- 0.006 (3) | 0.116 +- 0.005 (3) |
| THOR, frozen | 0.139 +- 0.025 (3) | 0.099 +- 0.031 (3) | 0.087 +- 0.015 (3) |
| THOR, fine-tuned | 0.134 +- 0.021 (3) | 0.078 +- 0.019 (3) | 0.072 +- 0.006 (3) |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind, frozen | 0.106 +- 0.007 (3) | 0.063 +- 0.011 (3) | 0.054 +- 0.024 (3) |
| TerraMind, fine-tuned | 0.120 +- 0.008 (3) | 0.056 +- 0.013 (3) | 0.049 +- 0.012 (3) |
| Prithvi 600M, frozen | 0.086 +- 0.022 (3) | 0.073 +- 0.022 (3) | 0.069 +- 0.022 (3) |
| Prithvi 600M, fine-tuned | 0.128 +- 0.018 (3) | 0.084 +- 0.008 (3) | 0.082 +- 0.004 (3) |
| THOR, frozen | 0.111 +- 0.020 (3) | 0.073 +- 0.027 (3) | 0.057 +- 0.011 (3) |
| THOR, fine-tuned | 0.107 +- 0.015 (3) | 0.053 +- 0.012 (3) | 0.049 +- 0.003 (3) |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind: fine-tuned minus frozen | +0.014 +- 0.005 (3/3) | -0.010 +- 0.008 (0/3) | -0.005 +- 0.013 (1/3) |
| Prithvi 600M: fine-tuned minus frozen | +0.045 +- 0.025 (3/3) | +0.009 +- 0.022 (2/3) | +0.015 +- 0.025 (2/3) |
| THOR: fine-tuned minus frozen | -0.006 +- 0.005 (0/3) | -0.021 +- 0.018 (0/3) | -0.015 +- 0.010 (0/3) |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind, frozen | 10, 13, 10 / 9, 8, 10 | 12, 8, 8 / 7, 12, 8 | 9, 12, 14 / 7, 12, 6 |
| TerraMind, fine-tuned | 7, 14, 12 / 14, 10, 12 | 6, 5, 13 / 14, 12, 13 | 14, 14, 14 / 10, 9, 6 |
| Prithvi 600M, frozen | 7, 13, 12 / 9, 13, 14 | 11, 7, 10 / 7, 13, 8 | 14, 14, 13 / 12, 8, 11 |
| Prithvi 600M, fine-tuned | 6, 11, 13 / 14, 13, 14 | 9, 5, 9 / 13, 5, 9 | 13, 14, 14 / 14, 12, 13 |
| THOR, frozen | 1, 13, 10 / 14, 14, 11 | 8, 6, 6 / 3, 6, 13 | 12, 10, 13 / 13, 9, 6 |
| THOR, fine-tuned | 4, 14, 11 / 13, 14, 11 | 9, 11, 6 / 9, 10, 13 | 12, 12, 13 / 8, 12, 8 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind, frozen | 0.135 / 0.141 / 0.138 | 0.104 / 0.110 / 0.102 | 0.065 / 0.078 / 0.053 |
| TerraMind, fine-tuned | 0.149 / 0.154 / 0.151 | 0.110 / 0.116 / 0.106 | 0.061 / 0.077 / 0.061 |
| Prithvi 600M, frozen | 0.139 / 0.141 / 0.140 | 0.121 / 0.132 / 0.121 | 0.086 / 0.095 / 0.084 |
| Prithvi 600M, fine-tuned | 0.143 / 0.156 / 0.151 | 0.124 / 0.126 / 0.119 | 0.091 / 0.101 / 0.097 |
| THOR, frozen | 0.121 / 0.140 / 0.138 | 0.104 / 0.116 / 0.102 | 0.082 / 0.092 / 0.083 |
| THOR, fine-tuned | 0.132 / 0.138 / 0.136 | 0.107 / 0.115 / 0.100 | 0.078 / 0.083 / 0.065 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 32 training, 16 validation, 9 test chips.
* K = 100 %: 32 training chips, 697 parcels (draw 0).
* K = 20 %: 29 training chips, 148 parcels (draw 0).
* K = 5 %: 21 training chips, 44 parcels (draw 0).
* 54 fits, 2.0 GPU hours of fitting; machines cluster; commits e12ab2af.

**Fine-tuned arms, test Macro-F1** at the checkpoint of lowest validation loss (the protocol's) / of highest validation Macro-F1 (the secondary readout), mean over fits; the epochs of the second

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind, fine-tuned | 0.147 / 0.139 (epochs 14, 10, 12) | 0.078 / 0.095 (epochs 14, 12, 13) | 0.072 / 0.073 (epochs 10, 9, 6) |
| Prithvi 600M, fine-tuned | 0.159 / 0.153 (epochs 14, 13, 14) | 0.110 / 0.108 (epochs 13, 5, 9) | 0.116 / 0.114 (epochs 14, 12, 13) |
| THOR, fine-tuned | 0.134 / 0.132 (epochs 13, 14, 11) | 0.078 / 0.082 (epochs 9, 10, 13) | 0.072 / 0.083 (epochs 8, 12, 8) |

**Cost**: fitting hours per cell, mean over seeds; the arm's total; peak GPU memory in GB

| Arm | K = 100 % | K = 20 % | K = 5 % | Total h | Peak GB |
|---|---|---|---|---|---|
| TerraMind, frozen | 0.01 | 0.01 | 0.01 | 0.1 | 2.23 |
| TerraMind, fine-tuned | 0.04 | 0.04 | 0.05 | 0.4 | 12.53 |
| Prithvi 600M, frozen | 0.02 | 0.02 | 0.02 | 0.1 | 2.58 |
| Prithvi 600M, fine-tuned | 0.07 | 0.07 | 0.08 | 0.7 | 22.92 |
| THOR, frozen | 0.02 | 0.02 | 0.02 | 0.2 | 3.16 |
| THOR, fine-tuned | 0.05 | 0.06 | 0.06 | 0.5 | 19.29 |

**Readiness checks** (`check.py`): minimum cosine of the fine-tuned model's step-0 features to the core cache; the switch, frozen / trainable; seconds per training chip; peak GB; estimated hours for the nine cells on EE_2021_rev at that speed

| Arm | GPU | Equivalence | Switch | s per chip | Peak GB | Est. h |
|---|---|---|---|---|---|---|
| prithvi_eo_v2_600_tl_ft | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.108 | 23.13 | 17.1 |
| terramind_v1_large_ft | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.049 | 13.23 | 7.9 |
| terramind_v1_large_ft | NVIDIA RTX A4000 | 1.000000 | passed / passed | 0.348 | 12.69 | 55.6 |
| thor_v1_large_ft | NVIDIA RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.076 | 19.6 | 12.3 |
