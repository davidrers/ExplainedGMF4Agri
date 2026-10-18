**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.117 | 0.075 | 0.056 |
| TerraMind S2+S1, fine-tuned | 0.142 | 0.070 | 0.068 |
| Prithvi 600M, frozen | 0.092 | 0.111 | 0.106 |
| Prithvi 600M, fine-tuned | 0.153 | 0.139 | 0.093 |
| THOR S2+S1, frozen | 0.118 | 0.067 | 0.065 |
| THOR S2+S1, fine-tuned | 0.153 | 0.077 | 0.055 |
| TESSERA v2, frozen | 0.209 | 0.192 | 0.157 |
| TESSERA v2, fine-tuned | 0.195 | 0.190 | 0.168 |
| TESSERA v2, frozen, labelled pixels | 0.197 | 0.187 | 0.176 |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.088 | 0.045 | 0.038 |
| TerraMind S2+S1, fine-tuned | 0.111 | 0.044 | 0.042 |
| Prithvi 600M, frozen | 0.069 | 0.082 | 0.074 |
| Prithvi 600M, fine-tuned | 0.122 | 0.107 | 0.063 |
| THOR S2+S1, frozen | 0.093 | 0.043 | 0.040 |
| THOR S2+S1, fine-tuned | 0.130 | 0.050 | 0.039 |
| TESSERA v2, frozen | 0.168 | 0.157 | 0.131 |
| TESSERA v2, fine-tuned | 0.160 | 0.157 | 0.138 |
| TESSERA v2, frozen, labelled pixels | 0.163 | 0.154 | 0.149 |

**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1: fine-tuned minus frozen | +0.025 | -0.005 | +0.012 |
| Prithvi 600M: fine-tuned minus frozen | +0.061 | +0.028 | -0.013 |
| THOR S2+S1: fine-tuned minus frozen | +0.035 | +0.010 | -0.010 |
| TESSERA v2: fine-tuned minus frozen | -0.014 | -0.002 | +0.011 |
| TESSERA v2: fine-tuned minus frozen control | -0.002 | +0.002 | -0.009 |
| TESSERA v2: frozen control minus frozen | -0.012 | -0.005 | +0.020 |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 10 / 13 | 14 / 13 | 10 / 10 |
| TerraMind S2+S1, fine-tuned | 9 / 10 | 10 / 10 | 10 / 12 |
| Prithvi 600M, frozen | 7 / 7 | 11 / 14 | 14 / 14 |
| Prithvi 600M, fine-tuned | 9 / 14 | 11 / 11 | 13 / 14 |
| THOR S2+S1, frozen | 5 / 9 | 13 / 13 | 12 / 14 |
| THOR S2+S1, fine-tuned | 12 / 14 | 14 / 14 | 14 / 14 |
| TESSERA v2, frozen | 9 / 9 | 9 / 10 | 7 / 8 |
| TESSERA v2, fine-tuned | 7 / 14 | 5 / 9 | 5 / 10 |
| TESSERA v2, frozen, labelled pixels | 8 / 12 | 6 / 8 | 5 / 10 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.133 / 0.136 / 0.125 | 0.111 / 0.118 / 0.111 | 0.074 / 0.074 / 0.055 |
| TerraMind S2+S1, fine-tuned | 0.136 / 0.143 / 0.127 | 0.111 / 0.111 / 0.088 | 0.050 / 0.051 / 0.044 |
| Prithvi 600M, frozen | 0.133 / 0.133 / 0.128 | 0.114 / 0.117 / 0.117 | 0.079 / 0.079 / 0.079 |
| Prithvi 600M, fine-tuned | 0.141 / 0.153 / 0.153 | 0.129 / 0.129 / 0.114 | 0.077 / 0.089 / 0.089 |
| THOR S2+S1, frozen | 0.140 / 0.151 / 0.140 | 0.145 / 0.145 / 0.125 | 0.076 / 0.077 / 0.077 |
| THOR S2+S1, fine-tuned | 0.156 / 0.159 / 0.159 | 0.135 / 0.135 / 0.135 | 0.095 / 0.095 / 0.095 |
| TESSERA v2, frozen | 0.298 / 0.298 / 0.281 | 0.212 / 0.216 / 0.205 | 0.195 / 0.209 / 0.189 |
| TESSERA v2, fine-tuned | 0.258 / 0.271 / 0.271 | 0.201 / 0.219 / 0.195 | 0.174 / 0.187 / 0.171 |
| TESSERA v2, frozen, labelled pixels | 0.266 / 0.298 / 0.272 | 0.189 / 0.212 / 0.198 | 0.169 / 0.222 / 0.205 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 32 training, 16 validation, 9 test chips.
* K = 100 %: 32 training chips, 697 parcels (draw 0).
* K = 20 %: 29 training chips, 148 parcels (draw 0).
* K = 5 %: 21 training chips, 44 parcels (draw 0).
* 27 fits, 1.3 GPU hours of fitting; machines cluster, hub; commits 4ab89e44, 804dd161.

**This folder's arms, test Macro-F1** at the checkpoint of lowest validation loss (the protocol's) / of highest validation Macro-F1 (the secondary readout), mean over fits; the epochs of the second

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, fine-tuned | 0.142 / 0.121 (epochs 10) | 0.070 / 0.070 (epochs 10) | 0.068 / 0.045 (epochs 12) |
| Prithvi 600M, fine-tuned | 0.153 / 0.133 (epochs 14) | 0.139 / 0.139 (epochs 11) | 0.093 / 0.087 (epochs 14) |
| THOR S2+S1, fine-tuned | 0.153 / 0.137 (epochs 14) | 0.077 / 0.077 (epochs 14) | 0.055 / 0.055 (epochs 14) |
| TESSERA v2, fine-tuned | 0.195 / 0.204 (epochs 14) | 0.190 / 0.205 (epochs 9) | 0.168 / 0.187 (epochs 10) |
| TESSERA v2, frozen, labelled pixels | 0.197 / 0.180 (epochs 12) | 0.187 / 0.187 (epochs 8) | 0.176 / 0.182 (epochs 10) |

**Cost**: fitting hours per cell, mean over seeds; the arm's total; peak GPU memory in GB

| Arm | K = 100 % | K = 20 % | K = 5 % | Total h | Peak GB |
|---|---|---|---|---|---|
| TerraMind S2+S1, frozen | 0.07 | 0.05 | 0.04 | 0.2 | 2.08 |
| TerraMind S2+S1, fine-tuned | 0.05 | 0.05 | 0.04 | 0.1 | 18.52 |
| Prithvi 600M, frozen | 0.07 | 0.06 | 0.05 | 0.2 | 2.4 |
| Prithvi 600M, fine-tuned | 0.06 | 0.08 | 0.09 | 0.2 | 22.92 |
| THOR S2+S1, frozen | 0.05 | 0.05 | 0.04 | 0.1 | 2.01 |
| THOR S2+S1, fine-tuned | 0.07 | 0.06 | 0.06 | 0.2 | 26.51 |
| TESSERA v2, frozen | 0.04 | 0.04 | 0.04 | 0.1 | 4.12 |
| TESSERA v2, fine-tuned | 0.10 | 0.03 | 0.02 | 0.1 | 8.91 |
| TESSERA v2, frozen, labelled pixels | 0.00 | 0.00 | 0.00 | 0.0 | 1.36 |

**Readiness checks** (`check.py`): minimum cosine of the step-0 features (token grid, at the bottleneck input) or embeddings (TESSERA, per labelled pixel) to the core's cache; the switch, frozen / trainable; training speed; peak GB; estimated hours for the arm's three cells on EE_2021_rev at that speed

| Arm | GPU | Equivalence | Switch | Speed | Peak GB | Est. h |
|---|---|---|---|---|---|---|
| prithvi_eo_v2_600_tl_ft | RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.108 s per chip | 23.13 | 5.7 |
| terramind_v1_large_s2s1_ft | RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.064 s per chip | 18.69 | 3.5 |
| tessera_v2_medium_ft | RTX PRO 6000 Blackwell Server Edition | 0.999976 | - / passed | 36.1 us per labelled pixel | 8.81 | 10.9 |
| tessera_v2_medium_ft | RTX A4000 | - | - / passed | 144.1 us per labelled pixel | 8.81 | 43.6 |
| tessera_v2_medium_lab | RTX A4000 | - | passed / - | 1.0 us per labelled pixel | 0.58 | - |
| thor_v1_large_s2s1_ft | RTX PRO 6000 Blackwell Server Edition | 1.000000 | passed / passed | 0.113 s per chip | 26.66 | 6.1 |
