**Panoptic quality of the boundary parcels** (threshold 0.5), class-aware

| Arm | PQ, K = 100 % | SQ, K = 100 % | RQ, K = 100 % | PQ, K = 20 % | SQ, K = 20 % | RQ, K = 20 % | PQ, K = 5 % | SQ, K = 5 % | RQ, K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.256 | 0.756 | 0.288 | 0.220 | 0.755 | 0.249 | 0.160 | 0.712 | 0.181 |
| Prithvi 600M | 0.269 | 0.757 | 0.303 | 0.210 | 0.711 | 0.237 | 0.147 | 0.673 | 0.165 |
| THOR S2+S1 | 0.209 | 0.750 | 0.237 | 0.181 | 0.712 | 0.204 | 0.134 | 0.665 | 0.151 |
| TESSERA v2 | 0.307 | 0.813 | 0.340 | 0.287 | 0.766 | 0.318 | 0.263 | 0.772 | 0.292 |
| AlphaEarth | 0.210 | 0.762 | 0.236 | 0.205 | 0.759 | 0.230 | 0.182 | 0.749 | 0.206 |
| U-Net baseline | 0.338 | 0.777 | 0.371 | 0.290 | 0.727 | 0.320 | 0.234 | 0.668 | 0.262 |

**Class-agnostic PQ**, the geometry alone: the extent alone (floor), crop-type map, boundary head, both

| Arm | floor, K = 100 % | map, K = 100 % | head, K = 100 % | both, K = 100 % | floor, K = 20 % | map, K = 20 % | head, K = 20 % | both, K = 20 % | floor, K = 5 % | map, K = 5 % | head, K = 5 % | both, K = 5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.486 | 0.386 | 0.487 | 0.425 | 0.486 | 0.358 | 0.486 | 0.401 | 0.486 | 0.331 | 0.467 | 0.344 |
| Prithvi 600M | 0.486 | 0.370 | 0.512 | 0.420 | 0.486 | 0.344 | 0.484 | 0.372 | 0.486 | 0.301 | 0.469 | 0.311 |
| THOR S2+S1 | 0.486 | 0.367 | 0.466 | 0.377 | 0.486 | 0.349 | 0.471 | 0.354 | 0.486 | 0.320 | 0.478 | 0.317 |
| TESSERA v2 | 0.486 | 0.085 | 0.476 | 0.109 | 0.486 | 0.081 | 0.476 | 0.102 | 0.486 | 0.074 | 0.462 | 0.092 |
| AlphaEarth | 0.486 | 0.129 | 0.470 | 0.154 | 0.486 | 0.109 | 0.467 | 0.133 | 0.486 | 0.118 | 0.467 | 0.147 |
| U-Net baseline | 0.486 | 0.305 | 0.561 | 0.375 | 0.486 | 0.262 | 0.530 | 0.325 | 0.486 | 0.239 | 0.524 | 0.295 |

**Parcel Macro-F1 at IoU > 0.5 (RQ)**: true parcels, boundary parcels, the difference

| Arm | true, K = 100 % | pred., K = 100 % | diff., K = 100 % | true, K = 20 % | pred., K = 20 % | diff., K = 20 % | true, K = 5 % | pred., K = 5 % | diff., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.503 | 0.288 | -0.214 | 0.438 | 0.249 | -0.190 | 0.337 | 0.181 | -0.156 |
| Prithvi 600M | 0.482 | 0.303 | -0.179 | 0.431 | 0.237 | -0.194 | 0.314 | 0.165 | -0.149 |
| THOR S2+S1 | 0.456 | 0.237 | -0.219 | 0.390 | 0.204 | -0.186 | 0.283 | 0.151 | -0.132 |
| TESSERA v2 | 0.551 | 0.340 | -0.211 | 0.524 | 0.318 | -0.205 | 0.490 | 0.292 | -0.198 |
| AlphaEarth | 0.428 | 0.236 | -0.193 | 0.412 | 0.230 | -0.182 | 0.381 | 0.206 | -0.176 |
| U-Net baseline | 0.493 | 0.371 | -0.122 | 0.446 | 0.320 | -0.126 | 0.388 | 0.262 | -0.126 |

**Boundary pixels**: test F1 of the boundary head at 0.5, against the dense target

| Arm | F1, K = 100 % | F1, K = 20 % | F1, K = 5 % |
|---|---|---|---|
| TerraMind S2+S1 | 0.497 | 0.449 | 0.300 |
| Prithvi 600M | 0.559 | 0.406 | 0.269 |
| THOR S2+S1 | 0.291 | 0.147 | 0.038 |
| TESSERA v2 | 0.639 | 0.614 | 0.605 |
| AlphaEarth | 0.551 | 0.577 | 0.558 |
| U-Net baseline | 0.657 | 0.637 | 0.571 |

**Threshold**: class-agnostic PQ of the boundary parcels at 0.3, 0.5, 0.7

| Arm | 0.3, K = 100 % | 0.5, K = 100 % | 0.7, K = 100 % | 0.3, K = 20 % | 0.5, K = 20 % | 0.7, K = 20 % | 0.3, K = 5 % | 0.5, K = 5 % | 0.7, K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.535 | 0.487 | 0.472 | 0.534 | 0.486 | 0.471 | 0.512 | 0.467 | 0.481 |
| Prithvi 600M | 0.565 | 0.512 | 0.482 | 0.537 | 0.484 | 0.477 | 0.510 | 0.469 | 0.483 |
| THOR S2+S1 | 0.499 | 0.466 | 0.478 | 0.495 | 0.471 | 0.485 | 0.478 | 0.478 | 0.486 |
| TESSERA v2 | 0.466 | 0.476 | 0.466 | 0.469 | 0.476 | 0.465 | 0.445 | 0.462 | 0.459 |
| AlphaEarth | 0.484 | 0.470 | 0.463 | 0.480 | 0.467 | 0.459 | 0.474 | 0.467 | 0.462 |
| U-Net baseline | 0.570 | 0.561 | 0.531 | 0.541 | 0.530 | 0.512 | 0.535 | 0.524 | 0.501 |

**Joint against crop-type-only training** (main workflow, same draw and seed): pixel Macro-F1 crop-type only, joint, difference; parcel Macro-F1 of D28, difference

| Arm | pixel, ref., K = 100 % | pixel, joint, K = 100 % | diff., K = 100 % | D28 diff., K = 100 % | pixel, ref., K = 20 % | pixel, joint, K = 20 % | diff., K = 20 % | D28 diff., K = 20 % | pixel, ref., K = 5 % | pixel, joint, K = 5 % | diff., K = 5 % | D28 diff., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.661 | 0.664 | 0.002 | -0.002 | 0.566 | 0.578 | 0.012 | 0.001 | 0.433 | 0.440 | 0.006 | 0.014 |
| Prithvi 600M | 0.580 | 0.622 | 0.042 | 0.016 | 0.505 | 0.532 | 0.026 | 0.020 | 0.445 | 0.400 | -0.044 | -0.043 |
| THOR S2+S1 | 0.594 | 0.614 | 0.020 | 0.020 | 0.547 | 0.513 | -0.035 | -0.030 | 0.393 | 0.371 | -0.021 | -0.016 |
| TESSERA v2 | 0.618 | 0.618 | -0.000 | 0.003 | 0.582 | 0.585 | 0.003 | 0.000 | 0.504 | 0.548 | 0.044 | 0.035 |
| AlphaEarth | 0.523 | 0.525 | 0.002 | -0.009 | 0.497 | 0.504 | 0.008 | 0.006 | 0.449 | 0.444 | -0.006 | -0.001 |
| U-Net baseline | 0.616 | 0.619 | 0.004 | 0.009 | 0.564 | 0.541 | -0.023 | -0.011 | 0.467 | 0.473 | 0.006 | -0.003 |

**The crop-type-only fits' own parcels**: class-agnostic PQ of their crop-type map, and RQ with the true parcels

| Arm | PQ agn., map, K = 100 % | RQ, true, K = 100 % | PQ agn., map, K = 20 % | RQ, true, K = 20 % | PQ agn., map, K = 5 % | RQ, true, K = 5 % |
|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.387 | 0.506 | 0.354 | 0.439 | 0.322 | 0.322 |
| Prithvi 600M | 0.364 | 0.467 | 0.330 | 0.411 | 0.301 | 0.352 |
| THOR S2+S1 | 0.369 | 0.441 | 0.356 | 0.418 | 0.325 | 0.299 |
| TESSERA v2 | 0.088 | 0.551 | 0.081 | 0.522 | 0.072 | 0.452 |
| AlphaEarth | 0.115 | 0.435 | 0.114 | 0.409 | 0.104 | 0.382 |
| U-Net baseline | 0.304 | 0.483 | 0.280 | 0.452 | 0.232 | 0.390 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips; 55,575 true parcel pieces in the test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 18 fits, 17.3 GPU hours of fitting; machines cluster; commits faec7eac.

**True parcels found by the extent alone**, class-agnostic IoU > 0.5, by what they touch

| Arm | isolated, K = 100 % | other cl., K = 100 % | same cl., K = 100 % | isolated, K = 20 % | other cl., K = 20 % | same cl., K = 20 % | isolated, K = 5 % | other cl., K = 5 % | same cl., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |
| Prithvi 600M | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |
| THOR S2+S1 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |
| TESSERA v2 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |
| AlphaEarth | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |
| U-Net baseline | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 | 1.000 | 0.126 | 0.127 |

**True parcels found by the crop-type map alone**, class-agnostic IoU > 0.5, by what they touch

| Arm | isolated, K = 100 % | other cl., K = 100 % | same cl., K = 100 % | isolated, K = 20 % | other cl., K = 20 % | same cl., K = 20 % | isolated, K = 5 % | other cl., K = 5 % | same cl., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.983 | 0.472 | 0.224 | 0.979 | 0.455 | 0.227 | 0.977 | 0.412 | 0.213 |
| Prithvi 600M | 0.981 | 0.474 | 0.234 | 0.979 | 0.440 | 0.217 | 0.968 | 0.413 | 0.218 |
| THOR S2+S1 | 0.979 | 0.432 | 0.219 | 0.980 | 0.402 | 0.214 | 0.972 | 0.350 | 0.204 |
| TESSERA v2 | 0.933 | 0.540 | 0.242 | 0.932 | 0.530 | 0.227 | 0.920 | 0.518 | 0.232 |
| AlphaEarth | 0.953 | 0.482 | 0.215 | 0.942 | 0.473 | 0.223 | 0.946 | 0.455 | 0.210 |
| U-Net baseline | 0.974 | 0.499 | 0.226 | 0.971 | 0.497 | 0.228 | 0.961 | 0.468 | 0.220 |

**True parcels found by the boundary head at 0.5**, class-agnostic IoU > 0.5, by what they touch

| Arm | isolated, K = 100 % | other cl., K = 100 % | same cl., K = 100 % | isolated, K = 20 % | other cl., K = 20 % | same cl., K = 20 % | isolated, K = 5 % | other cl., K = 5 % | same cl., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.993 | 0.233 | 0.195 | 0.995 | 0.234 | 0.191 | 0.996 | 0.174 | 0.162 |
| Prithvi 600M | 0.993 | 0.291 | 0.238 | 0.994 | 0.226 | 0.188 | 0.995 | 0.173 | 0.160 |
| THOR S2+S1 | 0.995 | 0.160 | 0.157 | 0.997 | 0.149 | 0.141 | 0.999 | 0.130 | 0.131 |
| TESSERA v2 | 0.979 | 0.339 | 0.280 | 0.981 | 0.318 | 0.264 | 0.979 | 0.303 | 0.259 |
| AlphaEarth | 0.988 | 0.221 | 0.212 | 0.985 | 0.226 | 0.223 | 0.986 | 0.223 | 0.216 |
| U-Net baseline | 0.987 | 0.448 | 0.299 | 0.984 | 0.408 | 0.285 | 0.991 | 0.345 | 0.240 |

**True parcels found by the boundary head at 0.3**, class-agnostic IoU > 0.5, by what they touch

| Arm | isolated, K = 100 % | other cl., K = 100 % | same cl., K = 100 % | isolated, K = 20 % | other cl., K = 20 % | same cl., K = 20 % | isolated, K = 5 % | other cl., K = 5 % | same cl., K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.991 | 0.364 | 0.273 | 0.994 | 0.353 | 0.259 | 0.996 | 0.284 | 0.232 |
| Prithvi 600M | 0.991 | 0.429 | 0.334 | 0.993 | 0.360 | 0.280 | 0.996 | 0.274 | 0.222 |
| THOR S2+S1 | 0.995 | 0.263 | 0.207 | 0.996 | 0.248 | 0.189 | 0.996 | 0.190 | 0.175 |
| TESSERA v2 | 0.975 | 0.398 | 0.326 | 0.976 | 0.396 | 0.320 | 0.973 | 0.379 | 0.310 |
| AlphaEarth | 0.985 | 0.296 | 0.265 | 0.982 | 0.307 | 0.276 | 0.983 | 0.308 | 0.275 |
| U-Net baseline | 0.983 | 0.494 | 0.353 | 0.980 | 0.451 | 0.330 | 0.985 | 0.410 | 0.296 |

**Boundary pixels marked at 0.5**, by what lies across them (interior: false alarms)

| Arm | same class, K = 100 % | between classes, K = 100 % | outer, K = 100 % | interior, K = 100 % | same class, K = 20 % | between classes, K = 20 % | outer, K = 20 % | interior, K = 20 % | same class, K = 5 % | between classes, K = 5 % | outer, K = 5 % | interior, K = 5 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 0.100 | 0.361 | 0.444 | 0.036 | 0.087 | 0.353 | 0.395 | 0.041 | 0.067 | 0.136 | 0.243 | 0.035 |
| Prithvi 600M | 0.154 | 0.508 | 0.512 | 0.040 | 0.086 | 0.313 | 0.329 | 0.032 | 0.054 | 0.137 | 0.206 | 0.030 |
| THOR S2+S1 | 0.052 | 0.104 | 0.230 | 0.025 | 0.020 | 0.060 | 0.099 | 0.013 | 0.008 | 0.007 | 0.024 | 0.004 |
| TESSERA v2 | 0.165 | 0.436 | 0.637 | 0.035 | 0.140 | 0.402 | 0.584 | 0.028 | 0.144 | 0.379 | 0.590 | 0.034 |
| AlphaEarth | 0.102 | 0.205 | 0.522 | 0.026 | 0.127 | 0.224 | 0.578 | 0.035 | 0.118 | 0.224 | 0.555 | 0.037 |
| U-Net baseline | 0.169 | 0.683 | 0.631 | 0.036 | 0.168 | 0.629 | 0.620 | 0.041 | 0.148 | 0.570 | 0.560 | 0.054 |

**Predicted parcels of the boundary head at 0.5**: all, unmatched, unmatched slivers under 20 pixels (the extent alone: 23,146 predicted, 2,558 unmatched)

| Arm | predicted, K = 100 % | unmatched, K = 100 % | slivers, K = 100 % | predicted, K = 20 % | unmatched, K = 20 % | slivers, K = 20 % | predicted, K = 5 % | unmatched, K = 5 % | slivers, K = 5 % |
|---|---|---|---|---|---|---|---|---|---|
| TerraMind S2+S1 | 33,201 | 9,371 | 4,242 | 33,038 | 9,259 | 4,335 | 30,712 | 8,580 | 3,879 |
| Prithvi 600M | 35,493 | 9,702 | 4,295 | 32,728 | 9,174 | 4,530 | 30,069 | 8,019 | 3,765 |
| THOR S2+S1 | 29,639 | 7,864 | 3,610 | 27,249 | 5,982 | 2,542 | 24,615 | 3,877 | 948 |
| TESSERA v2 | 48,155 | 20,801 | 13,538 | 45,735 | 19,059 | 12,891 | 46,998 | 20,718 | 13,845 |
| AlphaEarth | 36,751 | 12,780 | 7,406 | 38,201 | 13,932 | 7,577 | 37,401 | 13,355 | 7,435 |
| U-Net baseline | 40,752 | 11,050 | 5,616 | 42,057 | 13,407 | 6,830 | 36,505 | 9,848 | 4,549 |
