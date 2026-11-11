# Tables of REPORT.md

Written by [report.py](report.py). Test tiles of `EE_2021_rev` (1,475), draw 0, seed 0 per cell.

## Borders and fields, two-head workflow

| Arm | K | border P / R / F1 within 1 px | border F1 at 0 / 2 px | PQ agn. | SQ / RQ agn. | PQ class-aware | fields / declared | RQ agn. of PRUE's own fields (cc) |
|---|---|---|---|---|---|---|---|---|
| TerraMind v1 L, S2 + S1 | 100 % | 0.926 / 0.699 / 0.797 | 0.442 / 0.866 | 0.327 | 0.783 / 0.417 | 0.281 | 38,246 / 55,575 | 0.371 |
| TerraMind v1 L, S2 + S1 | 20 % | 0.787 / 0.583 / 0.669 | 0.309 / 0.819 | 0.275 | 0.745 / 0.369 | 0.196 | 36,226 / 55,575 | 0.331 |
| TerraMind v1 L, S2 + S1 | 5 % | 0.645 / 0.468 / 0.542 | 0.221 / 0.743 | 0.215 | 0.709 / 0.302 | 0.088 | 32,989 / 55,575 | 0.253 |
| Prithvi-EO-2.0 600M | 100 % | 0.926 / 0.739 / 0.822 | 0.474 / 0.881 | 0.348 | 0.791 / 0.439 | 0.287 | 44,200 / 55,575 | 0.381 |
| Prithvi-EO-2.0 600M | 20 % | 0.863 / 0.640 / 0.735 | 0.362 / 0.852 | 0.303 | 0.765 / 0.395 | 0.211 | 38,302 / 55,575 | 0.354 |
| Prithvi-EO-2.0 600M | 5 % | 0.691 / 0.498 / 0.579 | 0.242 / 0.763 | 0.241 | 0.730 / 0.331 | 0.110 | 33,109 / 55,575 | 0.295 |
| THOR v1 L, S2 + S1 | 100 % | 0.847 / 0.538 / 0.658 | 0.296 / 0.785 | 0.226 | 0.732 / 0.309 | 0.191 | 28,483 / 55,575 | 0.273 |
| THOR v1 L, S2 + S1 | 20 % | 0.723 / 0.467 / 0.567 | 0.229 / 0.750 | 0.189 | 0.702 / 0.269 | 0.148 | 29,615 / 55,575 | 0.225 |
| THOR v1 L, S2 + S1 | 5 % | 0.564 / 0.350 / 0.432 | 0.156 / 0.631 | 0.127 | 0.668 / 0.191 | 0.066 | 24,675 / 55,575 | 0.147 |
| TESSERA v2 M | 100 % | 0.941 / 0.805 / 0.868 | 0.582 / 0.908 | 0.369 | 0.815 / 0.452 | 0.388 | 53,747 / 55,575 | 0.392 |
| TESSERA v2 M | 20 % | 0.903 / 0.737 / 0.811 | 0.445 / 0.895 | 0.328 | 0.781 / 0.421 | 0.330 | 52,001 / 55,575 | 0.368 |
| TESSERA v2 M | 5 % | 0.847 / 0.691 / 0.761 | 0.392 / 0.879 | 0.311 | 0.766 / 0.406 | 0.290 | 49,078 / 55,575 | 0.341 |
| AlphaEarth v1 | 100 % | 0.939 / 0.777 / 0.850 | 0.536 / 0.898 | 0.348 | 0.800 / 0.435 | 0.309 | 50,575 / 55,575 | 0.376 |
| AlphaEarth v1 | 20 % | 0.886 / 0.708 / 0.787 | 0.416 / 0.886 | 0.318 | 0.771 / 0.413 | 0.263 | 46,990 / 55,575 | 0.355 |
| AlphaEarth v1 | 5 % | 0.858 / 0.672 / 0.754 | 0.367 / 0.874 | 0.284 | 0.757 / 0.376 | 0.225 | 48,907 / 55,575 | 0.320 |
| Baseline: PRUE, S2 + S1 | 100 % | 0.940 / 0.807 / 0.868 | 0.582 / 0.908 | 0.374 | 0.814 / 0.460 | 0.389 | 51,798 / 55,575 | 0.394 |
| Baseline: PRUE, S2 + S1 | 20 % | 0.901 / 0.738 / 0.812 | 0.447 / 0.896 | 0.334 | 0.781 / 0.427 | 0.331 | 49,888 / 55,575 | 0.367 |
| Baseline: PRUE, S2 + S1 | 5 % | 0.847 / 0.677 / 0.752 | 0.380 / 0.874 | 0.305 | 0.764 / 0.399 | 0.276 | 47,082 / 55,575 | 0.337 |

## Crop types: two-head against crop-only workflow

| Arm | K | pixel Macro-F1 | crop-only | difference | parcel Macro-F1 | crop-only | difference | crop-only fine-tuned, pixel / parcel |
|---|---|---|---|---|---|---|---|---|
| TerraMind v1 L, S2 + S1 | 100 % | 0.598 | 0.661 | -0.063 | 0.531 | 0.583 | -0.052 | 0.598 / 0.545 |
| TerraMind v1 L, S2 + S1 | 20 % | 0.475 | 0.566 | -0.091 | 0.442 | 0.515 | -0.073 | 0.562 / 0.511 |
| TerraMind v1 L, S2 + S1 | 5 % | 0.275 | 0.433 | -0.158 | 0.251 | 0.381 | -0.129 | 0.351 / 0.316 |
| Prithvi-EO-2.0 600M | 100 % | 0.559 | 0.580 | -0.021 | 0.513 | 0.535 | -0.023 | 0.613 / 0.545 |
| Prithvi-EO-2.0 600M | 20 % | 0.471 | 0.505 | -0.034 | 0.425 | 0.472 | -0.047 | 0.540 / 0.492 |
| Prithvi-EO-2.0 600M | 5 % | 0.293 | 0.445 | -0.151 | 0.277 | 0.409 | -0.132 | 0.394 / 0.371 |
| THOR v1 L, S2 + S1 | 100 % | 0.562 | 0.594 | -0.032 | 0.507 | 0.516 | -0.010 | 0.627 / 0.545 |
| THOR v1 L, S2 + S1 | 20 % | 0.505 | 0.547 | -0.042 | 0.463 | 0.491 | -0.028 | 0.535 / 0.475 |
| THOR v1 L, S2 + S1 | 5 % | 0.317 | 0.393 | -0.076 | 0.282 | 0.349 | -0.067 | 0.404 / 0.369 |
| TESSERA v2 M | 100 % | 0.645 | 0.618 | +0.027 | 0.621 | 0.602 | +0.019 | 0.660 / 0.639 |
| TESSERA v2 M | 20 % | 0.614 | 0.582 | +0.032 | 0.594 | 0.574 | +0.020 | 0.630 / 0.603 |
| TESSERA v2 M | 5 % | 0.569 | 0.504 | +0.065 | 0.566 | 0.502 | +0.064 | 0.581 / 0.566 |
| AlphaEarth v1 | 100 % | 0.567 | 0.523 | +0.044 | 0.520 | 0.484 | +0.036 |  |
| AlphaEarth v1 | 20 % | 0.524 | 0.497 | +0.028 | 0.488 | 0.455 | +0.033 |  |
| AlphaEarth v1 | 5 % | 0.477 | 0.449 | +0.028 | 0.457 | 0.425 | +0.032 |  |
| Baseline: PRUE, S2 + S1 | 100 % | 0.655 | 0.616 | +0.039 | 0.629 | 0.553 | +0.076 |  |
| Baseline: PRUE, S2 + S1 | 20 % | 0.618 | 0.564 | +0.054 | 0.596 | 0.520 | +0.076 |  |
| Baseline: PRUE, S2 + S1 | 5 % | 0.561 | 0.467 | +0.094 | 0.550 | 0.446 | +0.104 |  |

The crop-only baseline is the ResNet-18 U-Net from scratch (`unet_r18_scratch_s2s1`); the two-head baseline is PRUE from ImageNet (`unet_b7_s2s1`).

## Border recall within 1 px by what lies across the border

| Arm | K | same crop | other crop | undeclared farmland | non-farm land | false borders on a declared parcel |
|---|---|---|---|---|---|---|
| TerraMind v1 L, S2 + S1 | 100 % | 0.272 | 0.770 | 0.734 | 0.806 | 0.046 |
| TerraMind v1 L, S2 + S1 | 20 % | 0.311 | 0.731 | 0.670 | 0.426 | 0.206 |
| TerraMind v1 L, S2 + S1 | 5 % | 0.320 | 0.632 | 0.546 | 0.255 | 0.350 |
| Prithvi-EO-2.0 600M | 100 % | 0.326 | 0.816 | 0.775 | 0.832 | 0.043 |
| Prithvi-EO-2.0 600M | 20 % | 0.317 | 0.755 | 0.720 | 0.551 | 0.127 |
| Prithvi-EO-2.0 600M | 5 % | 0.322 | 0.689 | 0.581 | 0.274 | 0.306 |
| THOR v1 L, S2 + S1 | 100 % | 0.181 | 0.619 | 0.541 | 0.677 | 0.092 |
| THOR v1 L, S2 + S1 | 20 % | 0.225 | 0.599 | 0.510 | 0.412 | 0.255 |
| THOR v1 L, S2 + S1 | 5 % | 0.261 | 0.459 | 0.387 | 0.239 | 0.427 |
| TESSERA v2 M | 100 % | 0.366 | 0.856 | 0.850 | 0.908 | 0.035 |
| TESSERA v2 M | 20 % | 0.349 | 0.818 | 0.827 | 0.679 | 0.089 |
| TESSERA v2 M | 5 % | 0.364 | 0.795 | 0.810 | 0.517 | 0.151 |
| AlphaEarth v1 | 100 % | 0.344 | 0.830 | 0.814 | 0.894 | 0.036 |
| AlphaEarth v1 | 20 % | 0.352 | 0.793 | 0.806 | 0.612 | 0.110 |
| AlphaEarth v1 | 5 % | 0.326 | 0.737 | 0.786 | 0.547 | 0.138 |
| Baseline: PRUE, S2 + S1 | 100 % | 0.373 | 0.856 | 0.850 | 0.909 | 0.036 |
| Baseline: PRUE, S2 + S1 | 20 % | 0.364 | 0.818 | 0.828 | 0.677 | 0.092 |
| Baseline: PRUE, S2 + S1 | 5 % | 0.352 | 0.774 | 0.795 | 0.506 | 0.151 |

## Training

| Arm | K | training tiles | labelled parcels | epoch kept | by object F1 alone | by crop loss alone | final crop weight | fit h |
|---|---|---|---|---|---|---|---|---|
| TerraMind v1 L, S2 + S1 | 100 % | 4,898 | 86,909 | 43 | 39 | 29 | 0.0012 | 1.45 |
| TerraMind v1 L, S2 + S1 | 20 % | 4,213 | 17,390 | 42 | 40 | 27 | 0.0060 | 1.26 |
| TerraMind v1 L, S2 + S1 | 5 % | 2,706 | 4,354 | 44 | 39 | 35 | 0.0024 | 0.91 |
| Prithvi-EO-2.0 600M | 100 % | 4,898 | 86,909 | 41 | 34 | 37 | 0.0014 | 2.47 |
| Prithvi-EO-2.0 600M | 20 % | 4,213 | 17,390 | 43 | 43 | 22 | 0.0024 | 2.05 |
| Prithvi-EO-2.0 600M | 5 % | 2,706 | 4,354 | 39 | 32 | 27 | 0.0025 | 1.42 |
| THOR v1 L, S2 + S1 | 100 % | 4,898 | 86,909 | 44 | 44 | 20 | 0.0021 | 1.58 |
| THOR v1 L, S2 + S1 | 20 % | 4,213 | 17,390 | 45 | 45 | 30 | 0.0028 | 1.33 |
| THOR v1 L, S2 + S1 | 5 % | 2,706 | 4,354 | 43 | 43 | 25 | 0.0043 | 1.03 |
| TESSERA v2 M | 100 % | 4,898 | 86,909 | 40 | 6 | 43 | 0.0018 | 3.49 |
| TESSERA v2 M | 20 % | 4,213 | 17,390 | 30 | 36 | 29 | 0.0070 | 2.91 |
| TESSERA v2 M | 5 % | 2,706 | 4,354 | 37 | 37 | 9 | 0.0130 | 1.92 |
| AlphaEarth v1 | 100 % | 4,898 | 86,909 | 43 | 36 | 45 | 0.0030 | 4.27 |
| AlphaEarth v1 | 20 % | 4,213 | 17,390 | 43 | 43 | 23 | 0.0146 | 3.69 |
| AlphaEarth v1 | 5 % | 2,706 | 4,354 | 34 | 31 | 14 | 0.0163 | 1.74 |
| Baseline: PRUE, S2 + S1 | 100 % | 4,898 | 86,909 | 43 | 32 | 42 | 0.0064 | 5.87 |
| Baseline: PRUE, S2 + S1 | 20 % | 4,213 | 17,390 | 43 | 34 | 37 | 0.0139 | 5.06 |
| Baseline: PRUE, S2 + S1 | 5 % | 2,706 | 4,354 | 36 | 36 | 24 | 0.0082 | 1.90 |

## TESSERA v2 at K = 100 %: frozen and fine-tuned, separate models and one network

| Encoder | Setting | crop pixel / parcel Macro-F1 | border F1 within 1 px | PQ agn. | RQ agn. | PQ class-aware | fit h |
|---|---|---|---|---|---|---|---|
| frozen | two separate models | 0.618 / 0.602 | 0.862 | 0.374 | 0.460 | 0.367 | 1.5 |
| frozen | one network, crop weight 0.1 | 0.639 / 0.608 | 0.844 | 0.347 | 0.432 | 0.352 | 1.2 |
| frozen | one network, crop weight 0.01 | 0.624 / 0.604 | 0.861 | 0.366 | 0.451 | 0.370 | 1.2 |
| frozen | one network, balanced (this workflow) | 0.645 / 0.621 | 0.868 | 0.369 | 0.452 | 0.388 | 3.5 |
| fine-tuned | two separate models | 0.660 / 0.639 | 0.868 | 0.378 | 0.463 | 0.397 | 54.1 |
| fine-tuned | one network, crop weight 0.1 | 0.687 / 0.646 | 0.847 | 0.360 | 0.445 | 0.380 | 55.0 |
| fine-tuned | one network, crop weight 0.01 | 0.678 / 0.658 | 0.870 | 0.376 | 0.459 | 0.411 | 60.0 |

Fit hours: the two separate models' sum; the frozen balanced network trained on whole chips for 45 epochs, the fixed-weight networks on 128 x 128 windows for 30 epochs.

## Crop types, fine-tuned against frozen, crop-only workflow (pixel / parcel Macro-F1)

| Arm | K | frozen | fine-tuned | difference |
|---|---|---|---|---|
| TerraMind v1 L, S2 + S1 | 100 % | 0.661 / 0.583 | 0.598 / 0.545 | -0.063 / -0.038 |
| TerraMind v1 L, S2 + S1 | 20 % | 0.566 / 0.515 | 0.562 / 0.511 | -0.004 / -0.004 |
| TerraMind v1 L, S2 + S1 | 5 % | 0.433 / 0.381 | 0.351 / 0.316 | -0.082 / -0.065 |
| Prithvi-EO-2.0 600M | 100 % | 0.580 / 0.535 | 0.613 / 0.545 | +0.033 / +0.009 |
| Prithvi-EO-2.0 600M | 20 % | 0.505 / 0.472 | 0.540 / 0.492 | +0.034 / +0.020 |
| Prithvi-EO-2.0 600M | 5 % | 0.445 / 0.409 | 0.394 / 0.371 | -0.051 / -0.038 |
| THOR v1 L, S2 + S1 | 100 % | 0.594 / 0.516 | 0.627 / 0.545 | +0.033 / +0.029 |
| THOR v1 L, S2 + S1 | 20 % | 0.547 / 0.491 | 0.535 / 0.475 | -0.012 / -0.016 |
| THOR v1 L, S2 + S1 | 5 % | 0.393 / 0.349 | 0.404 / 0.369 | +0.011 / +0.020 |
| TESSERA v2 M | 100 % | 0.618 / 0.602 | 0.660 / 0.639 | +0.042 / +0.037 |
| TESSERA v2 M | 20 % | 0.582 / 0.574 | 0.630 / 0.603 | +0.047 / +0.028 |
| TESSERA v2 M | 5 % | 0.504 / 0.502 | 0.581 / 0.566 | +0.077 / +0.063 |

## Field delineation: published on Fields of The World (PRUE, Table 1) and ours

| Model | Data | Pixel IoU | Object F1 |
|---|---|---|---|
| PRUE: U-Net + EfficientNet-B7, log-cosh Dice | FTW test sets, 24 countries | 0.76 | 0.47 |
| FTW baseline: U-Net + EfficientNet-B3 | FTW test sets, 24 countries | 0.70 | 0.38 |
| Mask2Former (Swin-S) | FTW test sets, 24 countries | 0.68 | 0.39 |
| SAM ViT-H, fine-tuned / zero-shot | FTW test sets, 24 countries | 0.45 / 0.32 | 0.37 / 0.17 |
| DECODE (FracTAL ResUNet, multi-task) | FTW test sets, 24 countries | 0.71 | 0.21 |
| Delineate Anything, zero-shot | FTW test sets, 24 countries | 0.37 | 0.09 |
| Clay ViT-L, frozen; fully fine-tuned | FTW test sets, 24 countries | 0.67; 0.73 | 0.36; 0.42 |
| Galileo ViT-B, DINOv3 ViT-L, frozen | FTW test sets, 24 countries | 0.66, 0.60 | 0.32, 0.31 |
| TerraMind ViT-B, Prithvi 2.0 ViT-L, frozen | FTW test sets, 24 countries | 0.57, 0.56 | 0.26, 0.25 |
| TerraMind v1 L, S2 + S1, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.42 grown, 0.37 interior only |
| Prithvi-EO-2.0 600M, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.44 grown, 0.38 interior only |
| THOR v1 L, S2 + S1, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.31 grown, 0.27 interior only |
| TESSERA v2 M, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.45 grown, 0.39 interior only |
| AlphaEarth v1, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.44 grown, 0.38 interior only |
| Baseline: PRUE, S2 + S1, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | 0.46 grown, 0.39 interior only |
