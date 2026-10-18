**Test Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.661 | 0.566 | 0.433 |
| TerraMind S2+S1, fine-tuned | 0.598 | 0.562 | 0.351 |
| Prithvi 600M, frozen | 0.580 | 0.505 | 0.445 |
| Prithvi 600M, fine-tuned | 0.613 | 0.540 | 0.394 |
| THOR S2+S1, frozen | 0.594 | 0.547 | 0.393 |
| THOR S2+S1, fine-tuned | 0.627 | 0.535 | 0.404 |
| TESSERA v2, frozen | 0.618 | 0.582 | 0.504 |
| TESSERA v2, fine-tuned | 0.660 | 0.630 | 0.581 |
| TESSERA v2, frozen, labelled pixels | 0.604 | 0.572 | 0.543 |

**Test parcel Macro-F1**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.583 | 0.515 | 0.381 |
| TerraMind S2+S1, fine-tuned | 0.545 | 0.511 | 0.316 |
| Prithvi 600M, frozen | 0.535 | 0.472 | 0.409 |
| Prithvi 600M, fine-tuned | 0.545 | 0.492 | 0.371 |
| THOR S2+S1, frozen | 0.516 | 0.491 | 0.349 |
| THOR S2+S1, fine-tuned | 0.545 | 0.475 | 0.369 |
| TESSERA v2, frozen | 0.602 | 0.574 | 0.502 |
| TESSERA v2, fine-tuned | 0.639 | 0.603 | 0.566 |
| TESSERA v2, frozen, labelled pixels | 0.591 | 0.557 | 0.529 |

**Test mIoU**, mean +- standard deviation (fits)

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.570 | 0.461 | 0.334 |
| TerraMind S2+S1, fine-tuned | 0.509 | 0.466 | 0.271 |
| Prithvi 600M, frozen | 0.479 | 0.398 | 0.338 |
| Prithvi 600M, fine-tuned | 0.513 | 0.439 | 0.297 |
| THOR S2+S1, frozen | 0.490 | 0.435 | 0.295 |
| THOR S2+S1, fine-tuned | 0.525 | 0.431 | 0.300 |
| TESSERA v2, frozen | 0.515 | 0.471 | 0.394 |
| TESSERA v2, fine-tuned | 0.568 | 0.527 | 0.482 |
| TESSERA v2, frozen, labelled pixels | 0.501 | 0.467 | 0.438 |

**Paired difference in test Macro-F1, per pixel**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1: fine-tuned minus frozen | -0.063 | -0.004 | -0.082 |
| Prithvi 600M: fine-tuned minus frozen | +0.033 | +0.034 | -0.051 |
| THOR S2+S1: fine-tuned minus frozen | +0.033 | -0.012 | +0.011 |
| TESSERA v2: fine-tuned minus frozen | +0.042 | +0.047 | +0.077 |
| TESSERA v2: fine-tuned minus frozen control | +0.056 | +0.058 | +0.037 |
| TESSERA v2: frozen control minus frozen | -0.014 | -0.010 | +0.039 |

**Paired difference in test Macro-F1, per parcel**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1: fine-tuned minus frozen | -0.038 | -0.004 | -0.065 |
| Prithvi 600M: fine-tuned minus frozen | +0.009 | +0.020 | -0.038 |
| THOR S2+S1: fine-tuned minus frozen | +0.029 | -0.016 | +0.020 |
| TESSERA v2: fine-tuned minus frozen | +0.037 | +0.028 | +0.063 |
| TESSERA v2: fine-tuned minus frozen control | +0.048 | +0.046 | +0.037 |
| TESSERA v2: frozen control minus frozen | -0.011 | -0.018 | +0.027 |

**Paired difference in best validation Macro-F1 over the epochs**, mean +- standard deviation (fits won by the first)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1: fine-tuned minus frozen | +0.012 | +0.008 | +0.010 |
| Prithvi 600M: fine-tuned minus frozen | +0.016 | +0.004 | +0.008 |
| THOR S2+S1: fine-tuned minus frozen | +0.010 | -0.000 | -0.033 |
| TESSERA v2: fine-tuned minus frozen | +0.053 | +0.051 | +0.056 |
| TESSERA v2: fine-tuned minus frozen control | +0.055 | +0.056 | +0.050 |
| TESSERA v2: frozen control minus frozen | -0.002 | -0.005 | +0.005 |

**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 13 / 28 | 12 / 39 | 12 / 44 |
| TerraMind S2+S1, fine-tuned | 5 / 38 | 7 / 25 | 7 / 44 |
| Prithvi 600M, frozen | 7 / 33 | 8 / 23 | 10 / 39 |
| Prithvi 600M, fine-tuned | 6 / 13 | 9 / 21 | 7 / 42 |
| THOR S2+S1, frozen | 7 / 44 | 14 / 34 | 9 / 41 |
| THOR S2+S1, fine-tuned | 8 / 31 | 9 / 24 | 9 / 15 |
| TESSERA v2, frozen | 34 / 41 | 22 / 27 | 4 / 25 |
| TESSERA v2, fine-tuned | 8 / 10 | 5 / 26 | 5 / 19 |
| TESSERA v2, frozen, labelled pixels | 17 / 30 | 38 / 26 | 34 / 26 |

**Validation Macro-F1** at the checkpoint / the best epoch / the last epoch, mean over fits

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, frozen | 0.659 / 0.676 / 0.667 | 0.569 / 0.616 / 0.596 | 0.428 / 0.505 / 0.505 |
| TerraMind S2+S1, fine-tuned | 0.595 / 0.687 / 0.678 | 0.554 / 0.624 / 0.612 | 0.339 / 0.515 / 0.515 |
| Prithvi 600M, frozen | 0.586 / 0.639 / 0.623 | 0.511 / 0.583 / 0.575 | 0.439 / 0.471 / 0.446 |
| Prithvi 600M, fine-tuned | 0.604 / 0.655 / 0.644 | 0.532 / 0.588 / 0.574 | 0.391 / 0.479 / 0.472 |
| THOR S2+S1, frozen | 0.580 / 0.643 / 0.643 | 0.553 / 0.578 / 0.565 | 0.389 / 0.460 / 0.447 |
| THOR S2+S1, fine-tuned | 0.625 / 0.653 / 0.649 | 0.536 / 0.578 / 0.551 | 0.393 / 0.427 / 0.354 |
| TESSERA v2, frozen | 0.610 / 0.622 / 0.610 | 0.581 / 0.594 / 0.566 | 0.515 / 0.558 / 0.530 |
| TESSERA v2, fine-tuned | 0.663 / 0.675 / 0.663 | 0.638 / 0.645 / 0.634 | 0.583 / 0.614 / 0.594 |
| TESSERA v2, frozen, labelled pixels | 0.601 / 0.620 / 0.607 | 0.574 / 0.589 / 0.589 | 0.544 / 0.564 / 0.528 |

**Run**

* Split blocks4_buf1600_seed0__6b0eb4cb: 4898 training, 769 validation, 1475 test chips.
* K = 100 %: 4898 training chips, 86,909 parcels (draw 0).
* K = 20 %: 4213 training chips, 17,390 parcels (draw 0).
* K = 5 %: 2706 training chips, 4,354 parcels (draw 0).
* 27 fits, 94.9 GPU hours of fitting; machines cluster; commits 02a10f18, 94f87ca5, b2b159cf, bf054ebb.

**This folder's arms, test Macro-F1** at the checkpoint of lowest validation loss (the protocol's) / of highest validation Macro-F1 (the secondary readout), mean over fits; the epochs of the second

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind S2+S1, fine-tuned | 0.598 / 0.688 (epochs 38) | 0.562 / 0.627 (epochs 25) | 0.351 / 0.523 (epochs 44) |
| Prithvi 600M, fine-tuned | 0.613 / 0.665 (epochs 13) | 0.540 / 0.601 (epochs 21) | 0.394 / 0.488 (epochs 42) |
| THOR S2+S1, fine-tuned | 0.627 / 0.647 (epochs 31) | 0.535 / 0.562 (epochs 24) | 0.404 / 0.439 (epochs 15) |
| TESSERA v2, fine-tuned | 0.660 / 0.671 (epochs 10) | 0.630 / 0.645 (epochs 26) | 0.581 / 0.612 (epochs 19) |
| TESSERA v2, frozen, labelled pixels | 0.604 / 0.620 (epochs 30) | 0.572 / 0.587 (epochs 26) | 0.543 / 0.558 (epochs 26) |

**Cost**: fitting hours per cell, mean over seeds; the arm's total; peak GPU memory in GB

| Arm | K = 100 % | K = 20 % | K = 5 % | Total h | Peak GB |
|---|---|---|---|---|---|
| TerraMind S2+S1, frozen | 1.04 | 0.98 | 0.69 | 2.7 | 2.54 |
| TerraMind S2+S1, fine-tuned | 4.78 | 4.13 | 2.80 | 11.7 | 18.52 |
| Prithvi 600M, frozen | 1.54 | 1.09 | 0.77 | 3.4 | 2.78 |
| Prithvi 600M, fine-tuned | 6.69 | 5.89 | 3.88 | 16.5 | 22.92 |
| THOR S2+S1, frozen | 0.89 | 0.82 | 0.57 | 2.3 | 2.38 |
| THOR S2+S1, fine-tuned | 8.03 | 7.03 | 4.68 | 19.7 | 26.51 |
| TESSERA v2, frozen | 0.94 | 0.78 | 0.55 | 2.3 | 12.34 |
| TESSERA v2, fine-tuned | 25.84 | 6.22 | 2.50 | 34.6 | 8.96 |
| TESSERA v2, frozen, labelled pixels | 0.88 | 0.55 | 0.36 | 1.8 | 1.7 |
