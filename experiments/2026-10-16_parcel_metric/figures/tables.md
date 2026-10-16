**Test Macro-F1 per pixel / per parcel**

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind L, S2+S1 | 0.661 / **0.583** | 0.566 / **0.515** | 0.433 / **0.381** |
| THOR L, S2+S1 | 0.594 / **0.516** | 0.547 / **0.491** | 0.393 / **0.349** |
| Prithvi 600M, S2 | 0.580 / **0.535** |  | 0.445 / **0.409** |
| TESSERA v2 + MLP | 0.605 / **0.589** | 0.562 / **0.547** | 0.504 / **0.502** |
| AlphaEarth + MLP | 0.521 / **0.474** | 0.465 / **0.432** | 0.427 / **0.408** |
| U-Net S2, 15 ep | 0.498 / **0.473** | 0.415 / **0.399** | 0.342 / **0.327** |
| U-Net S2, 45 ep | 0.544 / **0.502** | 0.477 / **0.451** | 0.413 / **0.395** |

**Parcel minus pixel Macro-F1**

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind L, S2+S1 | -0.078 | -0.051 | -0.052 |
| THOR L, S2+S1 | -0.078 | -0.057 | -0.043 |
| Prithvi 600M, S2 | -0.045 |  | -0.036 |
| TESSERA v2 + MLP | -0.016 | -0.016 | -0.002 |
| AlphaEarth + MLP | -0.047 | -0.034 | -0.019 |
| U-Net S2, 15 ep | -0.025 +- +0.005 | -0.016 +- +0.004 | -0.015 +- +0.002 |
| U-Net S2, 45 ep | -0.042 +- +0.005 | -0.026 +- +0.002 | -0.018 +- +0.012 |

**Parcel overall accuracy / kappa**

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind L, S2+S1 | 0.787 / 0.698 | 0.742 / 0.639 | 0.689 / 0.563 |
| THOR L, S2+S1 | 0.753 / 0.650 | 0.727 / 0.613 | 0.673 / 0.540 |
| Prithvi 600M, S2 | 0.763 / 0.661 |  | 0.677 / 0.557 |
| TESSERA v2 + MLP | 0.793 / 0.699 | 0.779 / 0.681 | 0.755 / 0.638 |
| AlphaEarth + MLP | 0.745 / 0.632 | 0.723 / 0.613 | 0.715 / 0.595 |
| U-Net S2, 15 ep | 0.738 / 0.624 | 0.709 / 0.586 | 0.683 / 0.531 |
| U-Net S2, 45 ep | 0.746 / 0.638 | 0.730 / 0.610 | 0.698 / 0.565 |

**Pixel overall accuracy / kappa**

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind L, S2+S1 | 0.830 / 0.797 | 0.775 / 0.734 | 0.712 / 0.656 |
| THOR L, S2+S1 | 0.790 / 0.750 | 0.755 / 0.708 | 0.685 / 0.626 |
| Prithvi 600M, S2 | 0.792 / 0.751 |  | 0.695 / 0.639 |
| TESSERA v2 + MLP | 0.784 / 0.741 | 0.768 / 0.721 | 0.729 / 0.672 |
| AlphaEarth + MLP | 0.738 / 0.686 | 0.710 / 0.656 | 0.690 / 0.630 |
| U-Net S2, 15 ep | 0.745 / 0.695 | 0.703 / 0.645 | 0.658 / 0.582 |
| U-Net S2, 45 ep | 0.759 / 0.713 | 0.731 / 0.677 | 0.683 / 0.620 |

**Misclassified pixels removed by the vote / correct pixels lost to it**

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind L, S2+S1 | 24% / 2% | 26% / 3% | 22% / 3% |
| THOR L, S2+S1 | 23% / 2% | 25% / 3% | 22% / 4% |
| Prithvi 600M, S2 | 24% / 2% |  | 25% / 5% |
| TESSERA v2 + MLP | 32% / 3% | 31% / 3% | 29% / 4% |
| AlphaEarth + MLP | 24% / 4% | 24% / 4% | 21% / 4% |
| U-Net S2, 15 ep | 22% / 3% | 22% / 4% | 16% / 3% |
| U-Net S2, 45 ep | 22% / 3% | 21% / 3% | 20% / 4% |

**Sensitivity of the parcel Macro-F1**: the main rule, and the change when only interior pixels vote, when the 3,370 parcels reaching outside the test chips are left out, and when the 1,724 parcels under 16 pixels are left out

| Arm | K | All parcels, all pixels | Interior pixels vote | Without cut parcels | Parcels of 16 px or more |
|---|---|---|---|---|---|
| TerraMind L, S2+S1 | 100 % | 0.583 | +0.001 | +0.002 | +0.022 |
| TerraMind L, S2+S1 | 20 % | 0.515 | -0.001 | +0.004 | +0.011 |
| TerraMind L, S2+S1 | 5 % | 0.381 | +0.001 | +0.002 | +0.010 |
| THOR L, S2+S1 | 100 % | 0.516 | +0.001 | +0.003 | +0.017 |
| THOR L, S2+S1 | 20 % | 0.491 | +0.002 | +0.006 | +0.011 |
| THOR L, S2+S1 | 5 % | 0.349 | +0.002 | +0.002 | +0.007 |
| Prithvi 600M, S2 | 100 % | 0.535 | +0.001 | +0.002 | +0.010 |
| Prithvi 600M, S2 | 5 % | 0.409 | +0.001 | +0.001 | +0.009 |
| TESSERA v2 + MLP | 100 % | 0.589 | +0.009 | -0.001 | +0.019 |
| TESSERA v2 + MLP | 20 % | 0.547 | +0.006 | -0.001 | +0.018 |
| TESSERA v2 + MLP | 5 % | 0.502 | +0.007 | -0.001 | +0.015 |
| AlphaEarth + MLP | 100 % | 0.474 | +0.004 | -0.001 | +0.019 |
| AlphaEarth + MLP | 20 % | 0.432 | +0.005 | +0.002 | +0.014 |
| AlphaEarth + MLP | 5 % | 0.408 | +0.002 | -0.000 | +0.010 |
| U-Net S2, 15 ep | 100 % | 0.473 | +0.002 | +0.001 | +0.009 |
| U-Net S2, 15 ep | 20 % | 0.399 | +0.002 | +0.002 | +0.006 |
| U-Net S2, 15 ep | 5 % | 0.327 | +0.002 | +0.002 | +0.005 |
| U-Net S2, 45 ep | 100 % | 0.502 | +0.003 | +0.002 | +0.013 |
| U-Net S2, 45 ep | 20 % | 0.451 | +0.002 | +0.002 | +0.008 |
| U-Net S2, 45 ep | 5 % | 0.395 | +0.002 | +0.001 | +0.007 |

**Ranking of the arms**

| K | Per pixel | Per parcel |
|---|---|---|
| 100 % | TerraMind L, S2+S1 > TESSERA v2 + MLP > THOR L, S2+S1 > Prithvi 600M, S2 > U-Net S2, 45 ep > AlphaEarth + MLP > U-Net S2, 15 ep | TESSERA v2 + MLP > TerraMind L, S2+S1 > Prithvi 600M, S2 > THOR L, S2+S1 > U-Net S2, 45 ep > AlphaEarth + MLP > U-Net S2, 15 ep |
| 20 % | TerraMind L, S2+S1 > TESSERA v2 + MLP > THOR L, S2+S1 > U-Net S2, 45 ep > AlphaEarth + MLP > U-Net S2, 15 ep | TESSERA v2 + MLP > TerraMind L, S2+S1 > THOR L, S2+S1 > U-Net S2, 45 ep > AlphaEarth + MLP > U-Net S2, 15 ep |
| 5 % | TESSERA v2 + MLP > Prithvi 600M, S2 > TerraMind L, S2+S1 > AlphaEarth + MLP > U-Net S2, 45 ep > THOR L, S2+S1 > U-Net S2, 15 ep | TESSERA v2 + MLP > Prithvi 600M, S2 > AlphaEarth + MLP > U-Net S2, 45 ep > TerraMind L, S2+S1 > THOR L, S2+S1 > U-Net S2, 15 ep |

**Run**

* Chip set EE_2021_rev, split blocks4_buf1600_seed0__6b0eb4cb: 1,475 test chips, 34,610 test parcels with labelled pixels, of which 3,370 reach outside the test chips and 1,724 have fewer than 16 pixels.
* Parcels by area inside the test chips: < 0.16 ha 1,724, 0.16 to 0.64 4,378, 0.64 to 2.6 10,802, 2.6 to 10 12,385, > 10 ha 5,321.
* 31 fits scored. Largest difference between the pixel Macro-F1 recomputed from the stored predictions and the one the fit logged: 0.0000.
* Left out, predicted with another checkpoint than the one tested: Prithvi 600M, S2, K = 20 %, seed 0 (pixel Macro-F1 0.5014 from the stored predictions, 0.5054 logged).
* Left out, predicted with another checkpoint than the one tested: U-Net S2, 45 ep, K = 100 %, seed 0 (pixel Macro-F1 0.5574 from the stored predictions, 0.5449 logged).

