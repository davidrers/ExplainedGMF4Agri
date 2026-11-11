## Headline

| Model | crop pixel Macro-F1 | crop parcel Macro-F1 | border P / R / F1 at 1 px | PQ agn. | PQ class-aware | fields | fit h |
|---|---|---|---|---|---|---|---|
| TESSERA frozen: two separate models | 0.618 | 0.602 | 0.947 / 0.790 / 0.862 | 0.374 | 0.367 | 47,264 | 1.51 |
| **one joint model, crop weight 0.1** | 0.639 | 0.608 | 0.937 / 0.768 / 0.844 | 0.347 | 0.352 | 47,860 | 1.21 |
| difference | +0.021 | +0.005 | F1 -0.017 | -0.027 | -0.015 | | |
| **one joint model, crop weight 0.01** | 0.624 | 0.604 | 0.940 / 0.795 / 0.861 | 0.366 | 0.370 | 50,680 | 1.20 |
| difference | +0.006 | +0.002 | F1 -0.000 | -0.008 | +0.002 | | |
| TESSERA fine-tuned: two separate models | 0.660 | 0.639 | 0.938 / 0.807 / 0.868 | 0.378 | 0.397 | 51,949 | 54.12 |
| **one joint model, crop weight 0.1** | 0.687 | 0.646 | 0.937 / 0.773 / 0.847 | 0.360 | 0.380 | 45,781 | 54.99 |
| difference | +0.027 | +0.007 | F1 -0.020 | -0.018 | -0.017 | | |
| **one joint model, crop weight 0.01** | 0.678 | 0.658 | 0.942 / 0.808 / 0.870 | 0.376 | 0.411 | 52,563 | 59.95 |
| difference | +0.018 | +0.019 | F1 +0.002 | -0.002 | +0.013 | | |

## Checkpoint

| Arm | epoch kept | val joint | val object F1 | val crop Macro-F1 | epoch by object F1 alone | by crop loss alone | by crop Macro-F1 alone | crop loss weight |
|---|---|---|---|---|---|---|---|---|
| TESSERA frozen, crop weight 0.1 | 29 | 0.5041 | 0.3745 | 0.6338 | 16 | 30 | 28 | 0.1 |
| TESSERA frozen, crop weight 0.01 | 29 | 0.5069 | 0.3936 | 0.6202 | 17 | 30 | 26 | 0.01 |
| TESSERA fine-tuned, crop weight 0.1 | 28 | 0.5395 | 0.3908 | 0.6882 | 28 | 20 | 30 | 0.1 |
| TESSERA fine-tuned, crop weight 0.01 | 29 | 0.5419 | 0.4066 | 0.6772 | 29 | 25 | 29 | 0.01 |

## Border recall at 1 px by type

| Model | same crop | other crop | undeclared farmland | non-farm land |
|---|---|---|---|---|
| TESSERA frozen: PRUE alone | 0.344 | 0.820 | 0.845 | 0.883 |
| joint, crop weight 0.1 | 0.298 | 0.780 | 0.823 | 0.889 |
| joint, crop weight 0.01 | 0.350 | 0.825 | 0.844 | 0.901 |
| TESSERA fine-tuned: PRUE alone | 0.377 | 0.841 | 0.856 | 0.905 |
| joint, crop weight 0.1 | 0.299 | 0.795 | 0.827 | 0.890 |
| joint, crop weight 0.01 | 0.366 | 0.858 | 0.857 | 0.903 |

## Per chip

| Arm | chips (>= 100 true border px) | joint better | median difference |
|---|---|---|---|
| TESSERA frozen, crop weight 0.1 | 1429 | 19.2% | -0.0141 |
| TESSERA frozen, crop weight 0.01 | 1431 | 45.6% | -0.0012 |
| TESSERA fine-tuned, crop weight 0.1 | 1432 | 22.6% | -0.0153 |
| TESSERA fine-tuned, crop weight 0.01 | 1433 | 56.5% | +0.0020 |

## Pixel F1 per crop class

| Class | test pixels | frozen: separate | joint 0.1 | joint 0.01 | fine-tuned: separate | joint 0.1 | joint 0.01 |
|---|---|---|---|---|---|---|---|
| pasture meadow grassland grass | 6,123,407 | 0.874 | 0.894 | 0.884 | 0.881 | 0.901 | 0.891 |
| winter common soft wheat | 2,768,472 | 0.923 | 0.936 | 0.926 | 0.943 | 0.957 | 0.948 |
| spring barley | 2,066,130 | 0.771 | 0.805 | 0.776 | 0.829 | 0.864 | 0.845 |
| legumes harvested green | 1,736,038 | 0.490 | 0.543 | 0.518 | 0.525 | 0.565 | 0.540 |
| winter rapeseed rape | 1,331,873 | 0.971 | 0.971 | 0.970 | 0.971 | 0.974 | 0.973 |
| spring common soft wheat | 1,053,952 | 0.562 | 0.629 | 0.607 | 0.708 | 0.692 | 0.699 |
| oats | 798,055 | 0.548 | 0.581 | 0.547 | 0.654 | 0.728 | 0.672 |
| clover | 781,153 | 0.470 | 0.508 | 0.492 | 0.540 | 0.544 | 0.536 |
| peas | 593,851 | 0.843 | 0.843 | 0.825 | 0.882 | 0.898 | 0.895 |
| winter barley | 537,983 | 0.868 | 0.903 | 0.863 | 0.909 | 0.934 | 0.911 |
| beans | 358,922 | 0.800 | 0.825 | 0.803 | 0.862 | 0.882 | 0.865 |
| grain maize corn popcorn | 265,060 | 0.946 | 0.951 | 0.945 | 0.960 | 0.966 | 0.966 |
| alfalfa lucerne | 263,560 | 0.127 | 0.235 | 0.220 | 0.207 | 0.295 | 0.346 |
| rye | 239,000 | 0.691 | 0.708 | 0.704 | 0.820 | 0.864 | 0.834 |
| spring rapeseed rape | 185,156 | 0.746 | 0.748 | 0.729 | 0.776 | 0.836 | 0.806 |
| buckwheat | 122,327 | 0.845 | 0.843 | 0.837 | 0.890 | 0.889 | 0.876 |
| potatoes | 86,702 | 0.802 | 0.820 | 0.798 | 0.842 | 0.828 | 0.838 |
| legumes dried pulses protein crops | 58,763 | 0.084 | 0.022 | 0.032 | 0.000 | 0.129 | 0.114 |
| orchards fruits | 4,914 | 0.000 | 0.003 | 0.001 | 0.000 | 0.001 | 0.002 |
| fresh vegetables | 3,961 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 |
