**Border precision, recall and F1** over the 1,475 test chips, by tolerance

| Method | Encoder | precision, 0 px | recall, 0 px | F1, 0 px | precision, 1 px | recall, 1 px | F1, 1 px | precision, 2 px | recall, 2 px | F1, 2 px |
|---|---|---|---|---|---|---|---|---|---|---|
| PRUE | frozen | 0.767 | 0.456 | 0.572 | 0.947 | 0.790 | 0.862 | 0.962 | 0.849 | 0.902 |
| PRUE | fine-tuned | 0.784 | 0.475 | 0.592 | 0.938 | 0.807 | 0.868 | 0.954 | 0.865 | 0.908 |
| PaPs | frozen | 0.560 | 0.272 | 0.367 | 0.851 | 0.516 | 0.642 | 0.907 | 0.607 | 0.727 |
| PaPs | fine-tuned | 0.572 | 0.298 | 0.392 | 0.863 | 0.557 | 0.677 | 0.916 | 0.646 | 0.758 |
| boundary head | frozen | 0.130 | 0.268 | 0.175 | 0.267 | 0.555 | 0.361 | 0.327 | 0.634 | 0.432 |
| boundary head | fine-tuned | 0.060 | 0.261 | 0.098 | 0.126 | 0.570 | 0.207 | 0.161 | 0.654 | 0.258 |

**Recall at 1 px by what lies across the real border** (share of the real border pixels)

| Method | Encoder | same crop (12.0%) | other crop (13.6%) | undeclared farmland (52.7%) | non-farm land (21.7%) |
|---|---|---|---|---|---|
| PRUE | frozen | 0.344 | 0.820 | 0.845 | 0.883 |
| PRUE | fine-tuned | 0.377 | 0.841 | 0.856 | 0.905 |
| PaPs | frozen | 0.278 | 0.670 | 0.525 | 0.528 |
| PaPs | fine-tuned | 0.293 | 0.729 | 0.566 | 0.571 |
| boundary head | frozen | 0.152 | 0.316 | 0.588 | 0.846 |
| boundary head | fine-tuned | 0.153 | 0.415 | 0.600 | 0.823 |

**Where the drawn borders lie, at 1 px** (shares of all drawn border pixels that are scored)

| Method | Encoder | drawn border px | real | false, on a declared parcel | false, on non-farm land |
|---|---|---|---|---|---|
| PRUE | frozen | 1,856,212 | 0.947 | 0.036 | 0.017 |
| PRUE | fine-tuned | 1,892,754 | 0.938 | 0.041 | 0.021 |
| PaPs | frozen | 1,518,553 | 0.851 | 0.114 | 0.035 |
| PaPs | fine-tuned | 1,626,189 | 0.863 | 0.108 | 0.028 |
| boundary head | frozen | 6,443,668 | 0.267 | 0.005 | 0.728 |
| boundary head | fine-tuned | 13,480,285 | 0.126 | 0.003 | 0.871 |

**Chip by chip** (1,008 test chips with at least 20 declared parcels), F1 at 1 px

| Method | Encoder | median | 25th pct | 75th pct |
|---|---|---|---|---|
| PRUE | frozen | 0.875 | 0.828 | 0.910 |
| PRUE | fine-tuned | 0.882 | 0.834 | 0.915 |
| PaPs | frozen | 0.659 | 0.578 | 0.731 |
| PaPs | fine-tuned | 0.696 | 0.616 | 0.757 |
| boundary head | frozen | 0.428 | 0.326 | 0.510 |
| boundary head | fine-tuned | 0.275 | 0.188 | 0.379 |

PRUE above PaPs (frozen) on 99.4% of these chips.

PRUE above PaPs (fine-tuned) on 99.2% of these chips.

**Example chips**

| Figure | Chip | Why |
|---|---|---|
| example_1 | EE_02327_01812 | PRUE at its 90th percentile |
| example_2 | EE_02302_01815 | PRUE at its median |
| example_3 | EE_02263_01778 | PRUE at its 10th percentile |
| example_4 | EE_02270_01785 | where PaPs beats PRUE most |
