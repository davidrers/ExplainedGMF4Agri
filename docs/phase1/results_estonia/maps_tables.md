# Tables of the maps of REPORT.md section 3.5

Written by [maps.py](maps.py). One draw, one seed per fit; the shares of a place come from its parcels alone and illustrate the maps, they are not test scores.

## Share of the scored parcels mapped right, K = 100 / 20 / 5 %

| Model | block 583 457 | block 593 445 | block 594 443 | tile EE 02332 01830 | tile EE 02372 01780 |
|---|---|---|---|---|---|
| TerraMind v1 L, frozen | 0.75 / 0.71 / 0.66 | 0.68 / 0.62 / 0.52 | 0.77 / 0.74 / 0.70 | 0.77 / 0.77 / 0.71 | 0.72 / 0.53 / 0.33 |
| TerraMind v1 L, fine-tuned | 0.75 / 0.74 / 0.68 | 0.66 / 0.61 / 0.53 | 0.74 / 0.76 / 0.71 | 0.77 / 0.65 / 0.58 | 0.53 / 0.49 / 0.30 |
| THOR v1 L, frozen | 0.72 / 0.69 / 0.65 | 0.62 / 0.54 / 0.50 | 0.76 / 0.73 / 0.72 | 0.68 / 0.65 / 0.65 | 0.58 / 0.40 / 0.30 |
| THOR v1 L, fine-tuned | 0.72 / 0.68 / 0.64 | 0.62 / 0.57 / 0.48 | 0.72 / 0.65 / 0.72 | 0.68 / 0.71 / 0.55 | 0.67 / 0.42 / 0.35 |
| Prithvi-EO-2.0 600M, frozen | 0.72 / 0.72 / 0.69 | 0.61 / 0.56 / 0.54 | 0.76 / 0.72 / 0.67 | 0.77 / 0.74 / 0.68 | 0.56 / 0.35 / 0.30 |
| Prithvi-EO-2.0 600M, fine-tuned | 0.70 / 0.77 / 0.66 | 0.63 / 0.60 / 0.57 | 0.76 / 0.75 / 0.72 | 0.71 / 0.77 / 0.68 | 0.56 / 0.44 / 0.23 |
| TESSERA v2 M, frozen | 0.77 / 0.74 / 0.74 | 0.71 / 0.66 / 0.68 | 0.79 / 0.76 / 0.75 | 0.81 / 0.74 / 0.77 | 0.77 / 0.77 / 0.70 |
| TESSERA v2 M, fine-tuned | 0.79 / 0.77 / 0.72 | 0.74 / 0.68 / 0.68 | 0.79 / 0.77 / 0.77 | 0.77 / 0.81 / 0.68 | 0.84 / 0.65 / 0.84 |
| AlphaEarth v1, frozen | 0.70 / 0.66 / 0.64 | 0.62 / 0.60 / 0.60 | 0.74 / 0.75 / 0.74 | 0.71 / 0.74 / 0.68 | 0.72 / 0.65 / 0.67 |
| U-Net from scratch | 0.74 / 0.75 / 0.71 | 0.68 / 0.65 / 0.54 | 0.77 / 0.77 / 0.75 | 0.81 / 0.74 / 0.77 | 0.63 / 0.70 / 0.40 |

## Test pixels outside any declared parcel given a crop other than grassland, all test tiles

| Model | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind v1 L, frozen | 0.616 | 0.613 | 0.689 |
| TerraMind v1 L, fine-tuned | 0.520 | 0.412 | 0.673 |
| THOR v1 L, frozen | 0.499 | 0.553 | 0.570 |
| THOR v1 L, fine-tuned | 0.571 | 0.671 | 0.609 |
| Prithvi-EO-2.0 600M, frozen | 0.473 | 0.588 | 0.683 |
| Prithvi-EO-2.0 600M, fine-tuned | 0.564 | 0.634 | 0.641 |
| TESSERA v2 M, frozen | 0.130 | 0.219 | 0.483 |
| TESSERA v2 M, fine-tuned | 0.030 | 0.028 | 0.024 |
| AlphaEarth v1, frozen | 0.239 | 0.316 | 0.273 |
| U-Net from scratch | 0.324 | 0.380 | 0.238 |
| TESSERA v2 M, frozen, labelled-pixel route (control) | 0.028 | 0.028 | 0.028 |
