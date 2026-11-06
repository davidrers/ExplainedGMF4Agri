**declared parcels (oracle)**: boundary parcels at tau 0.5

| Arm | floor PQ agn. | crop map PQ agn. | PQ agn. | SQ agn. | RQ agn. | PQ | TP | FP | FN | parcels | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA v2, fine-tuned | 0.486 | 0.099 | 0.514 | 0.908 | 0.566 | 0.355 | 28,689 | 17,132 | 26,886 | 45,821 | 0 |
| TESSERA v2, frozen | 0.486 | 0.085 | 0.476 | 0.903 | 0.527 | 0.307 | 27,354 | 20,801 | 28,221 | 48,155 | 0 |

**WorldCover 30 + 40**: boundary parcels at tau 0.5

| Arm | floor PQ agn. | crop map PQ agn. | PQ agn. | SQ agn. | RQ agn. | PQ | TP | FP | FN | parcels | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA v2, fine-tuned | 0.016 | 0.023 | 0.083 | 0.739 | 0.112 | 0.204 | 15,050 | 197,406 | 40,525 | 212,456 | 1,622 |
| TESSERA v2, frozen | 0.016 | 0.020 | 0.063 | 0.729 | 0.087 | 0.158 | 13,491 | 241,657 | 42,084 | 255,148 | 1,486 |

**whole chip**: boundary parcels at tau 0.5

| Arm | floor PQ agn. | crop map PQ agn. | PQ agn. | SQ agn. | RQ agn. | PQ | TP | FP | FN | parcels | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA v2, fine-tuned | 0.000 | 0.017 | 0.036 | 0.744 | 0.048 | 0.197 | 14,128 | 520,883 | 41,447 | 535,011 | 1,480 |
| TESSERA v2, frozen | 0.000 | 0.005 | 0.048 | 0.743 | 0.064 | 0.145 | 13,118 | 338,546 | 42,457 | 351,664 | 1,420 |

**WorldCover 30 + 40 + 90**: boundary parcels at tau 0.5

| Arm | floor PQ agn. | crop map PQ agn. | PQ agn. | SQ agn. | RQ agn. | PQ | TP | FP | FN | parcels | dropped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA v2, fine-tuned | 0.017 | 0.023 | 0.086 | 0.740 | 0.116 | 0.204 | 15,061 | 189,879 | 40,514 | 204,940 | 1,609 |
| TESSERA v2, frozen | 0.017 | 0.020 | 0.064 | 0.729 | 0.088 | 0.158 | 13,490 | 237,201 | 42,085 | 250,691 | 1,472 |

**Class-agnostic PQ against the threshold**

| arm | extent | boundary@0.3 | boundary@0.5 | boundary@0.7 |
|---|---|---|---|---|
| TESSERA v2, frozen | chip | 0.045 | 0.048 | 0.041 |
| TESSERA v2, frozen | oracle | 0.466 | 0.476 | 0.466 |
| TESSERA v2, frozen | worldcover | 0.078 | 0.063 | 0.047 |
| TESSERA v2, fine-tuned | chip | 0.108 | 0.036 | 0.028 |
| TESSERA v2, fine-tuned | oracle | 0.506 | 0.514 | 0.498 |
| TESSERA v2, fine-tuned | worldcover | 0.104 | 0.083 | 0.054 |

**The extents on the test chips**

| extent | true_pixels_inside | share_of_chip | background_share_of_extent | void_share_of_extent |
|---|---|---|---|---|
| oracle | 1.000 | 0.262 | 0.000 | 0.000 |
| worldcover | 0.958 | 0.393 | 0.335 | 0.026 |
| worldcover_30_40_90 | 0.968 | 0.412 | 0.361 | 0.025 |
| chip | 1.000 | 1.000 | 0.728 | 0.011 |

**The two heads** (boundary pixels at 0.5 inside E; crop-type pixel Macro-F1)

| Arm | pixel Macro-F1 | boundary F1 in E | precision | recall |
|---|---|---|---|---|
| TESSERA v2, fine-tuned | 0.671 | 0.665 | 0.772 | 0.584 |
| TESSERA v2, frozen | 0.618 | 0.639 | 0.754 | 0.555 |

**WorldCover on the test chips**: parcel pixels inside 30 + 40: 0.958, with 90: 0.968; share of WorldCover grassland on background 0.570, of cropland 0.066.
