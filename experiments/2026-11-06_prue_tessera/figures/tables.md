**Whole chip, no extent**

| Encoder | Fields from | PQ | SQ | RQ | PQ agn. | SQ agn. | RQ agn. | TP | FP | FN | fields |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE, fields grown into their rims | 0.367 | 0.772 | 0.429 | 0.374 | 0.813 | 0.460 | 23,642 | 23,622 | 31,933 | 47,264 |
| TESSERA frozen | PRUE, its own fields | 0.310 | 0.686 | 0.405 | 0.289 | 0.734 | 0.393 | 20,211 | 27,018 | 35,364 | 47,229 |
| TESSERA frozen | PRUE, predicted field unsplit | 0.082 | 0.667 | 0.105 | 0.100 | 0.760 | 0.131 | 4,647 | 10,502 | 50,928 | 15,149 |
| TESSERA frozen | PaPs, crop-map class | 0.322 | 0.734 | 0.393 | 0.336 | 0.790 | 0.425 | 22,034 | 25,962 | 33,541 | 47,996 |
| TESSERA frozen | PaPs, its own class | 0.303 | 0.695 | 0.370 | 0.336 | 0.790 | 0.425 | 22,034 | 25,962 | 33,541 | 47,996 |
| TESSERA frozen | boundary head (tau 0.5) | 0.145 | 0.702 | 0.186 | 0.048 | 0.743 | 0.064 | 13,118 | 338,546 | 42,457 | 351,664 |
| TESSERA fine-tuned | PRUE, fields grown into their rims | 0.397 | 0.725 | 0.463 | 0.378 | 0.815 | 0.463 | 24,905 | 27,044 | 30,670 | 51,949 |
| TESSERA fine-tuned | PRUE, its own fields | 0.336 | 0.653 | 0.435 | 0.294 | 0.740 | 0.397 | 21,328 | 30,583 | 34,247 | 51,911 |
| TESSERA fine-tuned | PRUE, predicted field unsplit | 0.086 | 0.670 | 0.109 | 0.096 | 0.766 | 0.126 | 4,481 | 11,197 | 51,094 | 15,678 |
| TESSERA fine-tuned | PaPs, crop-map class | 0.361 | 0.698 | 0.438 | 0.366 | 0.796 | 0.459 | 22,829 | 21,007 | 32,746 | 43,836 |
| TESSERA fine-tuned | PaPs, its own class | 0.357 | 0.697 | 0.432 | 0.366 | 0.796 | 0.459 | 22,829 | 21,007 | 32,746 | 43,836 |
| TESSERA fine-tuned | boundary head (tau 0.5) | 0.197 | 0.705 | 0.250 | 0.036 | 0.744 | 0.048 | 14,128 | 520,883 | 41,447 | 535,011 |
| ceiling (test labels) | perfect 3-class map, grown | 0.532 | 0.881 | 0.542 | 0.741 | 0.962 | 0.771 | 38,102 | 5,178 | 17,473 | 43,280 |
| ceiling (test labels) | perfect 3-class map, own | 0.408 | 0.720 | 0.508 | 0.494 | 0.762 | 0.647 | 32,001 | 11,279 | 23,574 | 43,280 |
| ceiling (test labels) | perfect 3-class map, unsplit | 0.239 | 0.759 | 0.268 | 0.391 | 0.908 | 0.430 | 15,929 | 2,550 | 39,646 | 18,479 |

**Cut to WorldCover 30 + 40**

| Encoder | Fields from | PQ | SQ | RQ | PQ agn. | SQ agn. | RQ agn. | TP | FP | FN | fields |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE, fields grown into their rims | 0.351 | 0.751 | 0.422 | 0.359 | 0.796 | 0.450 | 23,150 | 24,053 | 32,425 | 47,203 |
| TESSERA frozen | PRUE, its own fields | 0.302 | 0.680 | 0.398 | 0.283 | 0.730 | 0.387 | 19,876 | 27,275 | 35,699 | 47,151 |
| TESSERA frozen | PRUE, predicted field unsplit | 0.072 | 0.634 | 0.098 | 0.089 | 0.728 | 0.123 | 4,336 | 10,741 | 51,239 | 15,077 |
| TESSERA frozen | PaPs, crop-map class | 0.307 | 0.717 | 0.385 | 0.319 | 0.772 | 0.413 | 21,371 | 26,525 | 34,204 | 47,896 |
| TESSERA frozen | PaPs, its own class | 0.289 | 0.678 | 0.361 | 0.319 | 0.772 | 0.413 | 21,371 | 26,525 | 34,204 | 47,896 |
| TESSERA frozen | boundary head (tau 0.5) | 0.158 | 0.685 | 0.207 | 0.063 | 0.729 | 0.087 | 13,491 | 241,657 | 42,084 | 255,148 |
| TESSERA fine-tuned | PRUE, fields grown into their rims | 0.378 | 0.705 | 0.455 | 0.360 | 0.796 | 0.453 | 24,334 | 27,546 | 31,241 | 51,880 |
| TESSERA fine-tuned | PRUE, its own fields | 0.328 | 0.648 | 0.428 | 0.287 | 0.735 | 0.391 | 20,993 | 30,821 | 34,582 | 51,814 |
| TESSERA fine-tuned | PRUE, predicted field unsplit | 0.075 | 0.634 | 0.100 | 0.084 | 0.729 | 0.115 | 4,099 | 11,472 | 51,476 | 15,571 |
| TESSERA fine-tuned | PaPs, crop-map class | 0.345 | 0.681 | 0.430 | 0.348 | 0.779 | 0.447 | 22,205 | 21,556 | 33,370 | 43,761 |
| TESSERA fine-tuned | PaPs, its own class | 0.341 | 0.680 | 0.424 | 0.348 | 0.779 | 0.447 | 22,205 | 21,556 | 33,370 | 43,761 |
| TESSERA fine-tuned | boundary head (tau 0.5) | 0.204 | 0.692 | 0.262 | 0.083 | 0.739 | 0.112 | 15,050 | 197,406 | 40,525 | 212,456 |
| ceiling (test labels) | perfect 3-class map, grown | 0.501 | 0.844 | 0.535 | 0.690 | 0.923 | 0.747 | 36,886 | 6,245 | 18,689 | 43,131 |
| ceiling (test labels) | perfect 3-class map, own | 0.397 | 0.711 | 0.500 | 0.478 | 0.757 | 0.632 | 31,145 | 11,877 | 24,430 | 43,022 |
| ceiling (test labels) | perfect 3-class map, unsplit | 0.214 | 0.713 | 0.256 | 0.335 | 0.843 | 0.397 | 14,626 | 3,506 | 40,949 | 18,132 |

**Cut to WorldCover 30 + 40 + 90 (sensitivity)**

| Encoder | Fields from | PQ | SQ | RQ | PQ agn. | SQ agn. | RQ agn. | TP | FP | FN | fields |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE, fields grown into their rims | 0.351 | 0.751 | 0.422 | 0.359 | 0.796 | 0.451 | 23,203 | 24,031 | 32,372 | 47,234 |
| TESSERA frozen | PRUE, its own fields | 0.303 | 0.681 | 0.398 | 0.283 | 0.731 | 0.388 | 19,929 | 27,256 | 35,646 | 47,185 |
| TESSERA frozen | PRUE, predicted field unsplit | 0.072 | 0.634 | 0.098 | 0.090 | 0.727 | 0.124 | 4,373 | 10,734 | 51,202 | 15,107 |
| TESSERA frozen | PaPs, crop-map class | 0.307 | 0.717 | 0.385 | 0.319 | 0.772 | 0.414 | 21,394 | 26,507 | 34,181 | 47,901 |
| TESSERA frozen | PaPs, its own class | 0.289 | 0.678 | 0.361 | 0.319 | 0.772 | 0.414 | 21,394 | 26,507 | 34,181 | 47,901 |
| TESSERA frozen | boundary head (tau 0.5) | 0.158 | 0.685 | 0.207 | 0.064 | 0.729 | 0.088 | 13,490 | 237,201 | 42,085 | 250,691 |
| TESSERA fine-tuned | PRUE, fields grown into their rims | 0.379 | 0.705 | 0.455 | 0.361 | 0.795 | 0.454 | 24,394 | 27,517 | 31,181 | 51,911 |
| TESSERA fine-tuned | PRUE, its own fields | 0.328 | 0.648 | 0.428 | 0.288 | 0.735 | 0.392 | 21,041 | 30,812 | 34,534 | 51,853 |
| TESSERA fine-tuned | PRUE, predicted field unsplit | 0.075 | 0.634 | 0.100 | 0.085 | 0.728 | 0.116 | 4,135 | 11,468 | 51,440 | 15,603 |
| TESSERA fine-tuned | PaPs, crop-map class | 0.346 | 0.681 | 0.430 | 0.349 | 0.779 | 0.447 | 22,221 | 21,544 | 33,354 | 43,765 |
| TESSERA fine-tuned | PaPs, its own class | 0.341 | 0.680 | 0.424 | 0.349 | 0.779 | 0.447 | 22,221 | 21,544 | 33,354 | 43,765 |
| TESSERA fine-tuned | boundary head (tau 0.5) | 0.204 | 0.692 | 0.263 | 0.086 | 0.740 | 0.116 | 15,061 | 189,879 | 40,514 | 204,940 |
| ceiling (test labels) | perfect 3-class map, grown | 0.502 | 0.844 | 0.536 | 0.696 | 0.924 | 0.753 | 37,186 | 5,998 | 18,389 | 43,184 |
| ceiling (test labels) | perfect 3-class map, own | 0.397 | 0.711 | 0.501 | 0.482 | 0.758 | 0.637 | 31,423 | 11,685 | 24,152 | 43,108 |
| ceiling (test labels) | perfect 3-class map, unsplit | 0.215 | 0.713 | 0.257 | 0.339 | 0.845 | 0.401 | 14,791 | 3,369 | 40,784 | 18,160 |

**Cut to the declared parcels: upper bound, uses the test labels**

| Encoder | Fields from | PQ | SQ | RQ | PQ agn. | SQ agn. | RQ agn. | TP | FP | FN | fields |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE, fields grown into their rims | 0.401 | 0.814 | 0.442 | 0.487 | 0.872 | 0.559 | 25,614 | 10,458 | 29,961 | 36,072 |
| TESSERA frozen | PRUE, its own fields | 0.320 | 0.688 | 0.416 | 0.346 | 0.738 | 0.469 | 20,859 | 12,547 | 34,716 | 33,406 |
| TESSERA frozen | PRUE, predicted field unsplit | 0.095 | 0.699 | 0.116 | 0.142 | 0.812 | 0.174 | 5,634 | 3,378 | 49,941 | 9,012 |
| TESSERA frozen | PaPs, crop-map class | 0.345 | 0.753 | 0.410 | 0.435 | 0.820 | 0.531 | 22,885 | 7,704 | 32,690 | 30,589 |
| TESSERA frozen | PaPs, its own class | 0.324 | 0.714 | 0.385 | 0.435 | 0.820 | 0.531 | 22,885 | 7,704 | 32,690 | 30,589 |
| TESSERA frozen | boundary head (tau 0.5) | 0.307 | 0.813 | 0.340 | 0.476 | 0.903 | 0.527 | 27,354 | 20,801 | 28,221 | 48,155 |
| TESSERA fine-tuned | PRUE, fields grown into their rims | 0.440 | 0.775 | 0.481 | 0.511 | 0.882 | 0.579 | 27,108 | 10,944 | 28,467 | 38,052 |
| TESSERA fine-tuned | PRUE, its own fields | 0.348 | 0.656 | 0.448 | 0.358 | 0.744 | 0.482 | 21,878 | 13,409 | 33,697 | 35,287 |
| TESSERA fine-tuned | PRUE, predicted field unsplit | 0.104 | 0.716 | 0.123 | 0.144 | 0.826 | 0.174 | 5,616 | 3,213 | 49,959 | 8,829 |
| TESSERA fine-tuned | PaPs, crop-map class | 0.385 | 0.717 | 0.454 | 0.455 | 0.826 | 0.551 | 23,700 | 6,678 | 31,875 | 30,378 |
| TESSERA fine-tuned | PaPs, its own class | 0.380 | 0.716 | 0.448 | 0.455 | 0.826 | 0.551 | 23,700 | 6,678 | 31,875 | 30,378 |
| TESSERA fine-tuned | boundary head (tau 0.5) | 0.355 | 0.814 | 0.392 | 0.514 | 0.908 | 0.566 | 28,689 | 17,132 | 26,886 | 45,821 |
| ceiling (test labels) | perfect 3-class map, grown | 0.532 | 0.881 | 0.542 | 0.741 | 0.962 | 0.771 | 38,102 | 5,178 | 17,473 | 43,280 |
| ceiling (test labels) | perfect 3-class map, own | 0.408 | 0.720 | 0.508 | 0.494 | 0.762 | 0.647 | 32,001 | 11,279 | 23,574 | 43,280 |
| ceiling (test labels) | perfect 3-class map, unsplit | 0.239 | 0.759 | 0.268 | 0.391 | 0.908 | 0.430 | 15,929 | 2,550 | 39,646 | 18,479 |

**Per pixel, test chips** (three-class IoU on the pixels the target knows; boundary pixels inside the declared parcels)

| Encoder | IoU background | IoU interior | IoU boundary | field IoU | boundary F1 (in parcels) | boundary precision | boundary recall |
|---|---|---|---|---|---|---|---|
| TESSERA frozen | 0.984 | 0.901 | 0.524 | 0.963 | 0.700 | 0.733 | 0.670 |
| TESSERA fine-tuned | 0.986 | 0.910 | 0.567 | 0.968 | 0.740 | 0.771 | 0.712 |

**Share of the true parcels found, whole chip** (true parcels: isolated 15,518, touching other class 16,052, touching same class 24,005, < 0.5 ha 21,207, 0.5 to 2 ha 12,804, 2 to 5 ha 10,089, 5 to 10 ha 6,240, > 10 ha 5,235)

| Encoder | Fields from | all | isolated | touching other class | touching same class | < 0.5 ha | 0.5 to 2 ha | 2 to 5 ha | 5 to 10 ha | > 10 ha |
|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE grown | 0.425 | 0.471 | 0.489 | 0.353 | 0.083 | 0.518 | 0.626 | 0.728 | 0.839 |
| TESSERA frozen | PRUE own | 0.364 | 0.383 | 0.442 | 0.299 | 0.034 | 0.381 | 0.582 | 0.710 | 0.824 |
| TESSERA frozen | PaPs | 0.396 | 0.475 | 0.455 | 0.307 | 0.112 | 0.542 | 0.630 | 0.667 | 0.420 |
| TESSERA fine-tuned | PRUE grown | 0.448 | 0.487 | 0.509 | 0.383 | 0.096 | 0.560 | 0.652 | 0.750 | 0.850 |
| TESSERA fine-tuned | PRUE own | 0.384 | 0.394 | 0.460 | 0.326 | 0.039 | 0.420 | 0.611 | 0.734 | 0.835 |
| TESSERA fine-tuned | PaPs | 0.411 | 0.477 | 0.479 | 0.323 | 0.112 | 0.543 | 0.622 | 0.684 | 0.565 |

**What happened to the true parcels** (counts; merged as a share of each area bin)

| Encoder | Fields from | matched | merged | split | missed | merged, < 0.5 ha | merged, 0.5 to 2 ha | merged, 2 to 5 ha | merged, 5 to 10 ha | merged, > 10 ha |
|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PRUE grown | 23,642 | 21,540 | 1,149 | 9,244 | 0.516 | 0.388 | 0.334 | 0.244 | 0.143 |
| TESSERA frozen | PRUE own | 20,211 | 11,905 | 1,054 | 22,405 | 0.104 | 0.327 | 0.322 | 0.241 | 0.145 |
| TESSERA frozen | PaPs | 22,034 | 7,539 | 923 | 25,079 | 0.176 | 0.179 | 0.117 | 0.044 | 0.008 |
| TESSERA fine-tuned | PRUE grown | 24,905 | 21,839 | 1,177 | 7,654 | 0.564 | 0.366 | 0.309 | 0.223 | 0.129 |
| TESSERA fine-tuned | PRUE own | 21,328 | 10,804 | 1,156 | 22,287 | 0.097 | 0.298 | 0.292 | 0.212 | 0.128 |
| TESSERA fine-tuned | PaPs | 22,829 | 7,865 | 791 | 24,090 | 0.177 | 0.176 | 0.132 | 0.068 | 0.021 |

**The class of the parcels found** (the crop-type map's vote against the true class)

| Encoder | Fields from | parcels found | crop-map vote accuracy | crop-map vote macro F1 |
|---|---|---|---|---|
| TESSERA frozen | PRUE grown | 23,642 | 0.816 | 0.647 |
| TESSERA frozen | PRUE own | 20,211 | 0.814 | 0.658 |
| TESSERA frozen | PaPs | 22,034 | 0.814 | 0.642 |
| TESSERA fine-tuned | PRUE grown | 24,905 | 0.846 | 0.684 |
| TESSERA fine-tuned | PRUE own | 21,328 | 0.847 | 0.696 |
| TESSERA fine-tuned | PaPs | 22,829 | 0.846 | 0.686 |

**Fits**

| Encoder | train chips | best epoch | val object F1 (own) | val PQ agn. own | val PQ agn. grown | best val PQ agn. grown (epoch) | fit h | peak GB | trainable M | commit | PaPs best val PQ agn. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | 4,898 | 21 | 0.401 | 0.294 | 0.382 | 0.382 (21) | 0.562 | 2.137 | 67.168 | 647f4b66 | 0.345 |
| TESSERA fine-tuned | 4,898 | 25 | 0.407 | 0.300 | 0.386 | 0.390 (14) | 28.273 | 35.085 | 88.199 | 647f4b66 | 0.373 |
