**Whole chip, no extent**

| Encoder | Parcels from | PQ | SQ | RQ | PQ agn. | TP | FP | FN | parcels |
|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PaPs, its own class | 0.303 | 0.695 | 0.370 | 0.336 | 22,034 | 25,962 | 33,541 | 47,996 |
| TESSERA frozen | PaPs parcels, crop-map class | 0.322 | 0.734 | 0.393 | 0.336 | 22,034 | 25,962 | 33,541 | 47,996 |
| TESSERA frozen | boundary head (tau 0.5) | 0.145 | 0.702 | 0.186 | 0.048 | 13,118 | 338,546 | 42,457 | 351,664 |
| TESSERA frozen | crop-map regions | 0.004 | 0.643 | 0.005 | 0.005 | 9,427 | 2,692,554 | 46,148 | 2,701,981 |
| TESSERA frozen | extent alone | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 1,475 | 55,575 | 1,475 |
| TESSERA fine-tuned | PaPs, its own class | 0.357 | 0.697 | 0.432 | 0.366 | 22,829 | 21,007 | 32,746 | 43,836 |
| TESSERA fine-tuned | PaPs parcels, crop-map class | 0.361 | 0.698 | 0.438 | 0.366 | 22,829 | 21,007 | 32,746 | 43,836 |

**Cut to WorldCover 30 + 40**

| Encoder | Parcels from | PQ | SQ | RQ | PQ agn. | TP | FP | FN | parcels |
|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PaPs, its own class | 0.289 | 0.678 | 0.361 | 0.319 | 21,371 | 26,525 | 34,204 | 47,896 |
| TESSERA frozen | PaPs parcels, crop-map class | 0.307 | 0.717 | 0.385 | 0.319 | 21,371 | 26,525 | 34,204 | 47,896 |
| TESSERA frozen | boundary head (tau 0.5) | 0.158 | 0.685 | 0.207 | 0.063 | 13,491 | 241,657 | 42,084 | 255,148 |
| TESSERA frozen | crop-map regions | 0.022 | 0.638 | 0.028 | 0.020 | 11,382 | 778,479 | 44,193 | 789,861 |
| TESSERA frozen | extent alone | 0.023 | 0.565 | 0.033 | 0.016 | 1,817 | 98,469 | 53,758 | 100,286 |
| TESSERA fine-tuned | PaPs, its own class | 0.341 | 0.680 | 0.424 | 0.348 | 22,205 | 21,556 | 33,370 | 43,761 |
| TESSERA fine-tuned | PaPs parcels, crop-map class | 0.345 | 0.681 | 0.430 | 0.348 | 22,205 | 21,556 | 33,370 | 43,761 |

**Cut to WorldCover 30 + 40 + 90 (sensitivity)**

| Encoder | Parcels from | PQ | SQ | RQ | PQ agn. | TP | FP | FN | parcels |
|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PaPs, its own class | 0.289 | 0.678 | 0.361 | 0.319 | 21,394 | 26,507 | 34,181 | 47,901 |
| TESSERA frozen | PaPs parcels, crop-map class | 0.307 | 0.717 | 0.385 | 0.319 | 21,394 | 26,507 | 34,181 | 47,901 |
| TESSERA frozen | boundary head (tau 0.5) | 0.158 | 0.685 | 0.207 | 0.064 | 13,490 | 237,201 | 42,085 | 250,691 |
| TESSERA frozen | crop-map regions | 0.021 | 0.638 | 0.027 | 0.020 | 11,378 | 774,225 | 44,197 | 785,603 |
| TESSERA frozen | extent alone | 0.023 | 0.566 | 0.033 | 0.017 | 1,806 | 88,871 | 53,769 | 90,677 |
| TESSERA fine-tuned | PaPs, its own class | 0.341 | 0.680 | 0.424 | 0.349 | 22,221 | 21,544 | 33,354 | 43,765 |
| TESSERA fine-tuned | PaPs parcels, crop-map class | 0.346 | 0.681 | 0.430 | 0.349 | 22,221 | 21,544 | 33,354 | 43,765 |

**Cut to the declared parcels: upper bound, uses the test labels**

| Encoder | Parcels from | PQ | SQ | RQ | PQ agn. | TP | FP | FN | parcels |
|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PaPs, its own class | 0.324 | 0.714 | 0.385 | 0.435 | 22,885 | 7,704 | 32,690 | 30,589 |
| TESSERA frozen | PaPs parcels, crop-map class | 0.345 | 0.753 | 0.410 | 0.435 | 22,885 | 7,704 | 32,690 | 30,589 |
| TESSERA frozen | boundary head (tau 0.5) | 0.307 | 0.813 | 0.340 | 0.476 | 27,354 | 20,801 | 28,221 | 48,155 |
| TESSERA frozen | crop-map regions | 0.052 | 0.701 | 0.060 | 0.085 | 28,946 | 507,618 | 26,629 | 536,564 |
| TESSERA frozen | extent alone | 0.251 | 0.768 | 0.279 | 0.486 | 20,588 | 2,558 | 34,987 | 23,146 |
| TESSERA fine-tuned | PaPs, its own class | 0.380 | 0.716 | 0.448 | 0.455 | 23,700 | 6,678 | 31,875 | 30,378 |
| TESSERA fine-tuned | PaPs parcels, crop-map class | 0.385 | 0.717 | 0.454 | 0.455 | 23,700 | 6,678 | 31,875 | 30,378 |

**The class of the parcels PaPs found**

| Encoder | parcels PaPs found | class head accuracy | class head macro F1 | crop-map vote accuracy | crop-map vote macro F1 | crop map pixel Macro-F1 |
|---|---|---|---|---|---|---|
| TESSERA frozen | 22,034 | 0.797 | 0.603 | 0.814 | 0.642 | 0.618 |
| TESSERA fine-tuned | 22,829 | 0.838 | 0.674 | 0.846 | 0.686 | 0.660 |

**Share of the true parcels found** (true parcels: isolated 15,518, touching other class 16,052, touching same class 24,005, < 0.5 ha 21,207, 0.5 to 2 ha 12,804, 2 to 5 ha 10,089, 5 to 10 ha 6,240, > 10 ha 5,235)

| Encoder | Parcels from | all | isolated | touching other class | touching same class | < 0.5 ha | 0.5 to 2 ha | 2 to 5 ha | 5 to 10 ha | > 10 ha |
|---|---|---|---|---|---|---|---|---|---|---|
| TESSERA frozen | PaPs, whole chip | 0.396 | 0.475 | 0.455 | 0.307 | 0.112 | 0.542 | 0.630 | 0.667 | 0.420 |
| TESSERA frozen | boundary head, whole chip | 0.236 | 0.384 | 0.229 | 0.145 | 0.078 | 0.308 | 0.315 | 0.337 | 0.426 |
| TESSERA frozen | boundary head, WorldCover | 0.243 | 0.388 | 0.236 | 0.153 | 0.069 | 0.310 | 0.332 | 0.367 | 0.462 |
| TESSERA frozen | boundary head, declared parcels (upper bound) | 0.492 | 0.979 | 0.339 | 0.280 | 0.439 | 0.524 | 0.486 | 0.517 | 0.613 |
| TESSERA fine-tuned | PaPs, whole chip | 0.411 | 0.477 | 0.479 | 0.323 | 0.112 | 0.543 | 0.622 | 0.684 | 0.565 |

**Class-aware RQ (parcel Macro-F1 at IoU > 0.5), whole chip**

| Encoder | true parcels, crop-map class | PaPs, crop-map class | PaPs, its own class | boundary head |
|---|---|---|---|---|
| TESSERA frozen | 0.551 | 0.393 | 0.370 | 0.186 |
| TESSERA fine-tuned | 0.581 | 0.438 | 0.432 |  |

**Fits**

| Encoder | train chips | best epoch | val PQ | confidence | fit h | peak GB | commit |
|---|---|---|---|---|---|---|---|
| TESSERA frozen | 4,898 | 90 | 0.31 | 0.07 | 5.12 | 1.70 | c1d196d6 |
| TESSERA fine-tuned | 4,898 | 70 | 0.37 | 0.09 | 5.14 | 1.69 | c1d196d6 |
