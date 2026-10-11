# Tables

30 chips, 3,346 acquisitions, 2,468 distinct overpasses, 49 reprocessed at baseline 04.00, 1 unreadable.

## Share of labelled pixels gap-filled

| variant | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| export | 97.8% | 65.4% | 21.1% | 0.0% | 0.0% | 0.0% | 0.0% | 23.1% | 0.7% | 14.5% | 19.3% | 78.9% |
| export_merged | 97.9% | 66.0% | 21.7% | 0.0% | 0.0% | 0.0% | 0.0% | 23.2% | 0.7% | 14.5% | 20.3% | 82.4% |
| revised | 99.9% | 86.9% | 55.2% | 0.3% | 0.2% | 0.0% | 0.0% | 35.1% | 3.0% | 22.2% | 42.9% | 99.8% |
| revised_no_haze | 99.9% | 86.9% | 54.9% | 0.3% | 0.2% | 0.0% | 0.0% | 34.9% | 2.8% | 22.2% | 42.8% | 99.1% |
| revised_keep7 | 99.6% | 75.9% | 36.1% | 0.1% | 0.2% | 0.0% | 0.0% | 30.3% | 1.8% | 18.2% | 29.7% | 96.5% |
| revised_27_near | 99.7% | 76.0% | 36.2% | 0.0% | 0.0% | 0.0% | 0.0% | 30.2% | 1.7% | 18.3% | 27.4% | 95.0% |
| revised_keep2 | 99.9% | 83.5% | 45.8% | 0.1% | 0.0% | 0.0% | 0.0% | 32.2% | 2.1% | 20.5% | 35.2% | 97.9% |

## Clear dates per labelled pixel

| variant | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| export | 0.02 | 0.41 | 1.60 | 4.60 | 5.06 | 8.31 | 7.22 | 2.33 | 3.62 | 2.68 | 1.57 | 0.26 |
| export_merged | 0.02 | 0.39 | 1.57 | 4.57 | 5.04 | 8.26 | 7.17 | 2.31 | 3.62 | 2.66 | 1.55 | 0.22 |
| revised | 0.00 | 0.14 | 0.66 | 3.73 | 4.39 | 7.83 | 6.41 | 1.75 | 3.00 | 2.30 | 0.60 | 0.00 |
| revised_no_haze | 0.00 | 0.14 | 0.66 | 3.75 | 4.40 | 7.86 | 6.46 | 1.76 | 3.02 | 2.30 | 0.60 | 0.01 |
| revised_keep7 | 0.00 | 0.27 | 1.10 | 4.08 | 4.58 | 7.91 | 6.64 | 1.96 | 3.21 | 2.50 | 1.16 | 0.04 |
| revised_27_near | 0.00 | 0.26 | 1.06 | 4.17 | 4.64 | 7.89 | 6.62 | 1.96 | 3.21 | 2.54 | 1.20 | 0.05 |
| revised_keep2 | 0.00 | 0.17 | 0.80 | 3.92 | 4.53 | 7.84 | 6.46 | 1.82 | 3.10 | 2.38 | 0.76 | 0.02 |

## Seasons

| variant | filled Apr-Sep | white per 10k obs Apr-Sep | dips per 1k obs Apr-Sep | filled Oct-Mar | white per 10k obs Oct-Mar | dips per 1k obs Oct-Mar |
|---|---|---|---|---|---|---|
| export | 4.0% | 3.9 | 52.1 | 49.5% | 342.2 | 16.1 |
| export_merged | 4.0% | 3.9 | 51.8 | 50.4% | 261.2 | 15.0 |
| revised | 6.4% | 0.0 | 44.1 | 67.8% | 0.2 | 6.7 |
| revised_no_haze | 6.4% | 0.2 | 44.3 | 67.6% | 3.6 | 6.7 |
| revised_keep7 | 5.4% | 0.0 | 46.6 | 59.3% | 0.2 | 10.6 |
| revised_27_near | 5.3% | 0.0 | 47.0 | 58.8% | 0.2 | 9.5 |
| revised_keep2 | 5.8% | 0.0 | 46.5 | 63.8% | 0.2 | 7.1 |

## Per chip, April to September, minus revised: mean ± standard deviation over chips

| variant | filled, points | dips per 1k | chips with more dips |
|---|---|---|---|
| export | -2.48 ± 3.29 | +8.5 ± 16.0 | 24/30 |
| export_merged | -2.47 ± 3.28 | +8.2 ± 15.9 | 24/30 |
| revised_no_haze | -0.06 ± 0.12 | +0.2 ± 0.4 | 20/30 |
| revised_keep7 | -1.08 ± 1.51 | +3.0 ± 6.4 | 22/30 |
| revised_27_near | -1.09 ± 1.44 | +3.1 ± 6.8 | 22/30 |
| revised_keep2 | -0.62 ± 1.00 | +2.2 ± 6.0 | 21/30 |

## Share of the observations export keeps that each rule removes, labelled pixels

`removed`: all the rule removes; `only`: what no other revised rule removes.

| rule | removed | only | removed Apr-Sep | only Apr-Sep |
|---|---|---|---|---|
| class_2 | 3.7% | 2.7% | 2.4% | 1.8% |
| class_7 | 11.7% | 7.2% | 7.8% | 4.1% |
| buffer | 5.9% | 1.3% | 4.8% | 1.3% |
| snow | 1.1% | 0.0% | 0.2% | 0.0% |
| haze | 4.4% | 0.5% | 3.7% | 0.5% |
| merge | 0.6% | 0.6% | 0.3% | 0.3% |
| class_2_near | 1.6% | 0.6% | 1.0% | 0.4% |
| class_2_far | 2.1% | 2.1% | 1.4% | 1.4% |
| class_7_near | 5.8% | 1.8% | 4.4% | 1.1% |
| class_7_far | 6.0% | 5.4% | 3.5% | 3.0% |

## Observations removed by one rule only, against the pixel's clean median

| rule | season | n | dB02 median | share dB02 > 0.02 | dNIR median | share dNIR < -0.25 | dNDVI median | dNDVI q25 |
|---|---|---|---|---|---|---|---|---|
| buffer | Apr-Sep | 334,394 | +0.033 | 55% | -10% | 43% | -0.115 | -0.295 |
| buffer | Oct-Mar | 28,336 | +0.012 | 40% | +6% | 16% | -0.035 | -0.085 |
| class_2 | Apr-Sep | 438,304 | -0.018 | 8% | -55% | 90% | -0.035 | -0.115 |
| class_2 | Oct-Mar | 123,020 | -0.003 | 21% | -45% | 74% | -0.035 | -0.135 |
| class_2_far | Apr-Sep | 329,732 | -0.018 | 8% | -54% | 90% | -0.035 | -0.105 |
| class_2_far | Oct-Mar | 93,625 | -0.003 | 20% | -43% | 73% | -0.035 | -0.125 |
| class_2_near | Apr-Sep | 108,572 | -0.018 | 10% | -56% | 92% | -0.045 | -0.135 |
| class_2_near | Oct-Mar | 29,395 | -0.003 | 25% | -50% | 79% | -0.045 | -0.145 |
| class_7 | Apr-Sep | 1,025,824 | +0.022 | 53% | -24% | 50% | -0.085 | -0.225 |
| class_7 | Oct-Mar | 569,945 | +0.022 | 56% | -2% | 20% | -0.055 | -0.135 |
| class_7_far | Apr-Sep | 750,758 | +0.017 | 48% | -26% | 50% | -0.075 | -0.195 |
| class_7_far | Oct-Mar | 467,399 | +0.022 | 54% | -2% | 18% | -0.045 | -0.115 |
| class_7_near | Apr-Sep | 275,066 | +0.042 | 68% | -24% | 48% | -0.145 | -0.285 |
| class_7_near | Oct-Mar | 102,546 | +0.037 | 67% | -3% | 30% | -0.095 | -0.185 |
| haze | Apr-Sep | 137,075 | +0.133 | 100% | +32% | 4% | -0.185 | -0.315 |
| haze | Oct-Mar | 2,192 | +0.143 | 100% | +52% | 7% | -0.215 | -0.345 |
| kept | Apr-Sep | 23,096,054 | +0.003 | 11% | +1% | 8% | -0.005 | -0.055 |
| kept | Oct-Mar | 2,899,041 | +0.003 | 3% | +1% | 2% | +0.005 | -0.015 |
| merge | Apr-Sep | 74,017 | +0.027 | 56% | -7% | 34% | -0.085 | -0.215 |
| merge | Oct-Mar | 28,899 | +0.017 | 29% | +6% | 6% | -0.045 | -0.065 |
| snow | Apr-Sep | 1,663 | +0.033 | 65% | -26% | 52% | -0.025 | -0.185 |
| snow | Oct-Mar | 1,812 | +0.097 | 97% | +98% | 7% | +0.075 | -0.025 |
