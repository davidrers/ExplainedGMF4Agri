# Border accuracy of PRUE, PaPs and the boundary head: report, 7 November 2026

## The answer

**Is a drawn border real, and is each real border drawn?** Over the 1,475 test chips, with a tolerance of one pixel
(10 m):

| Method | Encoder | Precision: drawn borders that are real | Recall: real borders that are drawn | F1 |
|---|---|---|---|---|
| **PRUE** | frozen | **0.947** | **0.790** | **0.862** |
| **PRUE** | fine-tuned | 0.938 | **0.807** | **0.868** |
| PaPs | frozen | 0.851 | 0.516 | 0.642 |
| PaPs | fine-tuned | 0.863 | 0.557 | 0.677 |
| Boundary head | frozen | 0.267 | 0.555 | 0.361 |
| Boundary head | fine-tuned | 0.126 | 0.570 | 0.207 |

* **PRUE** matches the declared borders best. Of the borders it draws, 95 % are real. Of the real borders, it draws
  about 80 %. The fifth it misses lies mostly between two parcels of the same crop.
* **PaPs** draws real borders too (85 to 86 %), but only half of the real ones, because it leaves whole parcels
  out: a parcel whose centre it does not detect has no borders at all.
* **The boundary head** draws borders everywhere, across forest, wetland and villages: three quarters of what it
  draws (frozen) and seven in eight (fine-tuned) lie on non-farm land.
* **Fine-tuning the encoder** changes PRUE by +0.006 F1, PaPs by +0.035, and lowers the boundary head's F1 by 0.15.
* **Chip by chip**, PRUE is above PaPs on 99 % of the 1,008 test chips with at least 20 declared parcels.
* **EuroCrops labels only part of the farmland in a chip** (a third of WorldCover's farmland is undeclared), and
  none of the three methods is trained as if the chip were complete: each learns from the declared parcels and from
  WorldCover's non-farm land, and ignores the rest. The one place where the pipeline does treat a chip as complete
  is the parcel score (PQ) and the validation scores that chose the checkpoints, which count a field on undeclared
  farmland as false. The border score of this report does not ([EuroCrops' partial labels](#eurocrops-partial-labels-and-the-three-methods)).

## The question and how it is measured

The question is the delineation alone, crop type left out: does the delineation a method predicts match the
declared one? It is measured on the borders, as in segmentation work (the boundary F-score of Csurka et al., 2013):

1. **The real borders.** A pixel of a declared parcel is a border pixel when one of its four neighbours belongs to
   another parcel or to no parcel. Between two parcels the border is two pixels wide, one on each side; at an outer
   edge it is one pixel wide. The chip edge is no border.
2. **The drawn borders.** Each method's final fields, the output it delivers, built over the whole chip with no test
   label. Their borders are marked the same way.
3. **Precision**: the share of the drawn border pixels with a real border pixel within one pixel. **Recall**: the
   share of the real border pixels with a drawn border pixel within one pixel. **F1**: their harmonic mean.
4. **The tolerance of one pixel** (any direction, 10 m) accepts a border drawn one pixel beside the real one. Without
   it, such a border counts once as false and once as missed. The scores at 0 and 2 pixels are given below.
5. **Scored land**: the declared parcels and the land WorldCover calls non-agricultural (forest, water, built-up,
   wetland). **Not scored**: undeclared land that WorldCover calls grassland or cropland, where nobody says whether a
   field lies, and declared parcels of crops outside the 20 classes.

As a check, the declared parcels taken as the prediction score exactly 1. Every method is scored on the same 1,475
test chips (the scorer stops if the chip lists differ) against the same real borders.

The final fields of each method, from its stored output:

| Method | Output per pixel | Final fields |
|---|---|---|
| PRUE | three probabilities: background, interior, boundary | each patch of interior pixels, grown back over the boundary ring around it |
| PaPs | centre heatmap, saliency; per detected centre a box size, a 16 x 16 shape and a class | each centre's shape placed in its box; a pixel is in one parcel or in none |
| Boundary head | crop class and border probability | cores where the border probability is below 0.5, grown until they meet; over the whole chip |

## Results

### Precision, recall and F1

![Precision, recall and F1 at a tolerance of one pixel](figures/scores.png)

Bars: method by colour, fine-tuned hatched. The tolerance matters most from 0 to 1 pixel; from 1 to 2 it adds
0.04 to 0.09:

| Method | Encoder | F1, exact pixel | F1, 1 pixel | F1, 2 pixels |
|---|---|---|---|---|
| PRUE | frozen | 0.572 | 0.862 | 0.902 |
| PRUE | fine-tuned | 0.592 | 0.868 | 0.908 |
| PaPs | frozen | 0.367 | 0.642 | 0.727 |
| PaPs | fine-tuned | 0.392 | 0.677 | 0.758 |
| Boundary head | frozen | 0.175 | 0.361 | 0.432 |
| Boundary head | fine-tuned | 0.098 | 0.207 | 0.258 |

Most border errors are therefore a shift of one pixel, not a wrong border. The ranking is the same at every
tolerance.

### Which real borders are drawn

![Recall by what lies across the real border](figures/border_types.png)

Each real border pixel is typed by what lies across it: another declared parcel of the same crop, of another crop,
undeclared land WorldCover calls grassland or cropland, or non-farm land.

| Method | Encoder | same crop (12 %) | other crop (14 %) | undeclared farmland (53 %) | non-farm land (22 %) |
|---|---|---|---|---|---|
| PRUE | frozen | 0.344 | 0.820 | 0.845 | 0.883 |
| PRUE | fine-tuned | 0.377 | 0.841 | 0.856 | 0.905 |
| PaPs | frozen | 0.278 | 0.670 | 0.525 | 0.528 |
| PaPs | fine-tuned | 0.293 | 0.729 | 0.566 | 0.571 |
| Boundary head | frozen | 0.152 | 0.316 | 0.588 | 0.846 |
| Boundary head | fine-tuned | 0.153 | 0.415 | 0.600 | 0.823 |

In brackets, the share of the 3,121,006 real border pixels.

* **The border between two parcels of the same crop is the hard one** for every method. PRUE draws one in three,
  PaPs a little fewer (0.28 to 0.29), the boundary head one in seven. These borders are only 12 % of the real border
  pixels, but each one missed joins two parcels into one field.
* **Between different crops**, PRUE draws more than four in five.
* **Outer edges**, against undeclared farmland or non-farm land, are three quarters of the real border. PRUE draws
  84 to 91 % of them; PaPs about half, because it leaves parcels out; the boundary head draws those against non-farm
  land well (82 to 85 %) and those against undeclared farmland less (59 to 60 %).

### Where the false borders lie

| Method | Encoder | real | false, inside a declared parcel | false, on non-farm land |
|---|---|---|---|---|
| PRUE | frozen | 0.947 | 0.036 | 0.017 |
| PRUE | fine-tuned | 0.938 | 0.041 | 0.021 |
| PaPs | frozen | 0.851 | 0.114 | 0.035 |
| PaPs | fine-tuned | 0.863 | 0.108 | 0.028 |
| Boundary head | frozen | 0.267 | 0.005 | 0.728 |
| Boundary head | fine-tuned | 0.126 | 0.003 | 0.871 |

Shares of the scored drawn border pixels. A false border inside a declared parcel is a false cut, or the edge of a
field that stops short of the parcel's edge. PRUE cuts rarely (4 %). PaPs' false borders are mostly inside parcels
(11 %), likely because its shapes come from a 16 x 16 grid and are rounded (PaPs report), so they miss the
parcels' corners. The boundary head's are on non-farm land.

### Chip by chip

![F1 of every test chip with at least 20 declared parcels](figures/per_chip.png)

| Method | Encoder | median | middle half |
|---|---|---|---|
| PRUE | frozen | 0.875 | 0.828 to 0.910 |
| PRUE | fine-tuned | 0.882 | 0.834 to 0.915 |
| PaPs | frozen | 0.659 | 0.578 to 0.731 |
| PaPs | fine-tuned | 0.696 | 0.616 to 0.757 |
| Boundary head | frozen | 0.428 | 0.326 to 0.510 |
| Boundary head | fine-tuned | 0.275 | 0.188 to 0.379 |

Over the 1,008 test chips with at least 20 declared parcels. The pooled result holds chip by chip: PRUE is above
PaPs on 99.4 % (frozen) and 99.2 % (fine-tuned) of these chips, and its spread is the narrowest.

## Examples

Four test chips, picked by a fixed rule among those with at least 20 declared parcels: PRUE's (frozen) F1 at its
90th, 50th and 10th percentile, and the chip where PaPs (frozen) beats PRUE (frozen) most. Shown for the frozen
encoder; the fine-tuned versions are `figures/example_<n>_ft.png` and look alike.

How to read each figure:

* **Row 1, the declaration**: the July Sentinel-2 composite; ESA WorldCover 2021 in its own colours (farmland is
  grassland, yellow, and cropland, pink) with the declared parcels outlined, and in the title the share of the
  chip's WorldCover farmland that is declared; the declared parcels coloured by crop; the declared borders in black
  on the scored land (white: non-farm land; light: declared parcels; grey: not scored, mostly undeclared farmland).
* **Rows 2 to 4, one method each**: two intermediate outputs, the final fields (random colours), and the border
  check.
  * PRUE: the class per pixel (sand background, light blue interior, dark blue boundary) and the boundary
    probability.
  * PaPs: the centre heatmap, scaled to its maximum with the dots enlarged (each peak is a parcel centre), and the
    saliency.
  * Boundary head: the border probability and the cores where it is below 0.5.
* **The border check**, in the last column:
  * blue: drawn, with a real border within 10 m;
  * orange: drawn, with no real border within 10 m;
  * black: a real border with nothing drawn within 10 m.

  The title gives the chip's precision, recall and F1.

### 1. PRUE at its 90th percentile: large fields

![Example 1](figures/example_1.png)

EE_02327_01812, 68 declared parcels. PRUE draws almost every border (precision 0.98, recall 0.89, F1 0.94),
including the long ones of the large barley, legume and grassland parcels. PaPs (0.92, 0.47, 0.63) leaves the
largest parcels out: the saliency covers them, but no centre is detected, so their borders stay black. The boundary
head (0.60, 0.51, 0.55) draws borders in and around the forest on the upper left: the orange there.

### 2. PRUE at its median: a dense mosaic

![Example 2](figures/example_2.png)

EE_02302_01815, 129 declared parcels. PRUE draws nearly nothing false (0.99, 0.79, 0.88). What it misses (black)
lies inside blocks of grassland parcels: the borders between two parcels of the same crop. PaPs (0.89, 0.54, 0.67)
places rounded parcels and leaves many out. The boundary head (0.47, 0.50, 0.48) draws orange borders over the
unfarmed patches in the middle and on the right of the chip. WorldCover calls most of the chip farmland, and 74 % of
it is declared; the undeclared rest lies in strips and patches between the parcels (grey in the border checks).
PRUE covers much of it with fields, which are not scored.

### 3. PRUE at its 10th percentile: grassland blocks

![Example 3](figures/example_3.png)

EE_02263_01778, 46 declared parcels. On the right, a block of adjacent grassland parcels: PRUE draws its outline but
few of the borders inside it (0.87, 0.66, 0.75), and so do the other two. PRUE also places a few small fields in
the woodland (orange specks). The boundary head (0.12, 0.49, 0.19) fills the woodland with borders. Only 57 % of
the chip's WorldCover farmland is declared: WorldCover finds grassland in many clearings of the woodland.

### 4. Where PaPs beats PRUE most: little farmland

![Example 4](figures/example_4.png)

EE_02270_01785, 20 declared parcels; WorldCover calls most of the chip water, with forest, wetland and a little
grassland, of which 24 % is declared. PRUE draws fields on the undeclared grassland strip along the right edge. Their
insides are not scored, but their edges run onto the wetland and water beside it, where they count as false (0.44,
0.55, 0.49). PaPs places fewer fields there (0.70, 0.55, 0.62). This is the partial labelling at work: if the strip
is a field, those borders are real. On a chip with so little farmland a few fields decide the score; such chips are
rare (PRUE is above PaPs on 99 % of the chips with at least 20 parcels).

## EuroCrops' partial labels and the three methods

**The concern.** EuroCrops holds the parcels farmers declared for subsidies, not every field. A chip is therefore
not a complete map of its fields: a method that took every pixel outside the declared parcels as "no field" would
learn that real fields are not fields.

### What the chips hold

![What the chips hold, and how much of each chip's WorldCover farmland is declared](figures/landcover.png)

| Land | train | validation | test |
|---|---|---|---|
| declared parcel, one of the 20 classes | 25.5 % | 25.8 % | 26.2 % |
| declared parcel, other crop | 1.0 % | 0.9 % | 1.1 % |
| undeclared WorldCover grassland | 11.9 % | 11.5 % | 11.9 % |
| undeclared WorldCover cropland | 1.3 % | 1.4 % | 1.2 % |
| non-farm land (WorldCover) | 60.4 % | 60.5 % | 59.6 % |
| **WorldCover farmland inside a declared parcel** | **65.9 %** | **66.7 %** | **66.5 %** |

Shares of the chips' pixels. A third of the farmland WorldCover maps in the chips is undeclared, nearly all of it
grassland: meadows outside the scheme, margins, verges and strips between parcels. Chip by chip, the median training
chip has 63 % of its farmland declared, and few have more than 90 %.

### How each method treats each kind of land in training

| Land | PRUE (`prue.target`) | PaPs (`targets.py`, `paps_loss.py`) | Boundary head (`delineation.py`) |
|---|---|---|---|
| declared parcel, 20 classes | interior, its outer ring boundary | a parcel: centre, box, shape, class | crop class, and border on its outer ring |
| declared parcel, other crop | ignored on WorldCover farmland; background on non-farm land (0.04 % of pixels) | as PRUE | ignored |
| undeclared WorldCover farmland | **ignored**: no loss | **ignored**: the heatmap loss is summed over known pixels only, and no centre is taken there | **ignored** |
| non-farm land | background | known: no centre may be predicted there | **ignored** |

So the understanding is right: **every method is trained only on what is labelled**, with WorldCover's non-farm
land added as the only negative for PRUE and PaPs. The training code was read for this, not the reports:

* **PRUE** gives undeclared farmland the ignore label, and the log-cosh Dice zeroes ignored pixels in both the
  prediction and the truth. The training windows are cut before the target is built, so a window edge is never a
  border.
* **PaPs as published** (PASTIS) treats every unannotated pixel as background, which would be incompatible with
  EuroCrops. Our copy carries a `known` channel: the centre loss counts only declared parcels and non-farm land, and
  undeclared farmland belongs to no parcel's zone. Its confidence threshold is tuned on the centres inside declared
  parcels only.
* **The boundary head** learns from the pixels of declared parcels alone. It is compatible with partial labels, but
  for another reason it never sees non-farm land, so nothing tells it where fields stop: that, not the labels, is
  why its fields fill the chip.

### Where the partial labels still matter

1. **Every declared edge is taught as a border**, whatever lies beyond it. 53 % of the real border pixels are the
   edge of a declared parcel against undeclared farmland. Where a farmer declared only part of a field, that edge is
   invisible, and the target asks for a border that is not there. It seems rare: PRUE draws 85 % of these edges,
   about as many as of the edges against forest or water (88 %) and more than between two crops (82 %), so most of
   them are visible borders.
2. **WorldCover decides what is not a field.** Undeclared farmland that WorldCover maps as forest or wetland is
   trained as background by PRUE and PaPs. Its errors go uncounted without an independent map; one sign of them is
   that 3.5 to 4.7 % of the declared parcel pixels lie on land WorldCover calls non-farm (those stay parcels in
   training, since the declaration comes first).
3. **The parcel score treats the chip as complete.** PQ, and the validation scores that chose the checkpoints of PRUE
   (object F1) and PaPs (PQ), count a field drawn on undeclared farmland as a false parcel. 57 to 61 % of PRUE's false
   fields lie on undeclared WorldCover grassland (13,581 frozen, 16,487 fine-tuned), and PaPs' show the same pattern
   (13,071 to 16,940). This penalises a method for finding fields nobody declared, PRUE the most. The border score
   of this report leaves undeclared farmland unscored, so it does not, apart from the edges of such fields that run
   onto non-farm land (example 4).

**No method is incompatible with EuroCrops as trained here.** The partial labelling shows in two places: the
evaluation of whole parcels, which counts fields nobody declared as false, and the declared edges against
undeclared land, which the targets take as borders.

## Analysis

**PRUE draws the borders; PaPs places parcels.** PRUE's boundary class traces lines in the image: tracks,
ditches, tree lines and changes of crop. Where it draws a line, a real border lies within 10 m in 95 % of the
cases. PaPs does not draw borders: it places a rounded shape at each centre it detects. Its borders are real where
it places a parcel, but every parcel without a detected centre loses all its borders, which is why its recall
stays at half. The boundary head marks where farmland meets forest or water well (recall 0.82 to 0.85), but it has no output for
land that is no field, so its fields fill the whole chip and their borders cut through forest and wetland.

**What remains is the border between parcels of the same crop.** Two neighbouring parcels of the same crop often
look alike in every month: the same sowing, the same harvest. Their border is visible only where a track, ditch or
strip of other land separates them. PRUE draws a third of these borders, the other methods fewer. Each one missed
merges two parcels, which is why PRUE's parcel-level error is the merge (the PRUE report: 39 % of the true parcels
are merged).

**Fine-tuning does not change the borders.** PRUE gains +0.006 F1. PaPs gains +0.035: its encoder was fine-tuned on
crop type, which separates parcels of different crops (recall +0.06 there). The boundary head loses: fine-tuned, it
draws more than twice as many borders on non-farm land.

**The border F1 and the parcel score answer neighbouring questions.** The border F1 counts border pixels; the
parcel score (PQ, in the PRUE report) counts whole parcels. A single gap of a few pixels in a long line hardly moves
the border F1 but merges two parcels. Both give the same ranking: PRUE, then PaPs, then the boundary head.

## Caveats

* **One fit per method**: one draw, one seed, K = 100 %. No interval.
* **The declared parcels are the truth.** Some declared borders are not visible at 10 m in any month (two parcels
  of the same crop farmed alike, an administrative split), and some visible borders are not declared. The low
  recall on same-crop borders is partly this.
* **Undeclared farmland is not scored.** A field drawn there, and its borders, count neither as right nor as false.
  The PRUE report found 57 to 61 % of PRUE's false fields on undeclared WorldCover grassland, so PRUE's precision
  here does not include them; neither does PaPs' or the boundary head's. Where such a field's edge runs onto
  non-farm land beside it, that edge does count, as false (example 4): at most the 1.7 to 2.1 % of PRUE's drawn
  border that lies false on non-farm land.
* **Half of the real border (53 %) is the edge of a declared parcel against undeclared farmland**: field margins,
  verges, undeclared meadows. They count as real borders, since the declared parcel ends there.
* **The encoders' fine-tuning differs**: PRUE's encoder was trained end to end with it; PaPs' was fine-tuned on crop
  type and then frozen; the boundary head's was trained with both heads.

## Next steps

1. **The same-crop borders.** PRUE's boundary threshold below the argmax, tuned on the validation chips (the PRUE
   report's first next step), targets exactly these; `borders.py` measures the effect on them directly.
2. **How many of the missed same-crop borders are visible.** A look at a sample of them on the monthly composites
   would tell how much of the gap is the imagery and how much the model.

## Reproduce

    poetry run python experiments/2026-11-07_border_accuracy/borders.py --chips EE_2021_rev --workers 12
    poetry run python experiments/2026-11-07_border_accuracy/landcover.py --chips EE_2021_rev --workers 12
    poetry run python experiments/2026-11-07_border_accuracy/report.py --chips EE_2021_rev

`borders.py` writes `results/2026-11-07_border_accuracy/EE_2021_rev/borders.json` and `per_chip.npz`;
`landcover.py` `landcover.npz` beside them; `report.py` the figures and `figures/tables.md`.
