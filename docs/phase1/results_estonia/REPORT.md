# Crop-type maps of Estonia 2021 from geospatial foundation models: first results

19 October 2026; section 4 revised on 30 October 2026. Every number below comes from one random draw of the labels
and one training seed per setting, so none has a confidence interval yet (section 5). The figures and
[tables.md](tables.md) are made by [report.py](report.py).

## In short

- **Best map:** TESSERA v2 with its encoder fine-tuned. Test Macro-F1 per pixel is 0.660, 0.630 and 0.581 with
  100 %, 20 % and 5 % of the training parcels labelled, and 0.639, 0.603 and 0.566 per parcel. It leads at every
  budget except per pixel with all labels, where it ties frozen TerraMind (0.661).
- **Best frozen encoder:** TerraMind per pixel with all labels. With fewer labels, and per parcel at every budget,
  frozen TESSERA v2 leads (0.582 and 0.504 per pixel at 20 % and 5 %).
- **The baseline is hard to beat.** A U-Net trained from scratch, with no pretraining, scores 0.616, 0.564 and 0.467.
  TESSERA is above it at every budget, though frozen only narrowly with all labels (0.618). TerraMind beats it with
  all labels and ties it at 20 %. At 5 %, every token-grid model falls below it.
- **Fine-tuning pays off for TESSERA only:** +0.04 to +0.08 Macro-F1. For TerraMind, THOR and Prithvi the gain is
  about +0.01 in validation, within the noise of a single seed.
- **Per parcel, every model scores lower than per pixel,** by 0.002 to 0.08, because small parcels are mapped badly
  and the parcel score counts each parcel once.
- **Against published Estonian results:** the only published Estonian Macro-F1 is per parcel, from DirPA (Reuss et
  al. 2026), with at most 100 labelled parcels per class: 0.174 over 127 crop types and 0.307 over coarse groups for
  a Transformer trained from scratch, 0.064 and 0.255 for the Presto foundation model. At K = 5 %, a median of 74
  labelled parcels per class, our parcel Macro-F1 is 0.35 to 0.57. The gap comes mostly from the task, 20 crops
  against 127 (section 4), so it shows our numbers are plausible, not that our models are better. The ranking
  carries over: there Presto falls below the model trained from scratch, here every frozen model but TESSERA falls
  below the U-Net at 5 %. No published study maps Estonian crops per pixel, and none has tested these five
  foundation models on Estonia.

## 1. What was done

```mermaid
flowchart LR
    A["Parcel declarations<br/>EuroCrops, Estonia 2021"] --> C["7,402 tiles of 2.24 km<br/>crop label and parcel id per 10 m pixel"]
    B["Sentinel-2 and Sentinel-1<br/>12 monthly composites"] --> C
    C --> D["Spatial split<br/>train / validation / test blocks"]
    D --> E["Keep the labels of K % of<br/>the training parcels of each crop"]
    E --> F["Encoder (frozen or fine-tuned)<br/>+ trained decoder"]
    F --> G["Crop map of the test tiles,<br/>scored per pixel and per parcel"]
```

1. **Labels.** Farmers' declarations of 2021 from EuroCrops v11: a polygon and a crop per parcel. We keep the 20 most
   common crop classes of Estonia (each with at least 250 parcels). They hold 95 % of the declared parcels;
   grassland alone is about half. Pixels of other classes are left unlabelled.
2. **Tiles.** Estonia is cut into a fixed grid of 7,402 tiles of 2.24 x 2.24 km, 224 x 224 pixels of 10 m. Each tile
   has two label rasters: the crop of each pixel, and the parcel each pixel belongs to.
3. **Imagery.** For every tile, twelve monthly composites of 2021, each the median of the cloud-free views of that
   month: Sentinel-2 L2A (12 optical bands) and Sentinel-1 RTC radar (VV and VH, which see through cloud).
4. **Split.** Tiles are grouped into blocks of 4 x 4 tiles (9 km) and whole blocks go to training, validation or test,
   with a 1.6 km buffer, so the model is never tested next to a field it was trained on: 4,898 training, 769
   validation and 1,475 test tiles, with 34,610 test parcels.
5. **Label budget K.** To simulate scarce labels, we keep the labels of only K % of the training parcels of each crop,
   K = 100, 20 and 5 % (86,909, 17,390 and 4,354 parcels). Counting per crop keeps the rare crops in the training
   set. Validation and test stay fully labelled.
6. **Models.** Each model is an encoder that turns the imagery into features and a decoder that turns the features
   into a crop per pixel (section 2). **Frozen:** the pretrained encoder is kept as it is and only the decoder is
   trained. **Fine-tuned:** the encoder is trained too, with a small learning rate so it adapts without forgetting. The
   **baseline** is a U-Net that learns everything from our labels, with no pretraining.
7. **Training.** 45 epochs for every model; cross-entropy on the labelled pixels; random flips and rotations; the
   checkpoint with the lowest validation loss is the one tested.
8. **Scoring, on the test tiles.** *Per pixel:* Macro-F1, the mean of the 20 per-crop F1 scores, so a rare crop counts
   as much as grassland (the primary metric). *Per parcel:* each parcel takes the crop most of its pixels were
   predicted as and is right or wrong as a whole, as published EuroCrops studies score; we report Macro-F1, overall
   accuracy and Cohen's kappa.

## 2. The models

| Model | Pretrained on | What we give it | What the decoder receives | Decoder (trained) | Encoder size |
|---|---|---|---|---|---|
| **TerraMind v1 large** (IBM and ESA, Jakubik et al. 2025) | Single-date multimodal Earth observation (Sentinel-2, Sentinel-1 and more) | S2 + S1, each month run separately; the two sensors' features averaged | One 1,024-number feature per 160 m square (16 x 16 px), 14 x 14 per tile, for each of 12 months | Squeeze the 12 months to 768 channels, then a U-Net decoder that upsamples to 10 m: 53.3 M parameters | 306 M |
| **THOR v1 large** (Norwegian Computing Center and ESA, Forgaard et al. 2026) | Single-date Sentinel-1, 2 and 3 | S2 (10 m and 20 m bands) + S1, each month separately; the three band groups averaged | As TerraMind: 160 m features, 12 months | Same decoder as TerraMind, 53.3 M | 314 M |
| **Prithvi-EO-2.0 600M TL** (NASA and IBM, Szwarcman et al. 2024) | Time series of Harmonised Landsat and Sentinel-2, with date and location | S2 only (its six pretraining bands, all it accepts), the 12 months as one time series | One 1,280-number feature per 140 m square (14 x 14 px), for each month | Same design, 62.7 M | 631 M |
| **TESSERA v2 Medium** (University of Cambridge, Feng et al. 2025) | Per-pixel yearly time series of Sentinel-2 and Sentinel-1 | S2 (10 bands) + S1, the 12 months of each pixel | 128 numbers per 10 m pixel, summarising its year | A small per-pixel network (two layers, 512 and 256): 0.2 M | 21 M |
| **AlphaEarth Foundations v1** (Google DeepMind, Brown et al. 2025) | Many sources (optical, radar, lidar, climate) | Nothing: Google publishes the result for every pixel on Earth, and we download it | 64 numbers per 10 m pixel for 2021 | The same per-pixel network: 0.2 M | not released |
| **U-Net from scratch** (baseline) | Nothing | S2 + S1, the 12 months stacked as 168 input channels | (no separate encoder) | A ResNet-18 U-Net, everything trained: 14.8 M | none |

**Two families.** TerraMind, THOR and Prithvi see the tile in squares of 140 to 160 m and look at the whole tile at
once; their decoder has to rebuild the 10 m detail. TESSERA and AlphaEarth describe each 10 m pixel by its own year
and know nothing of its neighbours; their decoder judges each pixel alone. That 256-fold difference in starting
detail is the main reason the two families behave differently, so the decoder is matched within each family, not
across them.

**Fine-tuned versions.** TerraMind, THOR, Prithvi and TESSERA were also run with their encoders trained (learning
rate 1e-5 after a one-epoch warm-up; decoder and everything else unchanged). AlphaEarth cannot be fine-tuned, as only
its outputs are published. Fine-tuned TESSERA is run on the labelled pixels only, since unlabelled pixels carry no
loss for a per-pixel model.

## 3. Results

### 3.1 Accuracy against the number of labels

![Test Macro-F1 against the label budget, per pixel and per parcel](figures/label_curves.png)

| Test Macro-F1, per pixel / per parcel | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| TerraMind v1 large, frozen | **0.661** / 0.583 | 0.566 / 0.515 | 0.433 / 0.381 |
| TerraMind v1 large, fine-tuned | 0.598 / 0.545 | 0.562 / 0.511 | 0.351 / 0.316 |
| THOR v1 large, frozen | 0.594 / 0.516 | 0.547 / 0.491 | 0.393 / 0.349 |
| THOR v1 large, fine-tuned | 0.627 / 0.545 | 0.535 / 0.475 | 0.404 / 0.369 |
| Prithvi-EO-2.0 600M, frozen | 0.580 / 0.535 | 0.505 / 0.472 | 0.445 / 0.409 |
| Prithvi-EO-2.0 600M, fine-tuned | 0.613 / 0.545 | 0.540 / 0.492 | 0.394 / 0.371 |
| TESSERA v2 Medium, frozen | 0.618 / 0.602 | 0.582 / 0.574 | 0.504 / 0.502 |
| TESSERA v2 Medium, fine-tuned | 0.660 / **0.639** | **0.630** / **0.603** | **0.581** / **0.566** |
| AlphaEarth v1, frozen | 0.523 / 0.484 | 0.497 / 0.455 | 0.449 / 0.425 |
| U-Net from scratch (baseline) | 0.616 / 0.553 | 0.564 / 0.520 | 0.467 / 0.446 |

- **Fewer labels cost the token-grid models most.** From 100 % to 5 % of the labels, frozen TerraMind loses 0.23
  Macro-F1 and THOR 0.20, while TESSERA loses 0.11 and AlphaEarth 0.07. TESSERA also leads the frozen models at 20 %
  and 5 %.
- **The baseline is a strong reference.** Without any pretraining, the U-Net beats THOR, Prithvi and AlphaEarth with
  all labels, and every token-grid model, frozen or fine-tuned, at 5 %. Fine-tuned TESSERA stays clearly above it at
  every budget; frozen TESSERA ties it with all labels and is above it at 20 and 5 %; TerraMind is above it only
  with all labels.
- **AlphaEarth** is the weakest with many labels but the flattest curve: at 5 % it is level with Prithvi.

### 3.2 Frozen against fine-tuned

![Fine-tuned minus frozen, per model and label budget](figures/finetune_gain.png)

| Fine-tuned minus frozen, Macro-F1 | Test per pixel (100 / 20 / 5 %) | Test per parcel | Best validation |
|---|---|---|---|
| TerraMind | -0.063 / -0.004 / -0.082 | -0.038 / -0.004 / -0.065 | +0.012 / +0.008 / +0.010 |
| THOR | +0.033 / -0.012 / +0.011 | +0.029 / -0.016 / +0.020 | +0.010 / 0.000 / -0.033 |
| Prithvi | +0.033 / +0.034 / -0.051 | +0.009 / +0.020 / -0.038 | +0.016 / +0.004 / +0.008 |
| TESSERA | +0.042 / +0.047 / +0.077 | +0.037 / +0.028 / +0.063 | +0.053 / +0.051 / +0.056 |

- **TESSERA gains clearly**, under all three readouts and most with the fewest labels. Run frozen through the same
  fine-tuning code, its validation score moves by less than 0.005, so the gain comes from training the encoder.
- **TerraMind, THOR and Prithvi gain about +0.01** on the best validation score, the readout that does not depend on
  which checkpoint is kept. On test, the sign flips from one budget to the next.
- **The test swings come from the checkpoint rule.** We test the checkpoint of lowest validation loss. With the
  encoder training, that loss turns up after only 5 to 9 epochs, before the rarer crops are learnt, while
  validation Macro-F1 keeps rising to epochs 13 to 44. Tested at its best-Macro-F1 epoch instead, fine-tuned
  TerraMind would score 0.688, 0.627 and 0.523; the frozen models were not tested that way, so that comparison is
  not yet possible.
- **Cost.** Fine-tuning takes 4 to 9 times the frozen training time for the token-grid models, and 15 times for
  TESSERA (26 h for the full-label fit on one RTX PRO 6000).

Details: [`experiments/2026-10-15_kshot_finetune/REPORT.md`](../../../experiments/2026-10-15_kshot_finetune/REPORT.md).

### 3.3 Per pixel against per parcel

- **The parcel score is lower for every model** (section 3.1 table). Each parcel counts once, and 18 % of the test
  parcels are under 0.64 ha: parcels under 0.16 ha get the right crop only 24 to 38 % of the time, those above
  0.64 ha 67 to 87 %. (The size and vote figures in this section were measured on the 15-epoch fits; for the frozen
  token-grid models those are the same fits.)
- **The token-grid models lose most** (0.02 to 0.08), because their 140 to 160 m squares are larger than these
  fields. TESSERA loses 0.002 to 0.03, which is why it leads per parcel at every budget.
- **Potatoes show it most:** they are mostly household plots, with a median test parcel of 0.12 ha. Frozen TerraMind
  scores 0.75 F1 on potatoes per pixel (mostly the large commercial fields) but 0.18 per parcel.
- **The vote fixes little:** taking the majority per parcel corrects only 15 to 32 % of the wrong pixels. The rest
  sit in parcels mapped as the wrong crop as a whole.

Details: [`experiments/2026-10-16_parcel_metric/REPORT.md`](../../../experiments/2026-10-16_parcel_metric/REPORT.md).

### 3.4 Per crop

![Per-class test F1 per pixel at K = 100 and 5 %](figures/per_class.png)

Rows are sorted by the number of training parcels, given in brackets.

- **Easy crops:** grassland, winter wheat, winter rapeseed, winter barley and maize reach F1 0.87 to 0.97 with all
  labels for almost every model.
- **Confusable crops:** the forage crops that look like grassland (legumes harvested green, clover, F1 0.41 to
  0.55) and the spring cereals that look like each other (oats, spring wheat, 0.42 to 0.71).
- **Four crops are almost never found:** fresh vegetables, orchards, dried pulses and alfalfa score F1 0 to 0.26 for
  every model, even with all labels. They are mixed or small plots with few parcels. Being 4 of the 20 classes, they
  alone cap Macro-F1 near 0.80.
- **With 5 % of the labels the 10 m models keep the small crops** that the token-grid models and the U-Net lose:
  potatoes drop to F1 0 for TerraMind, THOR, Prithvi and the U-Net but stay at 0.52 for frozen TESSERA and 0.70
  fine-tuned. Buckwheat behaves alike: 0.70 and 0.81 for TESSERA against 0 to 0.53 for the others. Prithvi misses
  potatoes even with all labels.

## 4. Comparison with published results for Estonia

No published study maps Estonian crops per pixel, and none has run TerraMind, THOR, Prithvi, TESSERA or AlphaEarth
on Estonia. The published Estonian results classify whole parcels, so they are set against our per-parcel score.
**Only one study reports parcel Macro-F1 for Estonia:** DirPA (Reuss, Gikalo and Körner 2026, Table 6), on
EuroCropsML 2.0, with 1 to 100 labelled parcels per class and no full-label result. The table puts its models
trained with plain cross-entropy, as ours are, beside ours.

**Matching the label budgets.** DirPA labels at most k parcels of each class. We label K % of the training parcels of
each class, so the number per class varies with the crop: 17 to 2,167 at K = 5 % (median 74), 67 to 8,666 at 20 %
(median 296), 332 to 43,330 with all labels (median 1,477). Our K = 5 % therefore stands beside their 100 per class.

| Parcel Macro-F1 | Classes | k = 20 per class | k = 100 per class, or K = 5 % (median 74) | K = 20 % (median 296) | K = 100 % (median 1,477) |
|---|---|---|---|---|---|
| **Published:** *DirPA, mean of five runs* | | | | | |
| Transformer from scratch | 127 crop types | 0.117 | 0.174 | | |
| Presto, released weights, fine-tuned | 127 crop types | 0.038 | 0.064 | | |
| Transformer from scratch | coarse groups (HCAT level 3) | 0.272 | 0.307 | | |
| Presto, released weights, fine-tuned | coarse groups (HCAT level 3) | 0.213 | 0.255 | | |
| **Ours:** *one run* | | | | | |
| U-Net from scratch | 20 crops | | 0.446 | 0.520 | 0.553 |
| TerraMind v1 large, frozen | 20 crops | | 0.381 | 0.515 | 0.583 |
| THOR v1 large, frozen | 20 crops | | 0.349 | 0.491 | 0.516 |
| Prithvi-EO-2.0 600M, frozen | 20 crops | | 0.409 | 0.472 | 0.535 |
| TESSERA v2 Medium, frozen | 20 crops | | 0.502 | 0.574 | 0.602 |
| TESSERA v2 Medium, fine-tuned | 20 crops | | 0.566 | 0.603 | 0.639 |
| AlphaEarth v1, frozen | 20 crops | | 0.425 | 0.455 | 0.484 |

DirPA's own prior correction moves its Transformer by -0.05 to +0.06; no Estonian result in the paper exceeds 0.174
over the 127 types or 0.313 over the coarse groups. Its results at 1, 5 and 10 parcels per class are in the figure
and in [tables.md](tables.md); the fine-tuned TerraMind, THOR and Prithvi are in section 3.1.

![Parcel Macro-F1 against labelled parcels per class, ours beside DirPA](figures/literature.png)

The x-axis puts both on labelled parcels per class: at most k for DirPA, the median over our 20 crops for ours.

**How the numbers line up.**

- **At a similar number of labels per class, our parcel Macro-F1 is two to three times theirs over the 127 types**
  and above their coarse groups: 0.35 to 0.57 at K = 5 %, against 0.174 and 0.307 for their Transformer.
- **That gap comes mostly from the task, not the models** (table below). Over 127 crop types, Macro-F1 averages in
  the many crops with a handful of parcels, which every model nearly misses; we score only the 20 crops with at
  least 250 parcels. The coarse groups are fewer, but the paper gives neither their number nor their sizes for
  Estonia, so they cannot be put on our 20-crop scale either.
- **The ranking does carry over.** In DirPA, Presto from its released weights scores below the Transformer trained
  from scratch at every budget, by 0.04 to 0.11 over the 127 types and 0.04 to 0.06 over the coarse groups. Here, at
  K = 5 %, four of the five frozen foundation models fall below our U-Net trained from scratch (0.349 to 0.425
  against 0.446); only TESSERA is above it, frozen (0.502) and fine-tuned (0.566). With all labels TerraMind passes
  it too.

**Why the numbers are not on the same scale.** The comparison says our results are plausible, not that our models
are better:

| | Published (DirPA) | Ours | Effect on our numbers |
|---|---|---|---|
| Classes | 127 crop types, most with few parcels, or coarser groups | The 20 most common crops, 95 % of the parcels; the rest are not scored | Higher |
| Labelled set | At most k parcels of every class, so grassland gets no more labels than any other crop | K % of every class, so training keeps the test's crop shares | Not directly comparable; matched above by the median per class |
| Split | Random within each class: test parcels sit next to training parcels | Spatial blocks with a 1.6 km buffer | Lower: a harder, more honest test |
| Input | One median Sentinel-2 L1C spectrum per parcel per date | Every 10 m pixel, monthly S2 L2A and S1 composites | Pixels add field shape and radar; medians give clean per-parcel spectra |
| Model output | One class per parcel | One class per pixel, then a vote | Lower on small parcels |
| Runs | Mean of five | One draw, one seed | Ours carry no interval yet |

**Published results in other metrics.** The EuroCropsML benchmark (Reuss et al. 2026) and its follow-up *Mind the
gap* (Reuss, Gikalo and Körner 2025) report overall accuracy and Cohen's kappa for Estonia, not Macro-F1; *Mind the
gap* plots Macro-F1 but prints no values. Järveoja (2021) reports weighted F1 over 28 classes, 0.82 and 0.85 for 2018
and 2019. Overall accuracy is the share of test parcels given the right crop, and weighted F1 weights each crop by
its number of parcels; with grassland about half of the parcels, both mostly measure grassland, which is why our
metric is Macro-F1. For the record, with all labels the benchmark's Transformer reaches an overall accuracy of 0.787
and our models 0.75 to 0.82 ([tables.md](tables.md)).

## 5. Caveats and next steps

- **One draw, one seed.** A different random choice of labelled parcels or a different seed can move a cell by a
  few hundredths, about the size of several gaps above. Repeated draws with intervals are next in the protocol.
- **The checkpoint rule decides the token-grid results.** Lowest validation loss stops TerraMind, THOR and Prithvi
  long before their best validation Macro-F1, and training 45 instead of 15 epochs changed none of their frozen
  results. Testing the checkpoint of highest validation Macro-F1 for every model would settle whether fine-tuning
  helps them.
- **Only three budgets.** The protocol plans 1, 5, 10, 20, 50 and 100 %.
- **Latvia and Portugal** follow; the comparison to EuroCropsML extends to them.
- **A per-parcel anchor on our own split.** Training a classifier on per-parcel medians of our tiles, as EuroCropsML
  does, would put the published numbers and ours on the same split.

## Sources

- Our results: `results/kshot/EE_2021_rev/` (frozen and baseline) and `results/2026-10-15_kshot_finetune/EE_2021_rev/`
  (fine-tuned), 45 epochs, run on the UT HPC cluster, 16 to 18 October 2026. Method: [protocol.md](../protocol.md),
  [pipeline.md](../pipeline.md), and in plain language [pipeline_overview.md](../pipeline_overview.md).
- J. Reuss, J. Macdonald, S. Becker, E. Gikalo, K. Schultka, L. Richter, M. Körner (2026). Benchmarking for practice:
  few-shot time-series crop-type classification on the EuroCropsML dataset. *ISPRS Open Journal of Photogrammetry
  and Remote Sensing* 19, 100117. arXiv 2504.11022. Numbers from section 6.1, Table 4a and Table B.18.
- J. Reuss, E. Gikalo, M. Körner (2026). DirPA: addressing prior shift in imbalanced few-shot crop-type
  classification. arXiv 2603.12905 (v2, July 2026). Estonian parcel Macro-F1 from Table 6.
- J. Reuss, E. Gikalo, M. Körner (2025). Mind the gap: bridging prior shift in realistic few-shot crop-type
  classification. arXiv 2511.16218. Estonian accuracy and kappa in Table 1; Macro-F1 only plotted (Figure 4).
- The models: TerraMind, Jakubik et al. 2025, arXiv 2504.11171; THOR, Forgaard et al. 2026, arXiv 2601.16011;
  Prithvi-EO-2.0, Szwarcman et al. 2024, arXiv 2412.02732; TESSERA, Feng et al. 2025, arXiv 2506.20380; AlphaEarth
  Foundations, Brown et al. 2025, arXiv 2507.22291.
- M. Järveoja (2021). Põllukultuuride tuvastamise masinõppe mudeli tunnuste olulisuse hindamine. MSc thesis,
  University of Tartu. [dspace.ut.ee](https://dspace.ut.ee/items/5f143077-3e65-4f9c-b493-e68eca91d30e)
