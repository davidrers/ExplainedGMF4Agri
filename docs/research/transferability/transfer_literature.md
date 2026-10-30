# How transfer experiments in crop mapping are designed and reported

Literature gathered on 27 October 2026 for the planning of the spatial transfer section. Nothing here is a decision.
Numbers marked (s) were read from a model-extracted summary of the paper rather than its rendered tables, and must
be checked against the PDF before they are cited in the thesis.

## Four designs in use

| Design | What it asks | Fits it costs | Used by |
|---|---|---|---|
| **Selected source to target pairs**, ordered by distance | How much is lost from source to target, and does the loss grow with distance? | One per source | TimeMatch (5 pairs over 4 regions), CropGlobe (9 pairs in three tiers: cross-country, cross-continent, cross-hemisphere), Sen4AgriNet (3 scenarios) |
| **Held-out domain** (leave one country or region out) | Does a model trained on everything else work in a new place? | One per held-out domain | WorldCereal (10 held-out countries for crop type), Chang et al. (ID against pooled OOD regions), Fields of The World (held-out countries) |
| **Held-out year** | Does a model trained on past seasons work in a new season in the same place? | One | WorldCereal (2021 held out), Sen4AgriNet (spatio-temporal), Weikmann et al. 2021 (Austria 2018 to 2019 and 2020) |
| **Source plus K target labels** against target only | How many target labels does the source save? | One per pair and K | EuroCropsML (LV to EE, LV plus PT to EE), Chang et al. (OOD plus ID), protocol section 5.5 |

No paper found runs a full country by country matrix with a quantitative test of what explains the loss. The pair
designs choose pairs to form a ladder of distance and read the trend by eye; CropGlobe's own statement that
performance "decreases as the domain shift increases" holds on average, not per pair.

## Headline findings that bear on the design

| Study | Setting | Finding |
|---|---|---|
| Nyborg et al. 2022, TimeMatch | 4 Sentinel-2 tiles, 2017: Denmark, Austria, two in France; about 15 common classes plus "unknown" | Source-only macro-F1 averages 39.2 against 82.0 when trained on the target (s). The closest pair (France to France) loses 15 points, Denmark to France about 45 (s). The phenology of the same crop is shifted by about a month between Denmark and France (s), and estimating and removing that shift recovers most of the loss |
| Sykas et al. 2022, Sen4AgriNet | Catalonia and France, 2019 and 2020; 11 common classes covering 99.9 % of pixels, on a taxonomy built from FAO ICC | Pixel weighted F1 about 85 in-domain, 58 across countries, 44 across country and year (s) |
| Butsko et al. 2025, WorldCereal | Global; 7 crops plus other; random, held-out-country and held-out-year splits | Crop-type F1, fine-tuned Presto 0.81 / 0.69 / 0.65 (random / year / country); Presto from random weights 0.78 / 0.65 / 0.62; CatBoost 0.73 / 0.65 / 0.56 (s). A year costs less than a country; pretraining helps most out of distribution |
| Tong and Wang 2025, CropGlobe | 8 countries, 2023, 6 crops plus other; southern hemisphere windows shifted six months | Near transfers (France to Belgium, Netherlands, UK) reach 92 to 98 % OA with every method. Across hemispheres AlphaEarth plus MLP falls to 28 to 46 % OA and Presto plus RF to 33 to 45 %, while a small CNN trained with time-shift augmentation (plus or minus 30 days) keeps 75 to 88 %. Classes absent from the target are mapped to "other" or excluded |
| Chang et al. 2025, IGARSS | 5 regions on 5 continents, harmonised to maize, soybean, rice, wheat, other; frozen ResNet-50 and a U-Net decoder | Adding 4,000 out-of-region samples to 900 in-region ones lowers F1 (EuroCrops 0.48 to 0.37); the authors recommend training on other regions first and then fine-tuning on the target alone |
| Reuss et al. 2026, EuroCropsML | LV to EE, LV plus PT to EE | Adding Portugal to the source hurts Estonia (transfer learning 0.292 against 0.365 OA at 20 shots) |
| Shang et al. 2026 | Prithvi, SpectralGPT, SatMAE; held-out regions within four US states | All three predict only the common crops in unseen regions; winter wheat and sorghum IoU 0 |
| Kolluru et al. 2026 | Fine-tuned Prithvi-EO-2.0, 37 events in 12 countries on three continents | OA falls from 0.65 in the US to 0.40 in Europe; the abstract attributes most of the fall to the observation window missing the local crop calendar. Expected calibration error rose in seven of eight paired events, with 12 to 51 % of affected scenes confidently wrong |
| Kluger, Wang and Lobell 2021 | France (11 training departments) and Kenya | Separates the two shifts: crop mix (prior shift, corrected by reweighting posteriors) and features (a linear shift of the feature mean); correcting both removes 22 % of misclassifications on average in France |
| Mohammadi, Belgiu and Stein 2024 (ITC) | Corn, soybean and other, US Landsat series; cross-region and cross-year | Source-free adaptation raises macro-F1 more across regions (8.8 to 20.1 %) than across years (5.8 to 12.9 %) |
| Orynbaikyzy, Gessner and Conrad 2022 | 7 sites in Germany, 11 crops, 2018, Random Forest | Sentinel-1 features, or features chosen by spatial feature selection, transfer between sites better than optical ones |

## How the label space is handled

- **A fixed, coarse, harmonised list plus "other"** is the norm whenever more than two domains meet: 5 classes in
  Chang et al., 7 in CropGlobe, 8 in WorldCereal, 11 in Sen4AgriNet, about 15 plus "unknown" in TimeMatch. The list is
  fixed before results and chosen by coverage across the domains.
- **Classes the target has but the source lacks** are either excluded from the score (CropGlobe for France to USA)
  or counted as "other". Classes the source has but the target lacks have their predictions mapped to "other".
- **The ML vocabulary:** partial domain adaptation (target classes a subset of the source's), open-set (the target has
  classes the source lacks), universal (both). Few crop papers treat it explicitly; MultiCropGAN (ISPRS Annals 2024)
  is one.
- **Prior shift** is treated apart from feature shift: Kluger et al. 2021; DirPA on EuroCropsML (Reuss, Gikalo and
  Körner 2026); protocol section 5.6.

## How the year and the calendar are handled

- **Spatial designs fix one year** for every domain (TimeMatch 2017, CropGlobe 2023) and treat the year as a separate
  axis when they study it.
- **The crop calendar is the dominant failure across climates.** TimeMatch, CropGlobe's time-shift augmentation and
  Kolluru et al. all find that fixed calendar windows misalign phenology between regions, and that aligning the
  window recovers much of the loss. A model fed fixed calendar months, as ours are, will show this across zones.
- **A held-out year costs less than a held-out country** in WorldCereal (0.69 against 0.65) and in Sen4AgriNet.

## What these studies report

1. A table of source-only, target-trained and the gap per pair (TimeMatch), or scores by split side by side
   (WorldCereal: random, year, country).
2. Per-class F1 and confusion matrices under transfer: rare crops collapse first (Shang et al.); confusions between
   crops of one group (wheat against other cereals in Chang et al.).
3. Learning curves of target labels with and without the source (EuroCropsML, Chang et al.).
4. Calibration under shift (Kolluru et al.), which links to Phase 2.
5. Feature-space pictures (t-SNE in CropGlobe), qualitative only.
6. Rarely: a quantitative relation between the loss and a distance. Meyer and Pebesma's area of applicability
   (a dissimilarity index in predictor space, thresholded from cross-validation) is the established way to say where
   a model can be trusted; Miller et al.'s "accuracy on the line" tests whether in-domain rank predicts out-of-domain
   rank across models.

## Sources

- Nyborg, Pelletier, Lefèvre, Assent (2022). TimeMatch. [arXiv:2111.02682](https://arxiv.org/abs/2111.02682)
- Sykas, Sdraka, Zografakis, Papoutsis (2022). Sen4AgriNet. [arXiv:2204.00951](https://arxiv.org/abs/2204.00951)
- Selea (2023). AgriSen-COG, five countries, 2019 and 2020. [Remote Sensing 15, 2980](https://www.mdpi.com/2072-4292/15/12/2980)
- Butsko et al. (2025). Deploying geospatial foundation models in the real world: lessons from WorldCereal. [arXiv:2508.00858](https://arxiv.org/abs/2508.00858)
- Tong, Wang (2025). Invariant features for global crop type classification (CropGlobe). [arXiv:2509.03497](https://arxiv.org/abs/2509.03497)
- Chang et al. (2025). On the generalizability of foundation models for crop type mapping. [arXiv:2409.09451](https://arxiv.org/abs/2409.09451)
- Kerner et al. (2025). Fields of The World. [arXiv:2409.16252](https://arxiv.org/abs/2409.16252)
- Reuss et al. (2026). Benchmarking for practice, EuroCropsML. [arXiv:2504.11022](https://arxiv.org/abs/2504.11022)
- Shang, Das, Eldawy (2026). Benchmarking geospatial foundation models for agriculture applications. [arXiv:2606.29664](https://arxiv.org/abs/2606.29664)
- Kolluru et al. (2026). Transferability and operational reliability of a Prithvi crop classification foundation model under phenological and geographic shift. [arXiv:2610.08810](https://arxiv.org/abs/2610.08810) (abstract only read)
- Scarlat, Plajer, Băicoianu (2026). EuroCrops-driven Sentinel-2 crop segmentation, external tests on Belgian subsets. [arXiv:2606.00676](https://arxiv.org/abs/2606.00676)
- Kluger, Wang, Lobell (2021). Two shifts for crop mapping. [arXiv:2109.01246](https://arxiv.org/abs/2109.01246)
- Mohammadi, Belgiu, Stein (2024). Source-free unsupervised domain adaptation for cross-regional and cross-time crop mapping. Remote Sensing of Environment, [doi:10.1016/j.rse.2024.114385](https://doi.org/10.1016/j.rse.2024.114385)
- Orynbaikyzy, Gessner, Conrad (2022). Spatial transferability of Random Forest models for crop type classification. [Remote Sensing 14, 1493](https://www.mdpi.com/2072-4292/14/6/1493)
- Weikmann, Paris, Bruzzone (2021). Multi-year crop type mapping using pre-trained deep long short-term memory. [SPIE 11862](https://iris.unitn.it/handle/11572/330189)
- Meyer, Pebesma (2021). Predicting into unknown space? Area of applicability. [arXiv:2005.07939](https://arxiv.org/abs/2005.07939)
- Miller et al. (2021). Accuracy on the line. [arXiv:2107.04649](https://arxiv.org/abs/2107.04649)
- Survey of label-space settings in remote-sensing domain adaptation (2025). [arXiv:2510.15615](https://arxiv.org/abs/2510.15615)
