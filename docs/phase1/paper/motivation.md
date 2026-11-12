# Phase 1 paper: motivation

Draft of 11 November 2026. One page: the problem, why it matters, the gap, the contribution, the approach and the
expected results.

**Problem.** Field-scale crop-type maps feed agricultural statistics, food-security monitoring and the area monitoring
of the EU Common Agricultural Policy. Producing them requires labelled fields, and such labels are costly to collect,
become available late in the season and do not exist for most regions outside the EU's declaration systems.
Geospatial foundation models (GFMs) are pretrained on large volumes of unlabelled satellite imagery and are expected to
reduce the number of labels a crop map needs and to carry over to regions they were not trained in. For dense,
pixel-level crop-type segmentation with a frozen encoder, neither expectation has been tested directly. It is not
known how many labelled parcels a GFM needs to match a model trained from scratch, nor how much of its accuracy
remains in another country or climate.

**Why it matters.** A frozen GFM is attractive because one pretrained encoder could serve many regions, each with a
small decoder trained on few labels. If the encoder needs as many labels as a supervised model, the saving in
labelling does not exist; if its accuracy collapses outside the region it was trained in, maps produced with it in
new regions fail without any warning. The label budget at which a GFM pays off, and the boundary beyond which it
stops transferring, decide where new labels must be collected and where an existing model can be reused.

**Gap.** The evidence so far settles neither question. On a multi-domain benchmark, GFMs did not consistently
outperform supervised models, and with limited labels a supervised U-Net outperformed most of them (Marsocci et al.,
2024). On a national crop benchmark, a frozen GFM scored far below a supervised specialist (Lauber et al., 2026). Most
crop-specific evidence classifies per-pixel or per-parcel time series rather than segmenting dense maps, and counts
labels in pixels or samples rather than in parcels, the unit of annotation effort (Reuss et al., 2025). Transfer
studies hold out a few countries or regions and read the loss against distance by inspection (Nyborg et al., 2022;
Chang et al., 2025; Shang et al., 2026); we found none that separates the effect of a national border, where
labelling conventions change, from that of agro-climate. The current GFMs range from token-grid vision transformers
(Szwarcman et al., 2024; Jakubik et al., 2025) to per-pixel temporal embeddings (Feng et al., 2025), and they have
not been compared on the same chips, labels and decoder capacity. The harmonised labels of EuroCrops (Schneider et
al., 2023) make such a comparison possible across many countries.

**Contribution.**

- A controlled benchmark of four frozen GFMs against a U-Net trained from scratch for dense crop-type segmentation.
  Label budgets are counted in parcels per class, decoder capacity is matched, splits follow spatial blocks, and
  Macro-F1 is reported per pixel and per parcel. The benchmark identifies the label budget at which each GFM matches
  the supervised baseline, or shows that it does not.
- A transfer study in which a model is trained in each of 19 European countries and 12 GAEZ agro-ecological zones and
  tested in all the others, with the same labelled area for every model and balanced, spatially spread samples.
  Three loss matrices (country by country, zone by zone, country by zone) separate the effect of national borders
  from that of agro-climate, and the loss is related to climate differences after accounting for geographic distance.
- A test of whether a model trained on an agro-ecological zone, across national borders, can replace a national
  model on that country's land.

**Approach.** The labels are the EuroCrops 2021 declarations, harmonised to 21 crop species in 9 groups. The inputs
are twelve monthly Sentinel-2 L2A and Sentinel-1 RTC composites on 2.24 km chips at 10 m. The label-budget study
covers all of Estonia (7,402 chips) at 5, 20 and 100 per cent of the parcels per class
([results](../results_estonia/REPORT.md)). The transfer study draws, for each of the 19 countries and 12 zones,
250,000 ha of labelled training land and a test set of at least 50 blocks
([plan](../../research/transferability/transfer_setup.md); the reasoning behind each choice is in the
[decision record](../../research/transferability/transfer_plan.md)). TerraMind, THOR, Prithvi-EO-2.0 and TESSERA v2
remain frozen while a decoder is trained on their features; the baseline is a ResNet-18 U-Net trained from scratch on
the same inputs. The zones follow the GAEZ v5 classification (FAO and IIASA, 2025).

**Expected results.** In Estonia ([results](../results_estonia/REPORT.md)), frozen pretraining is not a uniform
advantage. With all labels, frozen TerraMind leads per pixel (Macro-F1 0.661 against 0.616 for the U-Net). With 5 per
cent of the parcels, only TESSERA v2 remains above the U-Net (0.504 against 0.467 per pixel, 0.502 against 0.446 per
parcel), and the token-grid models fall below it. Per-pixel temporal embeddings therefore appear to be the most
label-efficient representation, although each of these results comes from one draw and one seed and has no interval
yet. For transfer, we expect the loss to depend more on the difference in agro-climate than on crossing a border,
with the shift of the crop calendar into the Mediterranean zones as its largest component. We also expect the frozen
GFMs to lose less than the U-Net, and zone models to come close to national models on the countries they cover
([the three loss matrices](../../research/transferability/transfer_setup.md#the-matrices-we-will-get)).

**Sources in this repository:**

- Crop segmentation in Estonia, every arm frozen and fine-tuned, per pixel and per parcel:
  [docs/phase1/results_estonia/REPORT.md](../results_estonia/REPORT.md).
- The spatial transferability plan: [docs/research/transferability/transfer_setup.md](../../research/transferability/transfer_setup.md),
  with the record of its decisions in [transfer_plan.md](../../research/transferability/transfer_plan.md).

## References

- Chang et al. (2025). On the generalizability of foundation models for crop type mapping. arXiv:2409.09451.
- FAO and IIASA (2025). Global Agro-Ecological Zones version 5 (GAEZ v5). https://github.com/un-fao/gaezv5/wiki
- Feng et al. (2025). TESSERA: temporal embeddings of surface spectra for Earth representation and analysis. arXiv:2506.20380.
- Jakubik et al. (2025). TerraMind: large-scale generative multimodality for Earth observation. ICCV. arXiv:2504.11171.
- Lauber, Turkoglu, Ledain and Aasen (2026). SwissCrop25: a national multi-year benchmark for operational crop mapping. arXiv:2608.09497.
- Marsocci et al. (2024). PANGAEA: a global and inclusive benchmark for geospatial foundation models. arXiv:2412.04204.
- Nyborg, Pelletier, Lefèvre and Assent (2022). TimeMatch: unsupervised cross-region adaptation by temporal shift estimation. ISPRS Journal of Photogrammetry and Remote Sensing 188, 301 to 313.
- Reuss, Macdonald, Becker, Richter et al. (2025). The EuroCropsML time series benchmark dataset for few-shot crop type classification in Europe. Scientific Data 12, 664. doi:10.1038/s41597-025-04952-7.
- Schneider, Schelte, Schmitz and Körner (2023). EuroCrops: the largest harmonized open crop dataset across the European Union. Scientific Data 10, 612. doi:10.1038/s41597-023-02517-0.
- Shang, Das and Eldawy (2026). Benchmarking geospatial foundation models for agriculture applications. arXiv:2606.29664.
- Szwarcman et al. (2024). Prithvi-EO-2.0: a versatile multi-temporal foundation model for Earth observation applications. arXiv:2412.02732.
