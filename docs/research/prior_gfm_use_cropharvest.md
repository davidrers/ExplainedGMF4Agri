# Prior foundation model evaluations on CropHarvest

Literature check performed 11 September 2026 for the MSc thesis *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions*.

## Verdict

**CropHarvest is the opposite case to EuroCropsML: it is one of the most heavily used foundation model benchmarks in agricultural remote sensing.** Its three classification tasks are a standard evaluation for the pixel-time-series model family, and Presto, Galileo, AnySat and OlmoEarth all report numbers on it. Running Presto or Galileo on CropHarvest would therefore be a **replication**, valuable only as a sanity check that the thesis pipeline reproduces published figures. By contrast, **no evaluation of AlphaEarth Foundations, TESSERA, TerraMind or THOR on CropHarvest was found**, so all four of the thesis's currently-selected models would be a first on this dataset. The honest framing is that CropHarvest is a well-trodden benchmark where the thesis's contribution would be extending it to the embedding-tier and patch-tier models that the pixel-time-series community has not tested, plus the explainability and uncertainty layer, which nobody has applied to it.

## The dataset and its original benchmark

**Tseng, G., Zvonkov, I., Nakalembe, C., Kerner, H. (2021). CropHarvest: a global dataset for crop-type classification.** NeurIPS Datasets and Benchmarks 2021. [Proceedings entry](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/54229abfcfa5649e7003b83dd4755294-Abstract-round2.html)

90,480 geolocated points harmonised from 20 source datasets. Each point is a pixel time series of 12 monthly timesteps and 18 channels, combining Sentinel-2 (10 bands), Sentinel-1 (VV, VH), ERA5 (precipitation, temperature), NDVI, SRTM topography and Dynamic World land cover. Three benchmark tasks: Kenya (1,345 training points), Brazil (203) and Togo (1,319 training, 306 test). Metrics are F1 at a 0.5 threshold and AUC ROC. The structural match to the thesis is strong for the smallholder and tropical objective and weak on one axis: CropHarvest tasks are largely binary or few-class, whereas EuroCropsML is a 15 to 20 class problem.

## Published results

The consolidated table below draws on Tseng et al. (2024) and Tseng et al. (2025). All figures are as reported by their respective authors and have not been re-run.

### F1 (Tseng et al., 2024, Presto, Table 2)

| Method | Kenya | Brazil | Togo | Mean |
|---|---|---|---|---|
| Random Forest | 0.559 | 0.000 | 0.756 | 0.441 |
| MOSAIKS-1D | 0.790 | 0.746 | 0.679 | 0.738 |
| TIML (prior SOTA) | 0.838 | 0.835 | 0.732 | 0.802 |
| **Presto (regression head, full)** | 0.816 | 0.891 | 0.798 | **0.835** |
| Presto (no Dynamic World) | 0.861 | 0.888 | 0.760 | 0.836 |

### AUC ROC (same paper, Table 12)

| Method | Kenya | Brazil | Togo | Mean |
|---|---|---|---|---|
| Random Forest | 0.578 | 0.941 | 0.892 | 0.803 |
| MOSAIKS-1D | 0.693 | 0.890 | 0.836 | 0.806 |
| TIML | 0.794 | 0.988 | 0.890 | 0.890 |
| **Presto** | 0.834 | 0.997 | 0.921 | **0.917** |

### Galileo's cross-model comparison (Tseng et al., 2025, Table 6)

| Model | Togo | Brazil | Kenya | BreizhCrops |
|---|---|---|---|---|
| Presto (ViT-Presto, 0.4M) | 75.5 | 98.8 | 84.0 | 63.0 |
| AnySat (ViT-B) | 73.4 | 76.7 | 75.5 | 66.1 |
| Galileo ViT-Nano (0.8M) | 73.5 | 76.4 | 84.5 | 67.3 |
| Galileo ViT-Tiny (5.3M) | 74.7 | 97.2 | 85.4 | 69.0 |
| Galileo ViT-Base (85.0M) | 74.8 | 99.3 | 84.2 | 73.0 |

Note that the small Galileo variants are close to the base model on the CropHarvest tasks, which is the same "scale buys little at low label counts" signal the thesis expects to find.

### Other evaluations

- **Herzog, Bastani et al. (2025). OlmoEarth.** [arXiv:2511.13655](https://arxiv.org/abs/2511.13655). On CropHarvest-PRC, 73.4 to 76.1 % accuracy under kNN and linear probing, rising to 75.4 to 81.8 % fine-tuned. The release claims OlmoEarth outperforms DINOv3, Prithvi, TerraMind, CROMA, Panopticon, Satlas and Galileo, and matches AlphaEarth under kNN while beating it substantially after fine-tuning.
- **Nedungadi, V., Xiong, X., Rußwurm, M., Athanasiadis, I. N. (2026). Foundation models meet agriculture.** [arXiv:2608.30392](https://arxiv.org/abs/2608.30392). On CropHarvest Kenya, a plain supervised Transformer reaches 0.89 F1 against frozen Galileo's 0.70; on BreizhCrops the Transformer reaches 0.60 against frozen CropFM's 0.38. AlphaEarth is explicitly excluded for temporal-granularity mismatch and TESSERA is excluded as an embeddings product. This is the most directly threatening result for the thesis's premise and must be cited.
- **Tseng, G. et al. (2022). TIML: task-informed meta-learning for agriculture.** [arXiv:2202.02124](https://arxiv.org/abs/2202.02124). The pre-Presto state of the art, numbers above.
- **Tong, X.-Y., Wang, S. (2026). Invariant features for global crop type classification.** [arXiv:2509.03497](https://arxiv.org/abs/2509.03497). Uses CropHarvest with a purpose-built CropNet and TempCNN-style baselines rather than a foundation model.
- **Butsko, C., Van Tricht, K., Tseng, G. et al. (2025). Deploying geospatial foundation models in the real world: lessons from WorldCereal.** [arXiv:2508.00858](https://arxiv.org/abs/2508.00858). A Presto case study on operational deployment, relevant context for the thesis discussion though not a CropHarvest benchmark.

## Strong non-foundation-model baselines

Random Forest at 0.441 mean F1 (and 0.000 on Brazil, a useful illustration of small-sample failure), MOSAIKS-1D at 0.738, TIML at 0.802 and a supervised Transformer at 0.89 on Kenya (Nedungadi et al., 2026). The last of these is the real bar: a from-scratch supervised sequence model beats frozen GFM features on CropHarvest Kenya by roughly 19 F1 points.

## First versus replication, model by model

| Model | Verdict on CropHarvest |
|---|---|
| TerraMind | **First.** Named as a comparison target in OlmoEarth's release but no CropHarvest number located. It also cannot consume CropHarvest's pixel time series directly, for the same reason it cannot consume EuroCropsML's. |
| THOR | **First.** No agricultural evaluation of any kind located. |
| AlphaEarth Foundations | **First.** Explicitly excluded from the closest comparable study (Nedungadi et al., 2026) on temporal-granularity grounds. Note the practical obstacle: CropHarvest points span many years and AlphaEarth annual layers begin in 2017, so year alignment would need care. |
| TESSERA | **First.** Feng et al. (2026) benchmark on TreeSatAI-TS, PASTIS-R, a new Austrian Crop dataset, Biomassters and Borneo canopy height, not CropHarvest. |
| Presto | **Replication.** 0.835 mean F1 is the published reference. |
| Galileo | **Replication.** Table above is the published reference. |
| OlmoEarth | **Replication** on CropHarvest-PRC. |
| AnySat | **Replication.** |

## How hard the search was

Searches combined "CropHarvest" with foundation model, benchmark, few-shot, frozen encoder, linear probe, and with each of Presto, Galileo, AnySat, OlmoEarth, TESSERA, AlphaEarth, TerraMind, THOR and Prithvi. The Presto, Galileo, OlmoEarth and TESSERA papers were read for their evaluation tables, and the Foundation Models Meet Agriculture survey (2026) was used as a cross-check on which models the agricultural community has and has not tested. A Semantic Scholar citation enumeration was attempted but returned the wrong record, since the CropHarvest paper has no arXiv identifier; the gap was covered by title and keyword search instead. Coverage of indexed literature to September 2026 is good for the pixel-time-series family and weaker for the patch-encoder family, where a CropHarvest evaluation would in any case be architecturally awkward.
