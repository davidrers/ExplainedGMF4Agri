# Prior foundation model evaluations on EuroCropsML

Literature check performed 11 September 2026 for the MSc thesis *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions*.

## Verdict

**No published or preprint work has evaluated a released geospatial foundation model on EuroCropsML.** Benchmarking TerraMind, THOR, AlphaEarth Foundations or TESSERA on this dataset would be a **first** in every case, and the same holds for the recommended additions (Presto, Galileo, OlmoEarth, Prithvi-EO-2.0). The dataset's own benchmark papers use meta-learning, supervised transfer and self-supervised pretraining performed from scratch by the authors on Sen12MS-CR-TS and ERA5, not any publicly distributed pretrained encoder. Several 2026 papers that do use published GFMs cite EuroCropsML in their related work but evaluate elsewhere. The thesis can therefore state a genuine novelty claim, with the important qualifier that the claim is about this dataset, not about GFMs for crop classification in general, which is a crowded field (see the companion note on CropHarvest).

## What has actually been run on EuroCropsML

**Reuss, J., Macdonald, J., Becker, S., Gikalo, E., Schultka, K., Richter, L., Körner, M. (2026). Benchmarking for practice: few-shot time-series crop-type classification on the EuroCropsML dataset.** *ISPRS Open Journal of Photogrammetry and Remote Sensing* 19, 100117. [arXiv:2504.11022](https://arxiv.org/abs/2504.11022)

This is the authoritative baseline paper and the one the thesis must beat or at least sit alongside. Protocol: a Transformer encoder with sinusoidal positional encoding and a linear head; K in {1, 5, 10, 20, 100, 200, 500} shots; five seeds; accuracy and Cohen's kappa; two transfer scenarios, Latvia to Estonia and Latvia plus Portugal to Estonia, with 103 to 127 crop classes per region. Self-supervised variants were pretrained by the authors on Sen12MS-CR-TS plus ERA5. Headline accuracies:

| Scenario | K | No pretraining | Transfer learning | MAML | FOMAML | ANIL | TIML (encoder) |
|---|---|---|---|---|---|---|---|
| LV to EE | 1 | 0.171 | 0.143 | 0.176 | 0.206 | 0.189 | **0.247** |
| LV to EE | 5 | **0.331** | 0.203 | 0.259 | 0.228 | 0.250 | 0.247 |
| LV to EE | 20 | 0.490 | 0.365 | **0.507** | 0.476 | 0.490 | 0.423 |
| LV to EE | 100 | 0.518 | 0.464 | 0.552 | 0.516 | **0.561** | 0.494 |
| LV+PT to EE | 20 | 0.490 | 0.292 | 0.452 | 0.445 | **0.480** | 0.438 |
| LV+PT to EE | 100 | 0.518 | 0.405 | 0.517 | 0.512 | **0.538** | 0.488 |

Three observations matter for the thesis. First, MAML-family methods beat supervised transfer by roughly 31 % relative overall accuracy at 20 shots, but the margin over the strongest simple baseline is small. Second, **training from scratch on the target beats every transfer method at 5 shots and is within a few points at 20 and 100 shots**, which is an uncomfortable and highly relevant result: the bar a frozen GFM must clear is not "beat transfer learning", it is "beat a Transformer trained from scratch on K samples". Third, adding Portugal to the pretraining pool degrades transfer to Estonia substantially, which is direct evidence for the geographic-domain-shift effect the thesis's cross-country arm is designed to measure.

**Reuss, J., Gikalo, E., Körner, M. (2026). The EuroCropsML time series benchmark dataset for few-shot crop type classification in Europe.** *Scientific Data* 12. [doi:10.1038/s41597-025-04952-7](https://www.nature.com/articles/s41597-025-04952-7); [arXiv:2407.17458](https://arxiv.org/abs/2407.17458)

The dataset paper. 706,683 parcels, 176 classes, 13 Sentinel-2 L1C bands, up to 216 timesteps in 2021, Estonia, Latvia and Portugal. Technical validation uses the same Transformer encoder at 1, 5, 10, 20, 100, 200 and 500 shots, with class-restricted pretraining variants at 81 and 93 classes and a 60/20/20 fine-tuning split. No foundation model is involved.

**Reuss, J., Gikalo, E., Körner, M. (2026). DirPA: addressing prior shift in imbalanced few-shot crop-type classification.** [arXiv:2603.12905](https://arxiv.org/abs/2603.12905), and the earlier **Mind the Gap: bridging prior shift in realistic few-shot crop-type classification**, [arXiv:2511.16218](https://arxiv.org/abs/2511.16218)

Methodological work on class-prior shift by the dataset authors. Notable because DirPA trains and evaluates on **EuroCropsML 2.0**, an extension covering Austria (2,589,192 parcels, 98 classes), Belgium, Flanders, Wallonia and Estonia (175,905 parcels, 127 classes) among others. No published GFM is used. The existence of version 2.0 is a live question for the thesis and should be raised with the supervisors; the local copy is version 1.

## Work that cites EuroCropsML but evaluates elsewhere

- **Lauber, T., Turkoglu, M. O., Ledain, S., Aasen, H. (2026). SwissCrop25.** [arXiv:2608.09497](https://arxiv.org/abs/2608.09497). Evaluates Galileo frozen and fine-tuned against U-TAE and TSViT on a Swiss national benchmark. Frozen Galileo-Nano reaches 14.1 % mIoU, fine-tuned 30.4 %, against TSViT at 48.1 %. Cites EuroCropsML as a European benchmark but does not use it.
- **Shang, Z., Das, S., Eldawy, A. (2026). Benchmarking geospatial foundation models for agriculture applications.** [arXiv:2606.29664](https://arxiv.org/abs/2606.29664). Prithvi, SpectralGPT and SatMAE fine-tuned on USDA Cropland Data Layer chips across four US states. Cites EuroCropsML; does not use it.
- **Nedungadi, V., Xiong, X., Rußwurm, M., Athanasiadis, I. N. (2026). Foundation models meet agriculture.** [arXiv:2608.30392](https://arxiv.org/abs/2608.30392). Galileo, CropFM and TabPFN on CY-Bench, YieldSAT, CropHarvest Kenya, BreizhCrops and BloomBench. EuroCrops and EuroCropsML explicitly not included.
- **Scarlat, A. N., Plajer, I. C., Băicoianu, A. (2026). A modelling and evaluation framework for EuroCrops-driven Sentinel-2 crop segmentation.** [arXiv:2606.00676](https://arxiv.org/abs/2606.00676). Uses the parent **EuroCrops** dataset for U-Net semantic segmentation from 256 x 256 patches across Austria, France, Germany, Slovakia and Czechia, with Random Forest baselines. Reports 0.7665 mIoU internally with degradation on unseen regions. No foundation model.

The parent dataset was itself updated this year: **EuroCrops v2.0**, *Earth System Science Data* 18, 4075 (2026), [essd.copernicus.org/articles/18/4075/2026](https://essd.copernicus.org/articles/18/4075/2026), now multi-annual and linked to EU survey, statistical and Earth observation products. No GFM evaluation on EuroCrops v2.0 was found either.

## Strong non-foundation-model baselines the thesis must beat

1. Transformer encoder trained from scratch on the K target samples: 0.331 at 5 shots, 0.490 at 20, 0.518 at 100 (LV to EE), Reuss et al. 2026.
2. ANIL, the best meta-learner overall: 0.561 at 100 shots (LV to EE).
3. MAML: 0.507 at 20 shots (LV to EE).
4. Supervised transfer learning, consistently the weakest pretrained option and below training from scratch at every K tested.
5. Random Forest on handcrafted spectral and phenological features, which the thesis supplies itself through the TIMESAT and monthly-grid baselines. No published Random Forest number on EuroCropsML was located, so the thesis will be establishing that reference itself.

Note that all published numbers are **accuracy**, while the thesis protocol specifies Macro-F1 with bootstrap confidence intervals. Direct numerical comparison is therefore not possible without recomputing, and the thesis should report both metrics or state clearly that its numbers are not comparable line by line with Reuss et al.

## How hard the search was

Searches combined "EuroCropsML" and "EuroCrops" with each of: foundation model, embeddings, pretrained representation, frozen encoder, linear probe, few-shot, transfer, AlphaEarth, TESSERA, Presto, Galileo, TerraMind, THOR, Prithvi, OlmoEarth and self-supervised. The full Semantic Scholar citation list of the dataset paper ([arXiv:2407.17458](https://arxiv.org/abs/2407.17458)) was enumerated and inspected. Ten citing works were found; two use published GFMs (SwissCrop25 with Galileo; Shang et al. with Prithvi, SpectralGPT and SatMAE) and neither runs them on EuroCropsML. The conclusion is robust for indexed literature to September 2026. Grey literature, theses and unindexed preprints were not exhaustively covered.
