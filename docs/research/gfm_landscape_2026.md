# The geospatial foundation model landscape, September 2026

Literature and model-hub survey prepared for the MSc thesis *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions* (D. Reyes, ITC, University of Twente). Compiled 11 September 2026. Covers Task 1 (what exists now) and Task 2 (recommendation on the selected GFM set).

Verification standard: every claim below is traced to a primary source (arXiv entry, published paper, GitHub repository, Hugging Face model card) or to direct inspection of the packages installed in the thesis environment (`terratorch 1.2.6`, `torchgeo 0.8.1`, `eurocropsml 0.4.1`). Claims that could not be confirmed are collected in the final section.

---

## 1. Executive summary and recommendation

The selected set of four (TerraMind, THOR, AlphaEarth Foundations, TESSERA) has one structural problem that dominates every other consideration, and it is not the one the proposal anticipated.

**EuroCropsML delivers one spatial-median spectral vector per parcel per acquisition date, that is, a `(T, 13)` pixel time series with no spatial patch.** Of the four selected models, only TESSERA is architecturally able to consume that representation through its own encoder. AlphaEarth has no released encoder at all and is consumed as a precomputed annual embedding, which discards intra-season phenology. TerraMind and THOR are both single-timestamp image-patch encoders: they require an `H x W` multispectral image, they have no temporal axis, and they cannot be fed a `(T, 13)` vector sequence under any configuration. Using them at all requires rebuilding the dataset from raw Sentinel-2 tiles, which is a separate data-engineering work package measured in hundreds of gigabytes and GPU-hours.

The consequence for Phase 2 is severe and has not been stated in the proposal. The backbone tier exists in order to support Integrated Gradients, Occlusion and AttnLRP attribution back to bands and dates. As currently constituted, **no model in the backbone tier actually consumes bands and dates in the form the thesis dataset provides them**, so the per-date attribution that RQ2.1 asks for cannot be produced on either TerraMind or THOR without first constructing a patch dataset, and even then the temporal axis is synthesised by the pipeline rather than seen by the encoder.

There is a clean fix, and it is cheap. A family of open-weight, pixel-time-series foundation models now exists and is mature: **Presto** (Tseng et al., 2024), **Galileo** (Tseng et al., 2025) and, decisively, **TESSERA itself, whose encoder weights are downloadable under CC-0** and which the thesis currently mis-classifies as embedding-tier only. Each of these consumes a `(T, bands)` pixel sequence natively, each is small enough to run on a single-student compute budget, and each supports gradient attribution directly onto the band-by-date grid.

**Headline recommendation.**

| Action | Model | One-line reason |
|---|---|---|
| **Promote** | TESSERA | Encoder weights are public under CC-0 (verified on GitHub); it is a backbone-tier model, not an embedding-tier one, and it is the only currently-selected model that natively consumes a Sentinel-2 pixel time series. |
| **Add (primary)** | Galileo (ViT-Base, 85M) | Open MIT weights; consumes a single-pixel space-time block; the reference pixel-time-series GFM of 2025 to 2026; fixes the thin backbone tier at near-zero ingest cost. |
| **Add (primary)** | Presto (0.4M) | Open MIT weights; the smallest credible GFM; a 0.4M-parameter model is a devastating control for RQ1 because if it matches the 300M-parameter patch models at low K, that is the finding. |
| **Keep** | AlphaEarth Foundations | Keep as the operational-reality and negative-control arm; the annual-aggregation caveat is now independently corroborated (Feng et al., 2026; Ma et al., 2026) and is a publishable result rather than a flaw. |
| **Demote to a gated sub-study** | TerraMind | Retain as the sole patch-tier representative and the only route to spatial attribution and AttnLRP, but only on a size-filtered parcel subset and behind an explicit go/no-go gate on the raw-tile ingest. |
| **Demote to stretch** | THOR | Weights are genuinely public (contrary to the risk the proposal flagged), but THOR shares TerraMind's single-timestamp patch limitation, requires a non-core `thor_terratorch_ext` dependency absent from the installed `terratorch 1.2.6`, and its own head-to-head comparison paper concludes that patch size and decoder choice explain more variance than model identity (Schindlegger et al., 2026). It therefore adds cost without adding an axis. |

**Recommended primary set of five:** TESSERA, Galileo, Presto, AlphaEarth, TerraMind.
**Recommended stretch set:** OlmoEarth, THOR, Prithvi-EO-2.0.

This set has four models with downloadable weights (TESSERA, Galileo, Presto, TerraMind) against two previously, and three of those four consume the thesis data representation natively, against zero previously. It also spans the design axes the thesis actually wants to compare: pixel time series against image patch, tiny against large, open weights against embeddings-as-a-product.

---

## 2. Comparison table of all candidate models

### 2.1 Identity and provenance

| Model | Authors / institution | Release | Venue or identifier | Licence | Parameters |
|---|---|---|---|---|---|
| TerraMind v1 | Jakubik et al., IBM Research and ESA Φ-lab | Apr 2025 | ICCV 2025; [arXiv:2504.11171](https://arxiv.org/abs/2504.11171) | Apache-2.0 | tiny / small / base / large; base is ViT-B class (depth 12, 12 heads), large is ViT-L class (depth 24, 16 heads), verified in installed source |
| THOR v1 | Forgaard, Reksten, Waldeland, Marsocci, Longépé, Kampffmeyer, Salberg; Norwegian Computing Center and ESA Φ-lab (FM4CS) | Jan 2026 | [arXiv:2601.16011](https://arxiv.org/abs/2601.16011) | MIT (GitHub repo) / Apache-2.0 (HF card) | THOR-Tiny, THOR-Base; counts not disclosed |
| AlphaEarth Foundations | Brown et al., Google DeepMind | Jul 2025 | [arXiv:2507.22291](https://arxiv.org/abs/2507.22291) | Embeddings CC-BY-4.0; weights not released | not disclosed |
| TESSERA v1 | Feng, Atzberger, Jaffer, Knezevic et al., University of Cambridge et al. | Jun 2025, rev. Apr 2026 | [arXiv:2506.20380](https://arxiv.org/abs/2506.20380) | CC-BY-4.0 (paper); weights and embeddings CC-0, software MIT | encoder 45.7M (projector 1.34B, discarded at inference) |
| TESSERA v2 | Feng, Jaffer, Shokar et al., Cambridge | Jul 2026 | [arXiv:2607.03949](https://arxiv.org/abs/2607.03949) | as above | teachers 0.5B / 1B / 2B; distilled students 1.07M to 43.83M |
| Galileo | Tseng, Fuller, Reil, Herzog, Beukema, Bastani, Green, Shelhamer, Kerner, Rolnick; NASA Harvest, Mila, McGill, ASU, Ai2, Carleton | Feb 2025 | [arXiv:2502.09356](https://arxiv.org/abs/2502.09356) | MIT | ViT-Nano 0.8M, ViT-Tiny 5.3M, ViT-Base 85.0M |
| Presto | Tseng, Cartuyvels, Zvonkov, Purohit, Rolnick, Kerner; Mila, McGill, KU Leuven, UMD, ASU | Apr 2023, v4 Feb 2024 | IJCAI 2023; [arXiv:2304.14065](https://arxiv.org/abs/2304.14065) | MIT | 0.4M |
| OlmoEarth | Herzog, Bastani et al., Allen Institute for AI | Nov 2025 | [arXiv:2511.13655](https://arxiv.org/abs/2511.13655) | restricted (no military, defence or extractive use) | Nano 1.4M, Tiny 6.2M, Base 90M, Large 300M |
| Prithvi-EO-2.0 | IBM and NASA, 42 authors, 12 institutions | Dec 2024 | [arXiv:2412.02732](https://arxiv.org/abs/2412.02732) | Apache-2.0 | 100M, 300M, 600M (plus TL variants) |
| Clay v1 / v1.5 | Clay Foundation | 2024 to 2025 | project docs, no peer-reviewed paper found | Apache-2.0 (model), CC-BY / ODC-By for embeddings | ViT-B class |
| DOFA | Xiong, Wang, Zhu et al., TU Munich | 2024 to 2025 | ISPRS / arXiv | CC-BY-4.0 | base and large ViT |
| Copernicus-FM | Wang, Braham, Xiong, Liu, Albrecht, Zhu; TU Munich | Mar 2025 | ICCV 2025; [arXiv:2503.11849](https://arxiv.org/abs/2503.11849) | CC-BY-4.0 | ViT-B |
| Panopticon | Waldmann et al. | 2025 | see TorchGeo registry | see model card | ViT-B |
| CROMA | Fuller, Millard, Green | 2023 | NeurIPS 2023 | MIT | base and large |
| AnySat | Astruc, Gonthier, Mallet, Landrieu | Dec 2024 | [arXiv:2412.14123](https://arxiv.org/abs/2412.14123) | see repo | ViT-B class |
| TerraFM | MBZUAI Oryx | Jun 2025 | [arXiv:2506.06281](https://arxiv.org/abs/2506.06281) | see repo | ViT-B / L |
| AgriFM | Wang et al. | May 2025; RSE 2026 | [arXiv:2505.21357](https://arxiv.org/abs/2505.21357) | Apache-2.0 | Video Swin |
| SkySense V2 | Zhang et al., Ant Group and collaborators | Jul 2025 | ICCV 2025; [arXiv:2507.13812](https://arxiv.org/abs/2507.13812) | not confirmed | large |
| SpectralGPT | Hong et al. | 2023 | TPAMI | open | ViT |
| CropFM | Nedungadi, Xiong, Rußwurm, Athanasiadis; WUR, Bonn | Aug 2026 | [arXiv:2608.30392](https://arxiv.org/abs/2608.30392) | not released as a standalone model | Presto-class |

### 2.2 Technical fit to the thesis task

| Model | Native input | Temporal handling | Bands / modalities | Distribution | In `terratorch 1.2.6`? | In `torchgeo 0.8.1`? | Tier | Consumes `(T, 13)` directly? |
|---|---|---|---|---|---|---|---|---|
| TerraMind v1 | image patch, default 224 x 224, patch size 16 | **none**; single timestamp | S2L2A (12 bands), S2L1C, S1GRD, S1RTC, DEM, RGB, LULC, NDVI | HF `ibm-esa-geospatial` | **yes**, 16 entries | no | backbone | **no** |
| THOR v1 | image patch, flexible patch size and ground cover (1 km to 100 km) | not documented | S1, S2, S3 OLCI, S3 SLSTR; 10 m to 1000 m | HF `FM4CS`, GitHub `FM4CS/THOR` | **no**; needs `thor_terratorch_ext` | no | backbone | **no** |
| AlphaEarth | none (product, not encoder) | annual composite, 2017 to 2025 | S2, Landsat, S1, PALSAR-2, GEDI, ERA5-Land, GRACE, GLO-30, NLCD, text | GEE `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, 64-D | n/a | n/a | embedding | n/a (bypasses the question) |
| TESSERA v1 / v2 | **pixel time series** | ~40 sampled acquisitions per year, encoded then pooled to an annual 128-D vector | S2 10 bands + S1 VV/VH | HF weights (CC-0) **and** precomputed embeddings via GeoTessera | no | no | **backbone** (and embedding) | **yes**, with an S1 gap |
| Galileo | space-time block `[B, H, W, T, bands]`, **H = W = 1 permitted** | explicit time axis, monthly | S1, S2, ERA5, SRTM, Dynamic World, location, derived maps | GitHub + HF `nasaharvest/galileo` | no | no | **backbone** | **yes** |
| Presto | `[B, T, 18]` pixel time series | 1 to 24 monthly steps | S2 (10 bands), S1, ERA5, SRTM, NDVI, Dynamic World, lat/lon | GitHub `nasaharvest/presto` | no | no | **backbone** | **yes** |
| OlmoEarth | multimodal image time series; spatial crops from 1 x 1 to 96 x 96 tokens | 3 to 12 monthly steps | S1, S2, Landsat-8 plus six derived maps | HF + GitHub + OlmoEarth Platform | no | no | **backbone** | **yes** (1 x 1 crop) |
| Prithvi-EO-2.0 | image patch 224 x 224 | `num_frames` up to 4 | 6 HLS bands | HF `ibm-nasa-geospatial` | **yes**, 8 entries | no | backbone | **no** |
| Clay v1 / v1.5 | image patch | single timestamp plus time embedding | multi-sensor | HF `made-with-clay` | partially (`timm_clay_v1_base`; `clay_v15` code present but unregistered) | no | backbone | **no** |
| DOFA | image patch | single timestamp | any spectral, via wavelength hypernetwork | HF | **yes**, 3 entries | **yes** | backbone | **no** |
| Copernicus-FM | image patch | single timestamp | any Sentinel modality plus metadata | HF `wangyi111/Copernicus-FM` | no | **yes** | backbone | **no** |
| Panopticon | image patch | single timestamp | S1, S2, WV2/3, NAIP | HF | no | **yes** | backbone | **no** |
| CROMA | image patch | single timestamp | S1 + S2 | HF | no | **yes** | backbone | **no** |
| Scale-MAE | image patch | single timestamp | RGB, scale-aware | HF | **yes** (source present) | **yes** | backbone | **no** |
| AnySat | patches, with 1 x 1 sub-patches for time series | yes, per dataset configuration | aerial, S1, S2, MODIS; 0.2 m to 500 m | HF `g-astruc/AnySat` | no | no | backbone | partially |
| TerraFM | image patch | single timestamp | S1 + S2 | GitHub `mbzuai-oryx/TerraFM` | no | no | backbone | **no** |
| AgriFM | video patch stack (Video Swin) | yes, long series | MODIS, Landsat-8/9, S2 | GitHub `flyakon/AgriFM` | no | no | backbone | **no** |
| SkySense V2 | image patch | limited | multi-modal | not confirmed | no | no | backbone | **no** |

The registry columns were established by enumerating `terratorch.registry.BACKBONE_REGISTRY` (1400 entries, of which 113 are non-`timm`) and `torchgeo.models` weight enums (25 entries) in the thesis virtual environment on 11 September 2026.

---

## 3. Serious candidates, one subsection each

### 3.1 TESSERA (Cambridge) — reclassify as backbone tier

TESSERA encodes a full year of Sentinel-1 and Sentinel-2 observations at a single 10 m pixel into a 128-dimensional embedding, using dual-branch Transformer encoders over roughly 40 sampled acquisitions per year, GRU pooling, a fusion MLP and Barlow Twins self-supervised pretraining (Feng et al., 2026, [arXiv:2506.20380](https://arxiv.org/abs/2506.20380)). The v1 encoder is 45.7M parameters; the 1.34B projector is discarded at inference. TESSERA v2 ([arXiv:2607.03949](https://arxiv.org/abs/2607.03949)) adds 0.5B, 1B and 2B teachers and four distilled students from 1.07M to 43.83M parameters, with Matryoshka embeddings truncatable to 16, 32 or 64 dimensions.

The important correction to the thesis specification: the GitHub repository ([github.com/ucam-eo/tessera](https://github.com/ucam-eo/tessera)) states that **embeddings and weights are released under CC-0** and the software under MIT, with encoder-only checkpoints of roughly 221 MB for v1.1 and Hugging Face distributions for v2. The repository documents a self-hosted inference path. TESSERA therefore belongs in the backbone tier, and gradient attribution through its encoder onto the band-by-date grid is possible. Note that Fang et al. (2026, [arXiv:2601.13134](https://arxiv.org/abs/2601.13134)) list TESSERA as "MIT (closed weights)"; the repository contradicts this, and the repository is the later and more authoritative source, but the discrepancy is worth a sentence in the thesis.

Relevant result: on the Austrian Crop benchmark at 1 % labels, TESSERA reaches 66.15 weighted F1 against AlphaEarth at 37.22 and Presto at 32.74; the gap narrows but persists to 30 % labels (82.09 against 56.36 and 57.89). This is the closest published analogue to the thesis's RQ1 and it strongly favours TESSERA.

Limitation for this thesis: TESSERA expects Sentinel-1 alongside Sentinel-2, and EuroCropsML supplies only Sentinel-2. The SAR branch would have to be zero-filled or masked, which is off-distribution. This should be measured, not assumed, by comparing the self-run encoder output against the precomputed GeoTessera embedding at the same parcels.

### 3.2 Galileo (NASA Harvest and collaborators) — add, primary

Galileo is a highly multimodal Transformer trained with a dual global-and-local contrastive objective over ten remote-sensing products (Tseng et al., 2025, [arXiv:2502.09356](https://arxiv.org/abs/2502.09356)). The encoder accepts a `MaskedOutput` structure with space-time `[B, H, W, T, bands]`, space `[B, H, W, bands]`, time `[B, T, bands]` and static `[B, bands]` components plus per-modality masks, which is exactly the interface a `(T, 13)` parcel series needs. Weights are MIT-licensed on GitHub and Hugging Face (`nasaharvest/galileo`), in Nano (0.8M), Tiny (5.3M) and Base (85.0M) sizes.

Galileo's own CropHarvest pixel-time-series table (its Table 6) is the reference point:

| Model | Togo | Brazil | Kenya | BreizhCrops |
|---|---|---|---|---|
| Presto (ViT-Presto) | 75.5 | 98.8 | 84.0 | 63.0 |
| AnySat (ViT-B) | 73.4 | 76.7 | 75.5 | 66.1 |
| Galileo ViT-Nano | 73.5 | 76.4 | 84.5 | 67.3 |
| Galileo ViT-Tiny | 74.7 | 97.2 | 85.4 | 69.0 |
| Galileo ViT-Base | 74.8 | 99.3 | 84.2 | 73.0 |

A necessary counterweight: on SwissCrop25, a national multi-year crop mapping benchmark, **frozen** Galileo-Nano reached only 14.1 % mIoU against 30.4 % when fine-tuned, and both were far behind the supervised specialist TSViT at 48.1 % (Lauber et al., 2026, [arXiv:2608.09497](https://arxiv.org/abs/2608.09497)). Since the thesis is explicitly a frozen-encoder study, this is precisely the kind of result RQ1 exists to characterise, and it argues for including a strong supervised baseline rather than against including Galileo.

### 3.3 Presto (NASA Harvest, Mila, KU Leuven) — add, primary

Presto is a 0.4M-parameter Transformer over pixel time series, pretrained with structured masking so that it degrades gracefully when modalities are missing (Tseng et al., 2024, [arXiv:2304.14065](https://arxiv.org/abs/2304.14065), MIT licence, [github.com/nasaharvest/presto](https://github.com/nasaharvest/presto)). Input is `[batch, num_timesteps, bands]` with 1 to 24 timesteps, plus a Dynamic World channel, latitude and longitude, and a starting month; `presto.construct_single_presto_input()` accepts a Sentinel-2-only tensor and masks the rest. Presto embeddings are additionally distributed through Google Earth Engine (Fang et al., 2026).

Presto is valuable to this thesis for a reason beyond its accuracy. At 0.4M parameters it is three orders of magnitude smaller than TerraMind-large. If a 0.4M model matches or beats the 300M patch models at K = 5 or K = 10 on EuroCropsML, that is the single most quotable result the thesis can produce about label efficiency, and it directly answers the "does scale buy label efficiency" question that the GFM literature has left open (Corley et al., 2026). Its CropHarvest reference numbers are 0.835 mean F1 and 0.917 mean AUC ROC.

### 3.4 TerraMind (IBM and ESA Φ-lab) — keep, but gate

TerraMind v1 is a generative any-to-any multimodal model over nine modalities, accepted at ICCV 2025 ([arXiv:2504.11171](https://arxiv.org/abs/2504.11171)), Apache-2.0, and it is the best-supported model in the installed TerraTorch (16 registry entries including tiny, small, base, large, the `_tim` "thinking in modalities" variants and six tokenizers). Direct inspection of `terratorch/models/backbones/terramind/model/terramind_vit.py` confirms the constructor signature `(img_size=224, patch_size=16, ...)` with no `num_frames` or equivalent temporal argument, in contrast with `prithvi_vit.py`, which does expose `num_frames`. TerraMind is therefore single-timestamp by construction.

It should be retained for exactly one reason: it is the only route in the whole candidate set to spatial attribution, attention rollout and AttnLRP over patch tokens, which the proposal's backbone tier promises. But it should be scoped as a gated sub-study on a size-filtered parcel subset, not as a co-equal arm, because the raw Sentinel-2 tile ingest required to feed it is a work package in its own right.

One positive detail specific to this dataset: TerraMind ships an `S2L1C` modality alongside `S2L2A`, and EuroCropsML is Sentinel-2 **L1C**. Most competitors pretrain on L2A or HLS surface reflectance, so TerraMind is unusually well matched on the radiometric side even though it is badly matched on the structural side.

### 3.5 THOR (Norwegian Computing Center and ESA Φ-lab) — demote to stretch

The proposal flagged a risk that THOR weights might not be public. That risk is resolved: THOR-1.0-base is on Hugging Face under `FM4CS/THOR-1.0-base` with Apache-2.0 on the card and MIT in the GitHub repository, and the paper is [arXiv:2601.16011](https://arxiv.org/abs/2601.16011) (Forgaard et al., January 2026). THOR is the first model to unify the 10 m to 1000 m range across Sentinel-1, Sentinel-2 and Sentinel-3 OLCI and SLSTR, pretrained on a 22 TB aligned dataset on LUMI, with a compute-adaptive randomised patch-size strategy that allows any patch size at inference without retraining.

The case for demotion is not availability, it is redundancy plus friction. THOR is loaded through a separate `thor_terratorch_ext` package that is **not** present in the installed `terratorch 1.2.6`, so it is an extra dependency with its own version risk. Multi-temporal input is not documented. And the direct TerraMind-versus-THOR comparison (Schindlegger et al., 2026, [arXiv:2607.18504](https://arxiv.org/abs/2607.18504), with authors from both teams) concludes across ten use cases that "architectural design choices, patch size and decoder type in particular, explain more performance variance than model identity itself", with no single winner. Running both therefore buys a within-tier comparison whose expected information content is low, at the cost of a second full patch-ingest configuration.

### 3.6 AlphaEarth Foundations (Google DeepMind) — keep

Brown et al. (2025, [arXiv:2507.22291](https://arxiv.org/abs/2507.22291)) release 64-dimensional, 8-bit-quantised, 10 m annual embedding fields for 2017 onwards under CC-BY-4.0, fused from Sentinel-2, Landsat 8/9, Sentinel-1, PALSAR-2, GEDI, ERA5-Land, GRACE, GLO-30, NLCD and text sources. **Model weights are not released**, which fixes AlphaEarth permanently in the embedding tier. The paper reports a 23.9 % average reduction in error magnitude across 15 evaluations and explicit 1-shot, 10-shot and max-shot advantages over baselines.

The annual-aggregation caveat that the thesis already anticipates is now corroborated from three independent directions, which strengthens the thesis rather than weakening it. Ma et al. (2026, [arXiv:2601.00857](https://arxiv.org/abs/2601.00857); *Int. J. Appl. Earth Obs. Geoinf.*, doi:10.1016/j.jag.2026.105258) find AlphaEarth competitive on yield and tillage but explicitly limited in "spatial transferability", "interpretability" and "time sensitivity". Feng et al. (2026) show the 29-point Austrian Crop gap quoted above. Van der Plas et al. (2026, [arXiv:2605.18667](https://arxiv.org/abs/2605.18667)) find that on 18-class crop type classification, AlphaEarth fused with GeoCLIP reaches 63.4 % against 61.6 % for the best single model, so even in combination the absolute ceiling is modest. A within-region versus cross-region AlphaEarth crop study (agriRxiv 2025.00354) reports 91.4 to 99.1 % within-region overall accuracy but limited cross-regional transfer, which maps directly onto the thesis's in-country versus cross-country design.

One development worth a footnote: Google now offers **Custom Satellite Embeddings** through the Maps Platform at quarterly, monthly, weekly or 5-day intervals. This is a commercial product, not available through the free Earth Engine catalogue, so it does not change the thesis protocol, but it does mean the annual limitation is a property of the public release rather than of the model.

### 3.7 OlmoEarth (Allen Institute for AI) — stretch

OlmoEarth (Herzog, Bastani et al., November 2025, [arXiv:2511.13655](https://arxiv.org/abs/2511.13655)) is the most interesting near-miss. It offers Nano (1.4M), Tiny (6.2M), Base (90M) and Large (300M) sizes on Hugging Face and GitHub, processes multimodal image time series with spatial crops from 1 x 1 to 96 x 96 tokens and 3 to 12 monthly timesteps, and explicitly handles single-pixel time series for classification. It reports CropHarvest-PRC accuracies of 73.4 to 76.1 % under kNN and linear probing and 75.4 to 81.8 % fine-tuned, and the Ai2 release claims it outperforms DINOv3, Prithvi, TerraMind, CROMA, Panopticon, Satlas and Galileo, and matches AlphaEarth under kNN while beating it substantially after fine-tuning.

It is placed in the stretch set for one reason only: its licence carries use restrictions (no military, defence or extractive applications) rather than being a standard Apache-2.0 or MIT grant. For an MSc thesis this is almost certainly acceptable, but it warrants a check with the supervisors before it becomes a core deliverable.

### 3.8 Prithvi-EO-2.0 (IBM and NASA) — stretch

Prithvi-EO-2.0 (100M, 300M, 600M, plus temporal-location variants) is Apache-2.0, natively registered in the installed TerraTorch with eight entries, and is the only patch-tier candidate with a real temporal axis: inspection of `prithvi_vit.py` shows `num_frames` as a first-class argument, defaulting to 4 in v2 configurations and 3 in v1. It is restricted to six HLS bands, which is a poor match for 13-band L1C, and it still requires an `H x W` patch. The proposal's methods section already names Prithvi in the backbone-tier paragraph; the CLAUDE.md note says to treat the backbone tier as TerraMind and THOR unless Prithvi is deliberately added. Given that Prithvi is the only patch model that can represent time, and that it is already installed, adding it is nearly free **conditional on the patch ingest existing**, and it is therefore a natural companion to the TerraMind sub-study rather than a separate decision.

### 3.9 Models examined and not recommended

**Clay v1.5**: single-timestamp patch encoder, no peer-reviewed paper located, and only `timm_clay_v1_base` is actually registered in the installed TerraTorch even though `clay_v15` source is vendored. No temporal axis, so no advantage over TerraMind for this task.

**DOFA, Copernicus-FM, Panopticon, CROMA, Scale-MAE, SatMAE, SpectralGPT, TerraFM, SkySense V2**: all single-timestamp patch encoders. Copernicus-FM (Wang et al., ICCV 2025, [arXiv:2503.11849](https://arxiv.org/abs/2503.11849), CC-BY-4.0, in `torchgeo 0.8.1`) is the most attractive of these because its hypernetwork accepts arbitrary spectral configurations including all 13 L1C bands, and because Copernicus-Bench is a well-designed 15-task evaluation. It remains unsuitable because it has no temporal axis. None of these belongs in a crop-phenology thesis.

**AnySat** (Astruc et al., [arXiv:2412.14123](https://arxiv.org/abs/2412.14123)) uses 1 x 1 pixel sub-patches for time series and 10 x 10 for very-high-resolution imagery, and covers 0.2 m to 500 m. It is the closest patch-free alternative to Galileo, and Galileo's own table shows it is weaker on CropHarvest (73.4 / 76.7 / 75.5 against Galileo-Base's 74.8 / 99.3 / 84.2). It is a reasonable substitute if Galileo integration proves difficult, not a reason to add a sixth model.

**AgriFM** (Wang et al., [arXiv:2505.21357](https://arxiv.org/abs/2505.21357), Apache-2.0, *Remote Sensing of Environment* 2026) is genuinely agriculture-specific and genuinely temporal, trained on 25M samples from MODIS, Landsat-8/9 and Sentinel-2 with a modified Video Swin that synchronises temporal and spatial downsampling. It is excluded because Video Swin is irreducibly spatial: it cannot accept a 1 x 1 spatial extent. It would be an excellent choice if the thesis ever builds the patch dataset.

**CropFM** (Nedungadi et al., 2026, [arXiv:2608.30392](https://arxiv.org/abs/2608.30392)) is a purpose-built Presto extension with nine modalities and weekly timesteps, created as a research instrument inside the paper rather than released as a downloadable model. It is not a usable candidate but the paper it comes from is essential reading for the thesis discussion, since it reports that on BreizhCrops a plain supervised Transformer reaches 0.60 F1 against frozen CropFM's 0.38, and on CropHarvest Kenya 0.89 against frozen Galileo's 0.70.

---

## 4. Input-format feasibility: the `(T, 13)` question

This is the decisive practical section.

**What EuroCropsML gives.** One `.npz` per parcel containing the spatial median over the parcel's pixels at every cloud-free Sentinel-2 L1C acquisition in 2021, giving a `(T, 13)` array plus acquisition dates and a centroid coordinate. The 13 bands are confirmed from `eurocropsml.acquisition.config.S2_BANDS` in the installed package: `01, 02, 03, 04, 05, 06, 07, 08, 8A, 09, 10, 11, 12`, in that order. `T` varies per parcel and reaches up to 216 timesteps. There is no spatial patch, no Sentinel-1, and the product is top-of-atmosphere, not surface reflectance.

**Feasibility verdict per model.**

| Model | Can consume `(T, 13)` directly? | What it needs | Extra engineering cost |
|---|---|---|---|
| **Presto** | **Yes** | Composite the irregular series onto a 12-month grid; supply the Sentinel-2 subset Presto expects (the EuroCropsML 13-band set is a superset); pass the parcel centroid as `latlons` and mask S1, ERA5, SRTM and Dynamic World | **Very low.** One compositing function plus `construct_single_presto_input`. Perhaps one day. |
| **Galileo** | **Yes** | Build a `MaskedOutput` with `H = W = 1`, `T` monthly steps, the S2 band group populated and every other modality masked; supply the month index | **Low.** The repository ships `single_file_galileo.py` and a band-to-`MaskedOutput` helper. Two to three days including validation. |
| **TESSERA** | **Yes, with a caveat** | Resample to the ~40-acquisition annual sampling the encoder expects; supply the 10 Sentinel-2 bands it uses; zero-fill or mask the Sentinel-1 VV/VH branch, which EuroCropsML does not have | **Low to medium.** Self-hosted inference is documented but memory-hungry for tiles; running it on `(T, 13)` vectors rather than tiles avoids that. Budget one week, mostly for validating the S1-absent regime against the precomputed GeoTessera embeddings. |
| **OlmoEarth** | **Yes** | 1 x 1 spatial crop, 3 to 12 monthly steps, S2 modality populated | **Medium.** Newer tooling, less community documentation. |
| **AnySat** | **Partially** | 1 x 1 sub-patches for time series exist, but AnySat is configured per dataset and a new EuroCropsML configuration would have to be written | **Medium to high.** |
| **AlphaEarth** | **Not applicable** | No encoder exists. Sample the annual embedding at the parcel geometry, ideally as a zonal mean over the parcel's pixels rather than a centroid point | **Negligible**, but the `(T, 13)` series is discarded entirely and replaced by a different data product. Intra-season phenology is lost by construction. |
| **TerraMind** | **No** | Requires an `H x W` image, default 224 x 224 at patch size 16, with no time axis. The `(T, 13)` vector cannot be reshaped into a valid patch | **High.** Fetch 2021 Sentinel-2 L1C tiles from Copernicus, crop a window per parcel, store `(T, C, H, W)` stacks, run the encoder once per parcel per date, then pool over dates in the pipeline. Storage in the tens to hundreds of GB; a GPU forward pass per parcel per date; a scene-ID manifest must be locked for reproducibility. Weeks, not days. |
| **THOR** | **No** | Same as TerraMind. Flexible patch size and ground cover help with token contamination on small parcels but do not remove the requirement for an image | **High**, and additive to TerraMind's cost only in inference, not in ingest, since the same patch store can serve both. |
| **Prithvi-EO-2.0** | **No** | `H x W` patch, `num_frames` up to 4, six HLS bands | **High** (shares TerraMind's ingest) plus a 13-to-6 band reduction. |
| **Clay, DOFA, Copernicus-FM, Panopticon, CROMA, Scale-MAE, SatMAE, SpectralGPT, TerraFM, SkySense V2, AgriFM** | **No** | All require spatial extent | **High**, and none offers a temporal axis worth the cost except AgriFM. |

**Three consequences that should be written into the methods chapter.**

First, the thesis's four-model set currently contains **zero** models that both (a) consume the dataset's native representation and (b) expose gradients. TESSERA satisfies both once reclassified, and adding Presto and Galileo raises the count to three. Without that change, the Phase 2 backbone tier is an aspiration rather than a plan.

Second, the two ingest routes are not merely asymmetric in cost, they are asymmetric in what they measure. For a pixel-time-series model the encoder sees exactly the evidence the label refers to. For a patch model at a 16-pixel patch size, each token covers roughly 160 m x 160 m, so a 0.5 ha parcel sits inside a token whose footprint also contains neighbouring fields. This contamination is mild for Estonian and Latvian commercial parcels and severe for Portuguese smallholdings, and it will be confounded with model identity unless a size-stratified sensitivity stratum is reported. THOR's flexible patch size is the one genuine mitigation available, which is an argument for keeping THOR in the stretch set rather than discarding it entirely.

Third, radiometry. EuroCropsML is L1C top-of-atmosphere. TerraMind ships an explicit `S2L1C` modality; TESSERA, Presto and Galileo were pretrained largely on L2A or on CropHarvest's Earth Engine exports, and the exact level and band ordering for each must be checked against the repository constants before any number is reported. Feeding L1C reflectances through an L2A-normalised encoder without documenting the mismatch would invalidate the comparison.

---

## 5. Final ranked recommendation

### Primary set, five models

1. **TESSERA v1 encoder (45.7M), self-run on the parcel series, plus the precomputed GeoTessera embedding as a second arm.** Rank one because it is the only currently-selected model that fits the data, because it is the strongest published performer on a directly analogous label-budget crop benchmark, and because running the encoder yourself converts it from embedding tier to backbone tier and unlocks Integrated Gradients over the band-by-date grid. Running both the self-hosted encoder and the published embedding also produces a genuinely novel methodological result about whether embeddings-as-a-product lose information relative to the encoder that made them.
2. **Galileo ViT-Base (85M).** Rank two because it is the reference pixel-time-series GFM, MIT-licensed, `H = W = 1` capable, and because the frozen-encoder weakness documented on SwissCrop25 makes it a scientifically interesting rather than a safe choice: the thesis is precisely a frozen-encoder label-budget study, so a model with a known frozen-versus-fine-tuned gap is informative.
3. **Presto (0.4M).** Rank three because it is the scale control. Three orders of magnitude smaller than the patch models, trivially cheap, and a strong CropHarvest performer at 0.835 mean F1. If it holds its own at low K, that is the thesis's most quotable result about what label efficiency actually buys.
4. **AlphaEarth Foundations (64-D annual).** Rank four, unchanged, as the operational-reality arm. Its expected underperformance on a single growing season is a finding with three independent corroborations, and it is the only arm that reflects how a ministry or agronomist would actually obtain features today, at zero compute cost.
5. **TerraMind v1 base, gated.** Rank five, retained solely as the patch-tier and spatial-XAI representative. Scope it to a size-filtered parcel subset (for example parcels above 1 ha, where token contamination is mild), lock a Sentinel-2 scene-ID manifest for reproducibility, and place an explicit go/no-go decision point after the ingest prototype. If the ingest slips, drop this arm and report the patch tier as out of scope with the reason stated; the thesis survives intact because the remaining four arms answer RQ1 and RQ2 in full.

### Stretch set, in priority order

6. **OlmoEarth Base (90M)**, once the licence restrictions are cleared with the supervisors. The strongest 2025 to 2026 entrant that also handles single-pixel time series.
7. **THOR-1.0-base**, only if the TerraMind ingest succeeds and there is spare time. It reuses the same patch store, and its flexible patch size gives a clean internal test of the token-contamination hypothesis at no extra ingest cost.
8. **Prithvi-EO-2.0-300M**, also only if the patch ingest succeeds. It is already in the installed TerraTorch, it is the only patch model with a real temporal axis, and adding it makes the patch tier a three-model comparison rather than a two-model one.

### What this does for each research question

For **RQ1**, the primary set spans 0.4M to 85M parameters among the natively-compatible models, plus one patch model and one embedding product, so the label-budget curves separate the effect of scale, of input representation and of distribution format rather than confounding all three.

For **RQ2.1**, Presto, Galileo and TESSERA all expose an explicit `(T, bands)` input, so Integrated Gradients and Occlusion produce per-band and per-date attributions in the model's own coordinates, directly comparable with the TIMESAT phenometric baseline's feature importances. This is what makes RQ2.3 answerable at all. Under the current four-model set it is not answerable on any model.

For **RQ2.2**, nothing changes: MC Dropout and Deep Ensembles operate at head level over frozen embeddings and are indifferent to the encoder.

For **RQ3**, a richer and more trustworthy explanation block feeds the context document, which is the whole premise of the reporting layer.

### A practical note on tooling

Neither Galileo nor Presto is present in `terratorch 1.2.6` or `torchgeo 0.8.1`; both are loaded from their own repositories, which is straightforward. If a single unified interface is wanted, two options exist. **AiTLAS 2.0** (Apache-2.0, [github.com/biasvariancelabs/aitlas](https://github.com/biasvariancelabs/aitlas)) advertises support for AnySat, CACo, Copernicus-FM, CROMA, DOFA, Galileo, GASSL, Panopticon, Presto, Prithvi, SatMAE, SatMAE++, Scale-MAE, SeCo, TerraFM and TerraMind in one library. **rs-embed** (Ye et al., 2026, [arXiv:2602.23678](https://arxiv.org/abs/2602.23678), [github.com/cybergis/rs-embed](https://github.com/cybergis/rs-embed)) provides on-demand embeddings for roughly 16 models including AlphaEarth, AgriFM, AnySat, Galileo, Prithvi-EO-2.0, SatMAE, Scale-MAE, TESSERA, TerraFM, TerraMind and THOR. Neither has been tested in this environment, and adopting either adds a dependency risk that a direct per-repository integration does not. The recommendation is direct integration, with AiTLAS noted as a fallback.

### Context the thesis should cite when defending the design

Corley et al. (2026, [arXiv:2605.12678](https://arxiv.org/abs/2605.12678), "No One Knows the State of the Art in Geospatial Foundation Models") is the strongest available justification for the thesis's insistence on a fixed protocol, fixed seeds and recorded configurations. Across 152 papers they find 401 distinct benchmarks, 39 % of papers releasing no weights, and 46 cases where the same model, benchmark and protocol are reported with spreads of ten points or more, including Scale-MAE on NWPU-RESISC45 linear probing reported as 33.0 by one paper and 89.6 by another from the same checkpoint. A thesis that fixes one protocol across five models on one dataset is a direct response to that critique and should say so.

---

## 6. Could not verify

- **Exact parameter counts for TerraMind v1 base and large.** The Hugging Face card does not state them. Encoder depth 12 with 12 heads (base) and depth 24 with 16 heads (large) were read directly from `terramind_register.py` in the installed package, which implies ViT-B and ViT-L class sizes of roughly 86M and 300M, but this is inference from architecture rather than a stated figure.
- **THOR parameter counts.** Neither the arXiv abstract, the GitHub README nor the Hugging Face card discloses them.
- **Whether THOR supports multi-temporal input.** Absence of documentation is not proof of absence. The paper's full text should be read before THOR is finally excluded on this ground.
- **The latest TerraTorch release version.** A fetch of the releases page returned version and date fields that are internally inconsistent with the installed `1.2.6` (it reported v1.2.10 dated July 2024). The registry contents reported here come from direct inspection of the installed package and are reliable; the claim that newer TerraTorch versions do or do not add Galileo, Presto, Copernicus-FM, AnySat or THOR is **not** verified.
- **Exact Sentinel-2 band subsets and processing levels for Presto, Galileo and TESSERA.** Each is stated in prose as "10 bands" or similar, but the precise band indices and whether the expected product is L1C or L2A must be read from each repository's normalisation constants before any experiment is run. This is the highest-priority verification task before implementation.
- **Whether Galileo's public code path permits `H = W = 1` without modification.** The paper evaluates "pixel timeseries classification" on CropHarvest, and the `MaskedOutput` structure includes a `[B, T, bands]` temporal component, so the capability clearly exists; the exact call signature was not confirmed because the raw source file fetch returned 404 at the guessed path.
- **The AlphaEarth 2025 annual layer availability in the free Earth Engine catalogue.** Sources variously say coverage runs to 2024 or to 2025. Check the catalogue directly.
- **SkySense V2 weight availability.** No confirmed public checkpoint was located.
- **TESSERA licence discrepancy.** The GitHub repository states CC-0 for weights and embeddings with MIT software; Fang et al. (2026) tabulate TESSERA as "MIT (closed weights)". The repository is taken as authoritative here, but the licence should be re-checked immediately before any redistribution of derived embeddings.
- **`EuroCropsML 2.0`.** Reuss et al. (2026, DirPA, [arXiv:2603.12905](https://arxiv.org/abs/2603.12905)) train and evaluate on an "EuroCropsML 2.0", an extension covering Austria, Belgium, Flanders, Wallonia and Estonia among others, with parcel counts far above the original 706,683. Whether this is publicly released, and whether the thesis should migrate to it, was not established and is an urgent question for the supervisors.
