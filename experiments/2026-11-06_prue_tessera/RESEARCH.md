# Field boundary segmentation: the frontier methods, and why PRUE, 6 November 2026

Asked by the user on 6 November 2026: "make a deeper research on frontier methods for only border segmentation, pick
the most promising one and launch an experiment fine tuning it with tessera". This note surveys the methods that
delineate agricultural fields from satellite images, weighs them against Sentinel-2 at 10 m and against what the
delineation experiments of this repository have shown, and gives the reasons for the choice. The numbers are the
authors' own unless stated.

## What the earlier experiments showed

| Experiment | Method | What limited it |
|---|---|---|
| [`2026-10-18_parcel_delineation`](../2026-10-18_parcel_delineation/REPORT.md) | one boundary logit beside each arm's crop-type head, watershed | inside the oracle extent, no GFM arm beat the extent's own floor (class-agnostic PQ 0.486); TESSERA's head is a 1 x 1 layer on a per-pixel MLP, with no spatial context at all |
| [`2026-11-05_parcel_extent`](../2026-11-05_parcel_extent/README.md) | the same heads without the oracle extent | no background class: over the whole chip the boundary parcels of frozen TESSERA fall to PQ 0.048, with about 352,000 predicted parcels for 55,575 true ones |
| [`2026-11-05_paps_tessera`](../2026-11-05_paps_tessera/REPORT.md) | U-TAE's spatial U-Net + PaPs on TESSERA | class-aware PQ 0.303 frozen and 0.357 fine-tuned over the whole chip; finds 40 % of the parcels and 11 % of those under 0.5 ha |

So two things were missing for TESSERA: a spatial decoder that sees neighbouring pixels, and a learnt background,
so that fields are not built on forest and villages. PaPs supplied both and moved PQ from 0.05 to 0.30.

## The families of methods

**1. Edge detection, then closing.** A network predicts a boundary probability (HED, RCF, and their crisp-edge
successors such as DiffusionEdge, AAAI 2024), and a grouping step closes contours: oriented watershed and
ultrametric contour maps, multiscale combinatorial grouping. ITC's own work belongs here: Persello et al. (2019,
RSE) delineated smallholder fields with an FCN and combinatorial grouping on very high resolution images, and
Masoud, Persello and Tolpekin (2020, Remote Sensing) built a multiple-dilation FCN contour detector for Sentinel-2.
At 10 m a field edge is one or two pixels, so contours break and the grouping step carries the result. Our
boundary head was this family in its simplest form.

**2. Multi-task extent, boundary and distance.** The network predicts three maps, the field extent, the boundary and
the distance to the boundary, and a hierarchical watershed cuts the extent where the boundary is strong.
ResUNet-a (Waldner and Diakogiannis, 2020, RSE) and its FracTAL ResUNet successor "DECODE" (Waldner et al., 2021,
"Detect, consolidate, delineate", Australia) are the references, with the fractal Tanimoto loss with complement;
BsiNet (Long et al.), SEANet (Li et al.) and HBGNet (Zhao et al.) are multi-task and dual-branch variants for very
high resolution images; a 2025 study runs ResUNet-a d7 on Sentinel-2 Level-3A for early-season maps with a Gaussian
mixture step to split instances. The review of Zheng et al. (arXiv 2508.14558, 2025) calls the
multi-task semantic family the most accurate of the deep learning approaches and finds that object-detection
methods generally score a little lower. On the Fields of The World benchmark, however, DECODE reaches a pixel IoU of
0.71 but an object F1 of only 0.21 (PRUE, Table 1), its parcels built by connected components rather than by its
own tuned watershed.

**3. Satellite image time series transformers.** PTAViT3D and PTAViT3D-CA (Diakogiannis et al., RSE 2026,
arXiv 2409.13568, code github.com/feevos/tfcl) run a 3D vision transformer over raw Sentinel-2 or Sentinel-1 series,
the cross-attention variant fusing the two so that cloudy optical dates are tolerated, and predict extent, boundary
and distance with the FracTAL Tanimoto loss. They are evaluated on ePaddocks, Fields of The World, PASTIS and
AI4SmallFarms and claim the state of the art and continental scale over Australia. The released code holds the
model and loss but neither the target construction nor the instance post-processing. The model is itself a
temporal encoder: with it, TESSERA would be replaced, not tested.

**4. Instance segmentation.** Each field is an object: Mask R-CNN; Mask2Former; DeepSnake-style contour regression
(E2EVAP, Pan et al.); Delineate Anything (Lavreniuk et al., 2025, arXiv 2504.02534), a YOLOv11 instance model trained
on FBIS-22M (672,909 patches from 0.25 to 10 m, 22.9 M fields), with its country-scale pipeline Delineate Anything
Flow (arXiv 2511.13417) and a v2 (arXiv 2607.19069, 2026); and the Segment Anything family, fine-tuned with LoRA on
Sentinel-2 (arXiv 2506.16318, ICIAP 2025) or prompted by a DeepLabv3+ (fabSAM, 2025). These models are trained on
RGB at high resolution; on Sentinel-2's 10 m fields of FTW, Delineate Anything zero-shot reaches an object F1 of 0.09,
SAM ViT-H zero-shot 0.17 and fine-tuned 0.37, Mask2Former 0.39 (PRUE, Table 1). PaPs, which we ran, is the
panoptic member of this family.

**5. Three-class semantic segmentation: background, field interior, field boundary.** One softmax per pixel; the
fields are the connected regions of the interior class, the boundary class keeping adjacent fields apart and the
background class keeping fields off non-agricultural land. This is the formulation of Fields of The World (Kerner et
al., AAAI 2025; 70,462 samples, 24 countries on four continents, Estonia among them, bi-temporal Sentinel-2), whose
baseline is a U-Net with an EfficientNet-B3 encoder (object F1 0.38).

## PRUE, the frontier of family 5

PRUE, "A Practical Recipe for Field Boundary Segmentation at Scale" (CVPR 2026, arXiv 2603.27101, code
github.com/fieldsoftheworld/ftw-prue), is the first systematic comparison of segmentation models and geospatial
foundation models for field boundaries under one protocol: 18 models on the FTW test sets, presence-only countries
excluded, object precision and recall at a 0.5 confidence threshold, averaged over the countries.

| Model on FTW (PRUE, Table 1) | Pixel IoU | Object F1 |
|---|---|---|
| **PRUE: U-Net + EfficientNet-B7, log-cosh Dice, boundary weight 0.75** | **0.76** | **0.47** |
| FTW baseline: U-Net + EfficientNet-B3 | 0.70 | 0.38 |
| Mask2Former (Swin-S, 8 channels) | 0.68 | 0.39 |
| SAM ViT-H fine-tuned / zero-shot | 0.45 / 0.32 | 0.37 / 0.17 |
| DECODE (FracTAL ResUNet, multi-task) | 0.71 | 0.21 |
| Delineate Anything, zero-shot | 0.37 | 0.09 |
| Clay ViT-L (frozen encoder, conv decoder); fully fine-tuned | 0.67; 0.73 | 0.36; 0.42 |
| Galileo ViT-B, DINOv3 ViT-L (frozen) | 0.66, 0.60 | 0.32, 0.31 |
| TerraMind ViT-B, Prithvi 2.0 ViT-L (frozen) | 0.57, 0.56 | 0.26, 0.25 |

The recipe and its ablations (Table 5, object F1): the boundary class weight omega in `[0.05, 0.95 - omega, omega]`
matters most, from 0.32 at 0.60 to 0.42 at 0.75 and 0.80; log-cosh Dice beats cross-entropy (0.24), Dice (0.42),
Tversky (0.43), focal (0.18) and the fractal Tanimoto loss (0.38) at 0.44; the encoder adds 0.03 from B3 to B7; the
learning rate is best at 3e-3 and 1e-2 (0.39 each in the sweep); brightness, resize and channel-shuffle
augmentations add no object F1 but make the model robust to those shifts. Together they lift the FTW baseline from
0.38 to 0.47 object F1 and from 0.70 to 0.76 pixel IoU. The released configuration (`configs/release/3_class/vit.yaml`)
is a fully fine-tuned Clay ViT with weighted cross-entropy (`[0.04, 0.08, 0.88]`), not the paper's PRUE; this
experiment follows the paper.

## Why PRUE is the one to try

1. **The best measured result at our resolution.** On the largest field-boundary benchmark of Sentinel-2 at 10 m,
   with Estonia in it, PRUE beats the instance models, the multi-task model, the SAM family and eleven foundation
   models under one protocol. No other survey entry has an evaluation that broad on 10 m data.
2. **It supplies exactly what our boundary head lacked.** A background class, so no extent is needed and fields
   stay off forest and villages, and a strong spatial decoder over the 10 m embedding, so the boundary is seen
   from its neighbourhood. It needs nothing the PaPs experiment did not already have: the same WorldCover background,
   the same targets from the parcel rasters.
3. **Its output is the scoring's input.** Connected regions of one class are parcels, scored by the same void-rule
   PQ as PaPs and the boundary head, with FTW's object F1 as the class-agnostic RQ at IoU > 0.5.
4. **It leaves a question that is ours to answer.** PRUE finds frozen GFMs well below a U-Net trained on raw bands
   and full fine-tuning of Clay recovering part of the gap (0.36 to 0.42). TESSERA, a per-pixel temporal encoder,
   was not among its eighteen. Feeding TESSERA's embedding to PRUE's U-Net and training the two end to end asks
   whether a pixel time-series GFM closes that gap.
5. **It is simple and cheap.** One softmax, one loss, connected components; 67 M parameters, 300 km2/s at inference
   in the paper; the code is public.

**Not chosen.** PTAViT3D: a temporal encoder of its own on raw series, so TESSERA would leave the experiment, and its
post-processing is unreleased. Delineate Anything and SAM: RGB models pretrained at high resolution, which cannot take
a 128-dimension embedding without losing the pretraining they are chosen for, and which score lowest on 10 m FTW.
ResUNet-a and DECODE: below PRUE on FTW, their advantage resting on a hierarchical watershed tuned per region.
Edge detectors with grouping: our own boundary head showed how they fail at 10 m.

## How PRUE is adapted here

| | PRUE (paper) | This experiment |
|---|---|---|
| Input | two Sentinel-2 dates, RGB + NIR, 8 channels | TESSERA v2 Medium's 128 dimensions from twelve monthly S2 + S1 composites |
| Encoder of the input | none (raw bands) | TESSERA frozen (reference) or trained end to end at 1e-5 (the arm asked for) |
| Network | U-Net, EfficientNet-B7 (67.1 M) | the same, smp, ImageNet weights, first convolution widened to 128 channels |
| Classes | background, interior, boundary (FTW's masks) | the same, built from the parcel rasters: boundary = a parcel pixel whose 4-neighbour is another parcel or none (two pixels between neighbours); background = WorldCover non-agricultural land outside the drawn parcels; WorldCover grassland and cropland outside them ignored |
| Loss | log-cosh Dice, weights `[0.05, 0.20, 0.75]` | the same (ftw-prue's `logCoshDice`) |
| Optimiser | AdamW, rate swept (3e-3 best tied) | AdamW with amsgrad (ftw-prue), 3e-3, cosine to 1e-6, weight decay 0.01 |
| Augmentation | brightness, resize, channel shuffle | D4 rotations and flips only (FTW's datamodule); brightness and channel order have no meaning for an embedding |
| Training windows | FTW chips | random 128 x 128 windows of the 224 x 224 chips, so that the trainable encoder's cost per epoch stays affordable; validation and test on whole chips |
| Epochs, checkpoint | not stated in the paper; ftw-prue 30, best validation object F1 | 30, best validation class-agnostic RQ (object F1) of the `cc` fields |
| Fields | polygons of the interior class, under 500 m2 dropped | `cc`: the same on the raster; `grow`: `cc` grown into the boundary pixels by watershed; `field`: the predicted field unsplit, the floor |
| Score | object F1, pixel IoU, per FTW country | void-rule PQ and RQ, class-agnostic (and class-aware with the crop map's vote), whole chip, WorldCover and oracle cuts; pixel IoU and boundary F1 |

A perfect three-class map built from the test truth itself, passed through `grow`, keeps class-agnostic RQ 0.74 to
0.79 (`check.py`): the true pieces with no interior of five pixels, median two pixels, are slivers at chip edges and
fragments of multipart parcels that no method of this repository has found.

## Sources

- PRUE: [arXiv 2603.27101](https://arxiv.org/abs/2603.27101), [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/36402), [code](https://github.com/fieldsoftheworld/ftw-prue)
- Fields of The World: [arXiv 2409.16252](https://arxiv.org/pdf/2409.16252), [AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35034)
- PTAViT3D, "Tackling fluffy clouds": [arXiv 2409.13568](https://arxiv.org/abs/2409.13568), [code](https://github.com/feevos/tfcl)
- Review of parcel and boundary delineation: [arXiv 2508.14558](https://arxiv.org/pdf/2508.14558)
- Delineate Anything: [arXiv 2504.02534](https://arxiv.org/abs/2504.02534v1); Flow: [arXiv 2511.13417](https://arxiv.org/pdf/2511.13417); v2: [arXiv 2607.19069](https://arxiv.org/pdf/2607.19069)
- SAM for field delineation: [arXiv 2506.16318](https://arxiv.org/pdf/2506.16318); [Sentinel-2 seasonal SAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12857264/)
- Early-season multi-task ResUNet-a d7: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666017225000628)
- Crisp edge detection: [DiffusionEdge, arXiv 2401.02032](https://arxiv.org/pdf/2401.02032)
