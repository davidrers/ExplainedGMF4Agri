# Pixel-level vs patch-level GFMs on EuroCropsML — implications

Working note. Scope: how the two GFM families interact with EuroCropsML's per-parcel crop classification task across data ingestion, comparability, XAI, mapping, and compute. Used to inform the Methods chapter writing; **not for inclusion in the proposal as-is**.

## The two families

| | **Pixel-level GFMs** | **Patch-level (scene-token ViT) GFMs** |
|---|---|---|
| Examples | AlphaEarth Foundations, TESSERA | TerraMind, THOR, Prithvi |
| Encoder input | One pixel's time series (TESSERA) or annual signal (AlphaEarth) | 2D multispectral patch, e.g. (T, C, 224, 224) |
| Native resolution | 10 m (S2 pixel) | 10 m input, ~160 m per token (16-px patch tokenization) |
| Output per location | One D-dim vector per pixel | Grid of (H/P × W/P) tokens per patch |
| Distribution format | Pre-computed embeddings (GEE collection, Python lib) | Model weights (HuggingFace, TerraTorch) |
| Wall-to-wall mapping | Native — every pixel already has a vector | Needs a decoder (UNet / UperNet) to upsample tokens to pixel resolution |

## Implications across the Phase-1 pipeline

### 1. Data ingestion

- **Pixel-level.** Drop-in. Sample the precomputed embedding at the parcel centroid, or zonal mean over the parcel's pixels. Storage cost per parcel = D bytes (e.g., 64 for AlphaEarth, ~128 for TESSERA). Total per-parcel store for the full 706k EuroCropsML parcels ≈ 90–180 MB. Trivial.
- **Patch-level.** EuroCropsML does **not** ship 2D scene patches. You must fetch raw S2 L1C tiles from the Copernicus archive for 2021, crop a window (e.g., 224 × 224 px) around each parcel centroid, and store the patch stack `(T, C, 224, 224)` per parcel. Storage ≈ tens of GB to ~1 TB depending on T, C, and dtype. Encoder inference cost is GPU-bound, per parcel.

### 2. Effective resolution and parcel purity

- Pixel-level: native 10 m. A 0.5 ha parcel (~70 × 70 m) still has ~50 pixels to sample. Clean.
- Patch-level: each token covers a ~160 × 160 m footprint. A 0.5 ha parcel sits inside a single token whose footprint also includes neighbouring fields. **The token is contaminated by surrounding crops.** The contamination is mild for Estonian / Latvian commercial parcels (often ≥ 5 ha) and severe for Portuguese smallholder parcels (commonly < 1 ha).

### 3. Pooling choices for patch-level GFMs

Three options to collapse a token grid into one vector per parcel:

- **CLS token** (if exposed) — cheap, but trained for whole-patch semantics, ignores parcel polygon.
- **Mean over all tokens** — standard ViT classification baseline; same whole-patch averaging.
- **Masked mean over parcel-intersecting tokens** — methodologically clean analog of EuroCropsML's per-parcel spatial median. Requires rasterising the polygon onto the token grid. At 16-px granularity, small parcels may have no purely-internal token; in that case, fall back to "tokens whose footprint *overlaps* the polygon" weighted by overlap area.

Pixel-level GFMs avoid this entirely — they pool by zonal mean over native pixels.

### 4. Spatial context vs spatial purity trade-off

- Patch-level GFMs see neighbouring fields, hedgerows, roads, soils. For *crop type* classification this is noise; for *land-use* or *landscape-pattern* tasks it can be informative.
- Pixel-level GFMs see only the location's spectral-temporal trajectory. Pure but context-free.

This is one of the genuine architectural axes the thesis is benchmarking, not a defect.

### 5. Comparability of the FE+TH benchmark

The current Phase-1 framing claims uniform FE+TH adaptation across all four GFMs. The head architectures and protocol are uniform, but the **embedding-extraction step is not symmetric across families**. To bring it closer to symmetric:

- For patch-level GFMs, default to **masked-mean pooling over parcel-intersecting tokens** (analog of EuroCropsML's per-parcel spatial median).
- For pixel-level GFMs, use **zonal mean over parcel pixels** (also a spatial median / mean).

After both, the resulting vector per parcel is the output of a comparable spatial aggregation. What remains asymmetric is the encoder's effective resolution. Document this as a known caveat rather than try to eliminate it.

Optional further mitigation: report a sensitivity stratum on parcels above some minimum size threshold (e.g., > 1 ha) where patch-level contamination is mild. If model rankings are stable across the full set and the size-filtered set, the resolution asymmetry is not driving the headline result.

### 6. K-shot dynamics

- Pixel-level: K parcels → K vectors. Deterministic.
- Patch-level: K parcels → K patch stacks → K pooled vectors. The pooling choice (CLS vs mean vs masked-mean) is a hyperparameter that affects each K-shot run. Either fix the pooling choice up front or report it as a sub-grid in the learning curves.

### 7. Cross-region transfer (Phase 1c)

- Pixel-level: AlphaEarth annual embeddings are global and pre-computed; sampling in PT vs ET costs the same. TESSERA: pre-computed for given tiles; new regions may require fresh sampling via the library.
- Patch-level: target region requires fetching new S2 tiles from Copernicus and re-running the encoder. Same data-engineering cost as in-region setup.

### 8. XAI (Phase 2) implications

| | Spatial XAI | Temporal XAI |
|---|---|---|
| AlphaEarth | None natively (annual, single vector) | None natively (annual aggregation) |
| TESSERA | None (one pixel's time series per location) | Rich — attention over time tokens, IG over (T × B) |
| TerraMind / THOR | Rich — attention rollout / Chefer over patch tokens, maps to spatial saliency | Rich — temporal axis preserved in multi-temporal models |

Implication: a **fair cross-family XAI comparison must restrict to the axes both families expose**. The natural common axis is *spectral and temporal feature importance* (SHAP / IG over band-and-date features after projecting attributions back to inputs). Spatial attention maps are a patch-level-only contribution.

### 9. Wall-to-wall mapping (Path B from prior conversation)

- Pixel-level: head trained per parcel applied per pixel → map directly (with mild distribution-shift post-processing — majority filter / CRF).
- Patch-level: per-parcel head does **not** transfer to per-pixel inference. Wall-to-wall mapping requires a trained decoder (UNet) and pixel-level labels — i.e., the segmentation track. Separate experiment, separate dataset construction.

### 10. Compute and storage summary

| Stage | Pixel-level | Patch-level |
|---|---|---|
| Embedding extraction | One-time CPU lookup / API call per parcel | One-time GPU forward pass per parcel patch |
| Per-parcel storage | ~D bytes | ~T × C × H × W bytes for the patch + D for the pooled vector |
| Downstream head training | Same | Same |
| Wall-to-wall inference | Same as per-pixel sampling | Requires GPU forward pass + decoder upsampling |

### 11. Reproducibility

- Pixel-level: AlphaEarth reproducible on any GEE account with the dataset ID. TESSERA reproducible via the published Python lib at a pinned version.
- Patch-level: depends on the specific S2 scene IDs fetched from Copernicus. Lock the scene-ID manifest, the encoder weights' HuggingFace revision hash, and the pooling implementation. Without the manifest the patches are non-reproducible.

## What this means for the methods chapter

The Section-5 text already names the asymmetry ("two ingest routes converge on a shared adaptation and evaluation stage"). The richer story sits in the discussion / results chapters:

1. The four GFMs do not all play the same game. They are compared on the same task with the same head battery and protocol, and that comparison is meaningful — but it is a **comparison of artifact-as-released**, not a controlled architecture-vs-architecture experiment.
2. Patch-level token contamination on small parcels is a known, expected source of patch-level disadvantage on Portuguese smallholders.
3. The XAI comparison (Phase 2) must keep to axes both families expose. Spatial attention is a patch-level-only feature, not a cross-family axis.
4. Wall-to-wall mapping (a likely downstream use for an agronomist or ministry) is *operationally* easier with pixel-level GFMs; patch-level GFMs need a separate segmentation training step.

## Open methodological choices

- **Pooling for patch-level GFMs:** fix to masked-mean upfront, or report as a sub-grid?
- **Size-stratified reporting:** report aggregate results, or also a "parcels > 1 ha" sensitivity stratum?
- **Patch window size:** 224 px (~2.2 km) is standard; 128 px is cheaper and reduces contamination at the cost of less spatial context.
- **TerraMind patch resolution:** confirm the actual patch tokenization size (16 px assumed here; verify against the TerraMind paper).
