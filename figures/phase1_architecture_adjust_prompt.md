# Phase 1 architecture figure: adjustment prompt

An edit prompt for the existing architecture diagram in Claude Design, which was built from
[phase1_architecture_prompt.md](phase1_architecture_prompt.md). It adds the step that saves embeddings to disk,
brings the lanes in line with the agreed model set (THOR, AlphaEarth, U-Net baseline), and removes the neck and
head detail, which [pipeline.md](../docs/phase1/pipeline.md) now carries. Companion figure:
[phase1_experimental_setup_prompt.md](phase1_experimental_setup_prompt.md).

## Prompt

```
Edit the existing architecture diagram. Keep its style, colours, legend and top-to-bottom layout. Make these changes:

1. LANES. Go from three to six lanes, in this order, under three small group headers:
   - "Frozen GFM, token grid": TerraMind | THOR | Prithvi-EO-2.0
   - "Baseline": U-Net
   - "Frozen GFM, per-pixel embeddings": TESSERA | AlphaEarth
   THOR is a copy of the TerraMind lane with the name changed and its caption reduced to "Same handling as TerraMind".
   AlphaEarth is a copy of the TESSERA lane with the source "Google Earth Engine: precomputed annual embeddings, 2021".

2. DATA. The shared Data source / Preprocessing / Model input boxes now span the four left lanes (TerraMind, THOR, Prithvi, U-Net) and read "Sentinel-2 + Sentinel-1, 2021, 12 monthly composites". Across TESSERA and AlphaEarth, one shared box: "Raw extraction of precomputed embeddings". Remove the "TerraTorch EncoderDecoderFactory" label.

3. NEW BAND "Embeddings on disk", directly below Backbone, spanning all six lanes. Draw it as a wide bar with a double-line border, so it reads as storage. Contents:
   - across TerraMind, THOR, Prithvi: "Encode once: features saved per chip (training chips also in 8 flipped and rotated copies)"
   - across TESSERA, AlphaEarth: "Download once: embeddings saved per chip"
   - U-Net: a thin dashed line passing through the band, labelled "nothing saved, trains on the composites"
   The U-Net lane has dashed pass-throughs in Backbone too.

4. SIMPLIFY. Delete the Neck and Head bands. The Decoder band (amber) reads:
   - TerraMind, THOR, Prithvi: "Trained decoder, identical for the three models"
   - U-Net: "U-Net, all layers trained from scratch"
   - TESSERA, AlphaEarth: "Trained per-pixel decoder"
   Each Decoder box ends with "→ crop-type map, 224 × 224 px".

5. BRACKETS on the right instead of the frozen/trainable brace:
   - over Backbone + Embeddings on disk: "Runs once per model (frozen)"
   - over Decoder: "Repeated for every label budget, draw and seed (trained)"

6. OUTPUT. One green box across all six lanes: "Crop-type segmentation, evaluated identically for every lane".

7. LEGEND. Add "double-line bar = saved to disk".
```
