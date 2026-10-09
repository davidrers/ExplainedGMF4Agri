# Phase 1 architecture figure: prompt for a diagramming tool

The Mermaid version of this figure is [phase1_architecture.mmd](phase1_architecture.mmd). It is
adequate for the docs, but Mermaid cannot draw the stage-label column or the frozen/trainable
brace, so the thesis figure is to be produced from the prompt below.

Before rendering, note that **Sentinel-1 extraction** is shown for TerraMind but is not yet
implemented; the chip export in `src/gfm4agri/chips/s2_monthly.py` is Sentinel-2 only.

## Prompt

> Create a vertical architecture diagram for an MSc thesis (academic, clean, print-ready, white background, sans-serif, no icons or 3D). Three parallel vertical lanes titled **TerraMind**, **Prithvi-EO-2.0**, **TESSERA**, left to right. On the far left, a narrow column of stage labels defining horizontal bands that span all three lanes, top to bottom: *Data source*, *Preprocessing*, *Model input*, *Backbone*, *Neck*, *Decoder*, *Head*, *Output*. Every lane has exactly one box per band so the rows align perfectly; where a lane does nothing in a band, draw a thin dashed pass-through box.
>
> **Data source.** TerraMind and Prithvi share one box spanning both lanes: "Microsoft Planetary Computer STAC, 2021: Sentinel-2 L2A (both models) and Sentinel-1 RTC (TerraMind only)". TESSERA: "geotessera library: precomputed TESSERA embeddings (from S1 + S2 time series), 2021".
>
> **Preprocessing.** Shared TerraMind/Prithvi box: "pystac + stackstac; warp to EPSG:3035 at 10 m; S2 cloud and snow masking with the SCL layer; S1 VV and VH (TerraMind only); per-pixel monthly median; temporal gap filling; fixed monthly grid T = 12". TESSERA: "crop and align to the same chip grid, EPSG:3035, 10 m".
>
> **Model input.** Shared TerraMind/Prithvi box: "Chip 224 × 224 px (2.24 km), 12 months × bands, with crop-label mask and parcel-ID raster". TESSERA: "Chip 224 × 224 px, 128-dimensional embedding per pixel, same masks".
>
> A label on the boundary between Model input and Backbone: "TerraTorch EncoderDecoderFactory".
>
> **Backbone (frozen, hatched grey).** TerraMind: input "12 S2 bands + S1 VV, VH"; a TemporalWrapper container holding a stack of 12 small encoder boxes labelled "month 1 … month 12", each emitting its own embedding, and the 12 embeddings merging into one concatenation block. Numbered caption: "TerraMind is single-date, so the TemporalWrapper (1) folds the 12 months into the batch, (2) runs the frozen ViT once per month, with no attention between months, and (3) concatenates the 12 monthly embeddings of each token into 12 × D channels on a 14 × 14 token grid (~160 m). The months are stacked, not averaged; how they relate is learned later by the trainable neck". Prithvi-EO-2.0: input "6 S2 bands (Blue, Green, Red, B8A, SWIR 1, SWIR 2) + date and location"; one single wide encoder box with no wrapper, with all 12 months entering it together. Numbered caption: "Prithvi is natively spatio-temporal, so no TemporalWrapper is needed: (1) all 12 months enter together through a 3D patch embedding, (2) one frozen ViT pass over 12 × 196 tokens with joint space-time attention, (3) the embeddings leave the encoder already fused across months and are reshaped to 12 × D channels on the same 14 × 14 grid". Draw a small note between the two lanes: "TerraMind: months combined after the encoder (concatenation). Prithvi: months combined inside the encoder (attention)". TESSERA: "IdentityBackbone: the input is already the embedding".
>
> **Neck (trainable, amber).** TerraMind and Prithvi each, four small stacked steps: "SelectIndices (4 layers) → ReshapeTokensToImage → ChannelBottleneck to 768 channels → LearnedInterpolateToPyramidal (56, 28, 14, 7)". TESSERA: dashed pass-through "no neck".
>
> **Decoder (trainable, amber).** TerraMind and Prithvi: "UNetDecoder, output 64 × 56 × 56". TESSERA: "Pixel-wise MLP 128 → 512 → 256".
>
> **Head (trainable, amber).** All three: "SegmentationHead, 1 × 1 convolution to class logits". TerraMind and Prithvi add "bilinear upsampling ×4 to 224 × 224"; TESSERA adds "native 224 × 224, 10 m".
>
> **Output.** One wide box spanning all three lanes, green: "Crop-type segmentation, K-shot label budget: K % of parcel polygons per class labelled for training (at least one per class), all other pixels ignore_index; cross-entropy on labelled pixels only; several random draws per K; evaluated on fixed held-out spatial blocks with dense labels; Macro-F1 with 95 % bootstrap CI".
>
> To the right of the three lanes, a vertical brace from Backbone to Head labelled "frozen" over the Backbone band and "trainable" over the Neck to Head bands. Legend at the bottom: blue = data, grey = frozen, amber = trainable, green = evaluation. Arrows are straight and vertical only.
