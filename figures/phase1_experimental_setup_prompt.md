# Phase 1 experimental set-up figure: prompt for a diagramming tool

Companion to [phase1_architecture_prompt.md](phase1_architecture_prompt.md): that figure shows what each model is,
this one shows how the models are compared. It names no research question and no country.

Not yet carried into [protocol.md](../docs/phase1/protocol.md) or the code: the GAEZ zone screen, THOR and AlphaEarth,
nested repeated draws, equal training effort, and no validation labels at low K.

## Prompt

> Create a vertical flow diagram for an MSc thesis (academic, clean, white background, sans-serif, no icons). Two columns. Left, about two thirds of the width: eight numbered step boxes, top to bottom, joined by straight vertical arrows. Right: one tall panel titled **Fairness rules**, the full height of the steps, listing F1 to F8. No arrows between the columns; each step carries small rule tags in its top right corner (e.g. "F1 F2"). Keep every box to one or two short lines.
>
> **Steps**
>
> 1. **Zone screening** (blue): "Countries mapped to GAEZ v5 agro-ecological zones. Transfer only within the same climate zone."
> 2. **Data extraction** (blue), two side-by-side sub-boxes over one shared strip. Left: "TerraMind · THOR · Prithvi: Sentinel-2 + Sentinel-1, 12 monthly composites". Right: "TESSERA · AlphaEarth: precomputed embeddings, raw extraction". Shared strip: "Same chips and parcel label masks". Tags: F1.
> 3. **Spatial split** (blue): "Train, validation and test by whole blocks, with a buffer. Test blocks fixed and fully labelled." Tags: F3, F5.
> 4. **Label budget** (blue): "K = 1, 5, 10, 20, 50, 100 % of parcels per class. Nested, repeated draws." Tags: F2, F7.
> 5. **Models and training**, two side-by-side sub-boxes joined by an arrow. Left, grey: "Foundation model, frozen". Right, amber: "Decoder, trained". Below both: "Same training effort: equal tuning trials, same number of training steps at every K". Tags: F4, F6.
> 6. **Experiments** (amber), two side-by-side sub-boxes. **E1**: "Label-budget curves within a country". **E2**: "Transfer between countries of the same zone: zero-shot, target only, source + target". Tags: F2, F3.
> 7. **Evaluation** (green): "Fixed test blocks only. Macro-F1 (primary), mIoU, per-class F1, also per AEZ class. Paired comparisons with 95 % intervals." Tags: F3, F7, F8.
> 8. **Reporting** (green): "Every result logged with its settings; all draws released." Tags: F8.
>
> **Fairness rules**, each a number in a small circle, a bold name and one short line:
>
> - **F1 Same input.** Identical chips and masks for every model.
> - **F2 Same labels.** Identical parcels at each K and draw.
> - **F3 Same test.** One fixed, spatially separated test set.
> - **F4 Same effort.** Equal tuning trials and the same number of training steps for every model and K.
> - **F5 No leakage.** Buffer around held-out blocks; no validation labels at low K.
> - **F6 Like with like.** Headline comparisons within the same resolution group.
> - **F7 Repeated and paired.** Several draws per K; models compared on the same draws.
> - **F8 Fixed in advance.** Metrics and criteria declared before results are seen.
>
> Legend at the bottom: blue = data, grey = frozen, amber = trained, green = evaluation, outlined panel = fairness rules. No title inside the figure.
