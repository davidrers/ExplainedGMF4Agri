# The boundary head on TESSERA: the architecture, step by step

**In one sentence.** The boundary head adds one output to the crop-type model: for every pixel, the probability that
it lies on a parcel's edge. Parcels are then cut out of the chip along those edges, and each takes the crop most of its
pixels were given.

![The whole architecture of the boundary head on TESSERA, drawn from one real test chip](figures/architecture.png)

**Every image in the diagram is real.** They all come from one test chip, `EE_02265_01779`, which holds 34 declared
parcels (counted as the pieces inside the chip) and was never seen in training. It is the same chip as in
[PaPs' architecture](../2026-11-05_paps_tessera/ARCHITECTURE.md), so the two diagrams can be compared panel by panel.
The heads' outputs are the stored test predictions of the frozen-TESSERA model, fitted in the delineation experiment
(`../2026-10-18_parcel_delineation`). The post-processing is run exactly as `score.py` runs it.

**The chain in one line:**

    Sentinel-2 + Sentinel-1, 12 months  →  TESSERA encoder (frozen)  →  128 numbers per pixel
      →  per-pixel MLP (trained)  →  two heads (crop, edge)  →  cut the area along the edges  →  parcels with a crop

The diagram has three parts, shown larger below.

---

## Part 1. From satellite images to features

![Part 1: input, TESSERA, embedding, MLP](figures/architecture_1.png)

1. **Input.** One chip of 224 x 224 pixels of 10 m (2.24 km a side). Each pixel has 12 monthly Sentinel-2 images
   (10 bands) and 12 monthly Sentinel-1 radar images (VV and VH).
2. **TESSERA encoder (frozen).** The same pre-trained model as for PaPs (21 M parameters). It reads the 12-month
   series of **one pixel at a time** and turns it into **128 numbers**.
3. **The embedding.** The same image as in PaPs' diagram: the 3 main directions of variation of the 128 numbers,
   shown as colour.
4. **Per-pixel MLP (trained).** This is the crop-type decoder of the main workflow's TESSERA arm: 128 → 512 → 256
   numbers, 0.2 M parameters. Like TESSERA, it looks at **one pixel at a time** and never sees the neighbours. So
   nothing in this model looks at a neighbourhood.

---

## Part 2. The two heads: what each learns, and what it outputs

![Part 2: the two heads, their training targets and their outputs on the test chip](figures/architecture_2.png)

Each column is one head. The **upper** image is what the head is **trained to match** (drawn for this chip, for
comparison), and the **lower** image is what it **outputs** on this test chip.

| Head | Question it answers | Reads | Trained to match | Loss |
|---|---|---|---|---|
| 1 Crop type | Which crop is this pixel? | one pixel's MLP features | the declared crop of every pixel inside a declared parcel | cross-entropy |
| 2 Boundary | Is this pixel on a parcel's edge? | the same features | 1 on the outer ring of every declared parcel (a pixel next to another parcel or to no parcel), 0 inside | binary cross-entropy |

**The third column is the key to this model: what training sees.** Both losses are averaged over the pixels inside the
declared parcels of the 20 crops (white), and nowhere else. Forest, water, villages and undeclared grassland (grey)
never enter training. The heads still give an answer there, because they answer for every pixel, but nothing ever
taught them what "not a field" looks like.

**What to notice in the outputs:**

* **Head 1** paints a crop on every pixel of the chip, forest included (here mostly as grassland, orange). Inside the
  declared parcels it is right on 85 % of the pixels.
* **Head 2** draws the field edges well where they are sharp: inside the declared parcels its pixel F1 at 0.5 is
  0.68. Over the forest its probability is patchy, low in most places.

**Training.** 45 epochs on the 4,898 training chips, batches of 4 chips, AdamW at a learning rate of 0.001, random
flips and rotations. The loss is the crop-type loss plus the boundary loss (weight 1), and the epoch kept is the one
with the lowest validation loss.

---

## Part 3. From head outputs to parcels

![Part 3: post-processing in each area, and the final outputs](figures/architecture_3.png)

No head outputs a parcel. The edge probability is the only geometry the model gives. Parcels exist only after these
steps (`panoptic_x.from_boundary`), which run inside an **area**:

**Row a. No area given: the whole chip.** This is the realistic case, since at prediction time nobody hands us the
declared parcels.

1. **Find the seeds.** Pixels with an edge probability below 0.5 count as "inside a parcel". Each connected patch of
   them is a seed. This chip has 216 seeds.
2. **Grow them (watershed).** Each seed grows outwards over the edge probability, lowest values first, like water
   filling neighbouring basins, until **every pixel of the area belongs to one seed**. Pixels no seed can reach become
   parcels of their own. Here: 216 parcels.
3. **Give each a crop.** Each parcel takes the crop that most of its pixels got from head 1. **This is the final
   output.**

On this chip the whole-chip output has 216 parcels against 34 declared ones. 12 match a declared parcel (overlap IoU
above 0.5), 10 of them with the right crop; chip PQ 0.06. The forest, the water and the villages have become large
false parcels. Step 2 has to put every pixel somewhere, and wherever the edge probability is low a seed forms and
grows.

**Row b. An area given from outside: the same three steps, run only inside it.**

* **ESA WorldCover cropland + grassland** (a public map, no test label used). 212 parcels, 12 found, 10 with the
  right crop, chip PQ 0.06. WorldCover removes the forest, but it keeps all the open grassland, declared or not, and
  it does not separate neighbouring fields.
* **The declared parcels** (an *upper bound only*: it uses the test labels, so it is not a real prediction). 37
  parcels, 22 found, 19 with the right crop, chip PQ 0.21. Most of this "success" comes from the area itself: the gaps
  between declared parcels already separate them.

**Over all 1,475 test chips** (55,575 declared parcels):

| Area the parcels are built in | Frozen TESSERA: PQ (crop ignored) | Fine-tuned TESSERA: PQ (crop ignored) |
|---|---|---|
| Whole chip, no area given | 0.145 (0.048) | 0.197 (0.036) |
| WorldCover cropland + grassland | 0.158 (0.063) | 0.204 (0.083) |
| Declared parcels (upper bound, uses the test labels) | 0.307 (0.476) | 0.355 (0.514) |

The class-aware PQ stays above the crop-ignored one because it averages over the crops: most of the false parcels
are called grassland, so they weigh on one crop out of twenty. For comparison, PaPs reaches 0.303 (0.336) frozen and
0.357 (0.366) fine-tuned over the whole chip, with no area given.

---

## The fine-tuned arm

The diagram is the same, but TESSERA is no longer frozen. It is trained together with the MLP and both heads
(joint fine-tuning, 21 M more parameters trained), on the labelled pixels of the training chips. It trained for 45
epochs on the cluster (job 678194). Inside the declared parcels its parcels are better (PQ 0.355 against 0.307). Over the whole chip it makes even more parcels (535,011 against 351,664, for 55,575 declared ones), and the
crop-ignored PQ falls to 0.036.

## How the boundary head differs from PaPs

| | Boundary head | PaPs ([its architecture](../2026-11-05_paps_tessera/ARCHITECTURE.md)) |
|---|---|---|
| After TESSERA | per-pixel MLP: one pixel at a time | U-Net: sees neighbours |
| Heads | 2: crop, edge | 5: centre, saliency, size, shape, crop |
| Learns from | declared parcels only | declared parcels, plus WorldCover's non-farm land as background |
| A parcel is | a region between predicted edges | one detected centre with a mask drawn around it |
| Land with no parcel | impossible: every pixel of the area ends in some parcel | stays empty |

## Reproduce

    poetry run python experiments/2026-11-05_parcel_extent/architecture.py --chip EE_02265_01779

It needs the delineation experiment's stored frozen cell and the pilot TESSERA cache on the hub (for the embedding
image only). The chip must be one of the nine pilot test chips. The script writes `figures/architecture.png` and its
three parts (`architecture_1.png` to `architecture_3.png`).
