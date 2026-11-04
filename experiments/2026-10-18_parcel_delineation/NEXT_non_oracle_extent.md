# Next: parcels without the oracle extent

Handoff note of 5 November 2026, for the session that starts the follow-up experiment. It follows
[REPORT.md](REPORT.md) of this folder; read that first. Agreed with the user: add two non-oracle extents beside the
current one, and score them with the void rule below.

## Why

Every parcel of this experiment is built and scored inside **E**, `(mask >= 0) & (parcels > 0)`: the pixels of the
declared parcels of the 20 in-scheme classes, taken from the truth. E is an oracle first stage, stronger than a
cropland mask, and it gives the method three things a real map would not have:

* where fields end against forest, roads and villages, so every outer edge is free;
* every parcel isolated by non-crop land, 28 % of the test parcels, found for free;
* no false parcels outside the fields, since nothing outside E is built or scored.

The fair comparison still holds: every arm gets the same E, and the `extent` variant (E alone, nothing predicted,
class-agnostic PQ 0.486) measures what E gives away. Against that floor, the GFM boundary heads add nothing
(0.462 to 0.512) and the U-Net does (0.524 to 0.561). But the absolute values (class-aware PQ 0.13 to 0.34 at
tau 0.5) are an
upper bound of stage 2 under a perfect stage 1, and are not comparable with PASTIS, where background is predicted.
Stage 1, the cropland mask of `docs/thesis_design.md`, is not built (`docs/phase1/pipeline.md` section 10).

## What to add

Two extents beside E, each with the same variants as now:

| Extent X | Taken from | Reads as |
|---|---|---|
| `oracle` E (current) | the declared in-scheme parcels | upper bound |
| `worldcover` | ESA WorldCover 2021, cropland (40) plus grassland (30) | realistic, close to what stage 1 would give |
| `chip` | nothing, the whole chip | lower bound |

For every X: `extent` (the 4-connected regions of X alone, the new floor; for `chip` that is one parcel per chip),
`semantic`, `boundary@tau` and `combined@tau` for tau in 0.3, 0.5, 0.7. The `oracle` variant (true pieces, class
by vote) does not depend on X and stays as it is.

**No refit.** The stored `predictions_test.npz` cover the whole 224 x 224 chip, not only the labelled pixels
(checked 5 November 2026 on `tessera_v2_medium/P100_draw0_seed0`: outside the labelled pixels `pred` takes all 20
classes and `boundary` averages 0.33). Both heads were trained on labelled pixels only, so their output on forest
or villages is untrained extrapolation; that is what the non-oracle extents test. The crop-type head has no
background class, so a parcel built on forest still gets a crop class.

## Building the parcels: no ground truth

Unchanged from `panoptic.from_boundary`, with X in place of E:

1. seeds: the 4-connected pieces of `X & (P < tau)`;
2. `skimage.segmentation.watershed(P, markers=seeds, mask=X, connectivity=1)`;
3. a region of X holding no seed becomes one parcel;
4. each parcel's class: the vote of its predicted pixels (protocol D28), over all its pixels. The predictor does not
   know where void lies, so void is removed only when scoring.

## Scoring: the truth is used only here

Each test chip splits into three kinds of land, from `<id>.mask.tif` and `<id>.parcels.tif`. `parcels.tif`
rasterises every declared polygon and `mask.tif` only the in-scheme ones (`gfm4agri.data.chip_grid.rasterise_labels`):

| Land | Pixels | Role |
|---|---|---|
| true parcels | `mask >= 0` | the true units, unchanged: 4-connected pieces of each in-scheme parcel (55,575 on the test chips) |
| void | `mask == -1` and `parcels > 0` | declared parcels of crops outside the 20 classes; the truth knows a field is there but has no class for it |
| background | `parcels == 0` | not a declared parcel; a predicted parcel there is false |

The rule, the void handling of Kirillov et al. (2019). Check how utae-paps' `PanopticMeter` treats PASTIS' void
label before citing it as the same.

1. Void pixels are removed from every predicted parcel before the IoU: they count in neither the intersection nor
   the union.
2. The IoU of a predicted parcel and a true piece uses the predicted parcel's whole non-void area, background
   included.
3. Match at IoU > 0.5, which keeps matches unique.
4. An unmatched predicted parcel whose pixels are more than half void is dropped: neither TP nor FP.
5. Every other unmatched predicted parcel is an FP. Every unmatched true piece is an FN, including true pixels
   outside X, where no parcel was built.

Two places in `panoptic.py` assume X = E and must change:

* `match` raises `ValueError` when a true pixel has no predicted parcel (`seg == 0`); outside X that is now expected.
* `match` counts a segment's area only on true pixels (`as_ = np.bincount(seg[true > 0])`); it must count the
  segment's non-void pixels anywhere in the chip.

`chip_counts` counts the predicted parcels from `np.unique(seg[seg > 0])`; the dropped ones (rule 4) must leave
that count. Class-aware scoring is as now: a match must also agree on the class, per class, then a mean over the
classes that occur.

**Regression check.** With X = E there is no void and no background inside X, so the new scorer must reproduce
the stored `panoptic_metrics` of every cell's `results.json` to every digit. Add synthetic cases as `check.py`
does: a predicted parcel on void is dropped, one on background is an FP, a true piece outside X is an FN.

**Known leak.** Farmland nobody declared counts as background, so a correct parcel there scores as an FP. It should
be small in Estonia, where the declaration carries the subsidy. Report the share of WorldCover cropland outside any
declared parcel, and state the leak in the caveats.

The boundary pixel F1 stays scored inside E: outside E there is no border truth.

## WorldCover

* ESA WorldCover 2021 (v200), 10 m, on the Planetary Computer, as far as known collection `esa-worldcover` with
  asset `map`, tiles in EPSG:4326; verify the names when fetching. Classes: 10 tree cover, 20 shrubland,
  30 grassland, 40 cropland, 50 built-up, 60 bare or sparse vegetation, 70 snow and ice, 80 permanent water,
  90 herbaceous wetland, 95 mangroves, 100 moss and lichen.
* Fetch onto the chip grid (EPSG:3035, 10 m, nearest neighbour) for the 1,475 test chips only. `fetch_scl.py` of
  this folder does the same for SCL, through `gfm4agri.chips.s2_monthly._search` and `stackstac`.
* Save outside the chip set, e.g. `results/<name>/EE_2021_rev/worldcover_test.npz`, one uint8 array per chip id.
  Never write into `data/eurocrops_chips/`.
* Default mask: classes 40 and 30. Grassland is included because our grassland class falls mostly under WorldCover
  grassland, not cropland. Before fixing the set, tabulate for each of our 20 classes the share of its labelled
  test pixels in each WorldCover class. If herbaceous wetland (90) holds a notable share of a class, add a
  sensitivity run with it.

## SCL was considered and rejected as the extent

The July SCL of `fetch_scl.py` serves only the figures, to blank the crop-type maps outside vegetation. No score
uses it. As an extent it fails:

* it has no cropland class, and it calls forest vegetation;
* it drops mown and ripening fields: on example chip 3, July SCL calls a third of the labelled pixels not
  vegetated, 89 % of the legumes, 67 % of the winter wheat and 65 % of the grassland;
* the chips do not store it, since the compositor drops it after cloud screening.

## Where things are

* This folder: `panoptic.py` (`pieces`, `vote`, `from_boundary`, `match`, `chip_counts`, `quality`, `chip_parts`,
  `score_predictions`), `delineation.py` (`boundary_map`, the border target), `diagnose.py`, `report.py`,
  `fetch_scl.py`, `check.py`, `REPORT.md`.
* Results, 18 cells, six arms at K = 100, 20 and 5 %, draw 0, seed 0:
  `results/2026-10-18_parcel_delineation/EE_2021_rev/<arm>/P<k>_draw0_seed0/` (`results.json` with
  `panoptic_metrics`, `predictions_test.npz` with `pred`, `mask`, `boundary` as uint8 P x 255, `ids`). Beside them:
  `kshot_panoptic.json`, the main workflow's crop-type-only fits (`results/kshot/EE_2021_rev/`) scored with
  `oracle` and `semantic`; `diagnostics.json`.
* Chips: `data/eurocrops_chips/EE_2021_rev/chips/<id>.mask.tif` and `<id>.parcels.tif`, read with
  `gfm4agri.benchmark.parcel_metrics.read_parcels`.

## Suggested way to run it

* A **"run an experiment"** case: the main workflow is not touched. Start a new folder,
  `experiments/<date>_parcel_extent/`, from `experiments/_template/`. It fits nothing: it scores the stored
  predictions of this experiment. Its README says so.
* Copy `panoptic.py` into the new folder and extend the copy (an extent argument, the void rule). Leave this
  folder's files as they are, so its report stays reproducible.
* Scripts: `fetch_worldcover.py`, a scorer writing one JSON per cell and extent, `report.py`, then `REPORT.md`.
  Everything runs on the hub, on the CPU; no cluster is needed.
* Optional: the same extents on the `kshot` crop-type-only fits (`semantic` only), as `kshot_panoptic.json` does
  for E.

## What the report should show

* Per arm and K, for each extent: PQ class-aware and class-agnostic, SQ, RQ, TP, FP and FN, the number of predicted
  parcels, and the extent's own floor.
* The three bounds side by side, and what the boundary head adds over each extent's floor.
* The FPs split by where they lie: on background, by WorldCover class (trees, built-up, water), and the parcels
  dropped as void.
* Example chips with the WorldCover mask and the parcels built over it, and over the whole chip.
* Caveats: the undeclared-farmland leak, WorldCover's own errors, heads untrained outside labelled pixels, and
  boundary F1 still inside E.

## Open items

* The WorldCover class set: 40 + 30 by default, 90 as a sensitivity if the tabulation calls for it.
* Proposed, not yet agreed: reword the answer at the top of this folder's `REPORT.md` so that it names the oracle
  extent plainly, not only in the caveats.
* A `pipeline.md` section 11 entry, for this experiment and the follow-up, only once the user agrees.
