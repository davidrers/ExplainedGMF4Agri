# Parcel delineation by a boundary head trained beside the crop-type head, 18 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it; `run.py` swaps in this
folder's code (`delineation.py`, `panoptic.py`).

**Question.** Each arm of the K-shot workflow is fitted with one more output beside its crop-type head: a boundary
logit per pixel, trained jointly, under the same label budget. How well does each arm then delineate the test
parcels, scored by Panoptic Quality (PQ), and how much parcel-level accuracy is lost when the parcels come from the
model instead of from the parcel layer? Does the boundary head add anything over what the crop-type map gives on its
own, and does training it change the pixel Macro-F1 that answers RQ1?

Motivation: on 300 chips of `EE_2021_rev`, 52 % of the parcels touch a parcel of the same class, so even a perfect
crop-type map merges about half of them; the study of 16 October 2026 compared PaPs (utae-paps), Mask R-CNN and a
boundary head, and chose the last as the option that keeps the label budget, the decoders and the crop-type
predictions of the main workflow as they are.

**What it varies.** The six arms of `configs/experiments/kshot.yaml`, unchanged in `configs/arms/`, at K = 100, 20
and 5 %, draw 0, seed 0, 45 epochs, checkpoint of lowest validation loss. What differs from the main workflow:

* **Target** (`delineation.py`): a pixel of a parcel is a boundary pixel when a 4-neighbour inside the chip carries
  another parcel identifier or none. It exists on the labelled pixels only, so under a budget only the drawn parcels
  carry it (the strict reading: one polygon is one annotation, of class and geometry together). It travels inside
  the mask, as `class + 1000 x boundary`, so every loader keeps it aligned with the crop-type label.
* **Head**: one logit beside the arm's segmentation head, on the same decoder features, built like the head's
  classifier. Neck, decoder and encoder are the arm's, so the decoders stay matched within each resolution group.
* **Loss**: cross-entropy on crop types plus 1.0 x binary cross-entropy on the boundary, the same weight for every
  arm; `val/loss`, the checkpoint criterion, is the sum.
* **Parcels and score** (`panoptic.py`): inside the true parcel extent of every test chip (the labelled pixels),
  seeds where the boundary probability is below 0.5, grown by watershed; each predicted parcel takes the majority
  class of its pixels (the vote of protocol D28). Matched to the true parcel pieces at IoU > 0.5; PQ, SQ and RQ per
  class and class-agnostic. Three reference variants on the same units: `oracle` (the true parcels, only the class
  predicted), `semantic` (connected regions of one predicted class) and `combined` (boundary seeds also cut where
  the class changes); the threshold is varied to 0.3 and 0.7 as a check.

The crop-type-only fits it is compared with are the main workflow's at 45 epochs, `results/kshot/EE_2021_rev/`,
fitted on the same draws and seed; `report.py` scores their `oracle` and `semantic` variants too.

**How to run.**

    poetry run python experiments/2026-10-18_parcel_delineation/check.py                     # synthetic checks
    poetry run python experiments/2026-10-18_parcel_delineation/run.py -e experiments/2026-10-18_parcel_delineation/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini   # hub
    bash scripts/cluster/submit.sh experiments/2026-10-18_parcel_delineation/experiment.yaml EE_2021_rev                                                                    # cluster

A cell pulled back without panoptic scores gets them with the same `run.py ... --summarise`.

**Results.** `results/2026-10-18_parcel_delineation/<set>/`; `summary.csv` adds the headline panoptic columns, and
every `results.json` carries `panoptic_metrics` and the rules under `delineation`.

**Report.** `poetry run python experiments/2026-10-18_parcel_delineation/report.py --chips <set>` writes
`figures/`, and [REPORT.md](REPORT.md) is written from them.

**Outcome.** Pending.
