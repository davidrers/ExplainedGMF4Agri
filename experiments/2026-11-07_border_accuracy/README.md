# Drawing parcels from TESSERA: PRUE, PaPs and the boundary head compared, 7 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it.

**Question.** Does the delineation each method predicts match the declared one? Is a pixel drawn as a border a real
border, and is each real border drawn?

**What it uses.** No new fit. The stored test predictions of six fits on `EE_2021_rev`, K = 100 %, draw 0, seed 0:

| Method | Frozen TESSERA v2 | Fine-tuned TESSERA v2 |
|---|---|---|
| PRUE | `results/2026-11-06_prue_tessera/.../prue_tessera_v2_frozen` | `.../prue_tessera_v2_ft` (encoder trained end to end) |
| PaPs | `results/2026-11-05_paps_tessera/.../paps_tessera_v2_frozen` | `.../paps_tessera_v2_ft` (encoder fine-tuned on crop type, then frozen) |
| Boundary head | `results/2026-10-18_parcel_delineation/.../tessera_v2_medium` | `results/2026-11-05_parcel_extent/.../tessera_v2_medium_ft_bnd` (encoder trained with both heads) |

Each method's final fields over the whole chip, built with no test label, are compared with the declared parcels:
border precision, recall and F1 at a tolerance of 0, 1 and 2 pixels (`borders.py`). `prue.py` and `panoptic_x.py`
are unchanged copies from `../2026-11-06_prue_tessera/`, so this folder stands on its own.

**How to run.**

    poetry run python experiments/2026-11-07_border_accuracy/borders.py --chips EE_2021_rev --workers 12   # about 40 s
    poetry run python experiments/2026-11-07_border_accuracy/landcover.py --chips EE_2021_rev --workers 12 # about 30 s
    poetry run python experiments/2026-11-07_border_accuracy/robustness.py --chips EE_2021_rev --workers 12 # about 30 s
    poetry run python experiments/2026-11-07_border_accuracy/report.py --chips EE_2021_rev                 # about 1 min
    poetry run python experiments/2026-11-07_border_accuracy/walkthrough.py --chips EE_2021_rev            # about 30 s

`walkthrough.py` draws the WorldCover figure and the one-chip example of the architecture notes of PaPs, PRUE and the
boundary head, into those experiments' `figures/` (`worldcover_training.png`, `example_walkthrough.png`). `targets.py`
is an unchanged copy of PaPs' target builder, so the centre target drawn is the one PaPs was trained on.

**Results.** `results/2026-11-07_border_accuracy/EE_2021_rev/`: `borders.json` (every score), `per_chip.npz` (the
counts of every test chip at 1 pixel) and `landcover.npz` (declared parcels, undeclared WorldCover farmland and
non-farm land in every chip of the split) and `robustness.json` (strict precision, fields with nothing split,
recall on the borders between parcels). Figures and `tables.md` in `figures/`.

**Outcome** ([REPORT.md](REPORT.md), the comparison of the three methods: what each does in plain words, the
same chip through all three, the scores, and the limitations). With a tolerance of one pixel (10 m), PRUE's borders match the declared ones
best: 95 % of the borders it draws are real and it draws 79 to 81 % of the real ones (F1 0.86 frozen, 0.87
fine-tuned). PaPs draws real borders too (85 %) but only half of them (F1 0.64, 0.68), because it leaves parcels
out. The boundary head draws borders across forest and wetland (F1 0.36, 0.21). PRUE scores above PaPs on 99 % of
the test chips with at least 20 parcels. What all three miss most is the border between two parcels of the same
crop: PRUE draws one in three. A third of WorldCover's farmland in the chips is undeclared; none of the three
methods is trained as if a chip were complete (each ignores undeclared farmland), but the parcel score (PQ) and the
checkpoint choice count fields drawn there as false.
