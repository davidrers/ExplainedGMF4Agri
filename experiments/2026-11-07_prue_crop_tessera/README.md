# One model for crop types and field borders: PRUE with a crop-type head on TESSERA, 7 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it. Asked by the user on
7 November 2026: "for the sake of compactness it would be better if we can get to 1 model that makes the
segmentation + the border delineation with the same metrics of the individual ones", then "launch two experiments
for TESSERA frozen and fine-tuned only for 100 %, the goal is to achieve better or same results as the two
independent models that we have but with one". The crop head reads the embedding and the U-Net's decoder features
together (option 2 of the discussion, the user's choice).

**Question.** Can one network, PRUE's U-Net on TESSERA v2 Medium with a crop-type head beside its border head,
trained once on both tasks, match or beat the two separate models on each one's own metrics? The separate models,
at K = 100 %, draw 0, seed 0, on the 1,475 test chips of `EE_2021_rev`:

| Encoder | Crop types: the main workflow's MLP (pixel / parcel Macro-F1) | Borders: PRUE, grown fields (border P / R / F1 at 1 px; PQ agnostic; PQ class-aware) |
|---|---|---|
| frozen | 0.618 / 0.602 (`results/kshot`) | 0.947 / 0.790 / 0.862; 0.374; 0.367 with the MLP map's class |
| fine-tuned | 0.660 / 0.639 (`2026-10-15_kshot_finetune`) | 0.938 / 0.807 / 0.868; 0.378; 0.397 with the fine-tuned map's class |

**What it varies.**

| Arm (`arms_run/`) | Encoder |
|---|---|
| `joint_tessera_v2_frozen` | the core's frozen raster cache (`rasters.py`), as PRUE's frozen arm |
| `joint_tessera_v2_ft` | the pretrained v2 Medium student on the composites of every training window, trained with both heads end to end at 1e-5 after a one-epoch warm-up, as PRUE's fine-tuned arm (`tessera_ft.py`) |
| `joint_tessera_v2_frozen_cw001`, `joint_tessera_v2_ft_cw001` | as the two above with the crop loss weighted 0.01, added 8 November 2026 after the frozen 0.1 arm's crop pull on the U-Net ran at 1.4 to 9.8 times the border loss's and its borders fell below PRUE's |

**The model** (`joint.py`). PRUE's network unchanged: smp's U-Net with an EfficientNet-B7 encoder (ImageNet), its
first convolution on TESSERA's 128 dimensions, its head from the decoder's last 16 channels to background,
interior and boundary. Beside it, the main workflow's crop head, layer for layer (two 1 x 1 layers of 512 and 256,
each ReLU, batch norm and dropout 0.2, then dropout 0.2 and a 1 x 1 convolution to the 20 classes), reading the
128 dimensions of the embedding and the U-Net's 16 decoder channels together, 144 per pixel. The crop head keeps
the per-pixel evidence of the MLP map and adds the U-Net's context; its loss trains the U-Net through those 16
channels (and the student, fine-tuned). One forward pass gives both maps; 67.2 M parameters in the U-Net, 0.21 M in
the crop head.

**Training** (`fit_joint.py`, from PRUE's `fit_prue.py`). Loss: PRUE's log-cosh Dice (class weights 0.05 / 0.20 /
0.75) plus a weight times the crop head's cross-entropy on the labelled pixels of the drawn parcels (the main
workflow's `ce`). Adam's step does not depend on a gradient's scale, so the weight acts only on what the heads share:
it was set from `check.py --balance` so that the crop loss does not take over the U-Net PRUE's borders come from
(value and measurement in the arm files). The U-Net with PRUE's optimiser (AdamW with amsgrad, 3e-3, weight decay
0.01), the crop head with the main workflow's (AdamW, 1e-3, weight decay 0.05), both on one cosine to 1e-6 / 3e-3
of their rate; the encoder, fine-tuned, at 1e-5. PRUE's recipe otherwise: 30 epochs, batches of 8 random 128 x 128
windows with D4, bf16. The two losses' pulls on the U-Net are logged once per epoch.

**Checkpoint.** One model, one epoch: the best validation `(object F1 of PRUE's own fields + crop pixel Macro-F1)
/ 2`. The epochs each criterion alone would choose are recorded beside it.

**Scoring** (`score_joint.py`), each head exactly as its separate model: the crop types by the core's pixel and
parcel (D28) scores; the parcels by PRUE's void-rule PQ (`score_prue.py`, `panoptic_x.py`), class-aware with each
field's class voted by the model's own crop head; the borders by the border-accuracy experiment's precision, recall
and F1 of the grown fields at 0, 1 and 2 px (`borders.py`, copied unchanged). `check.py --scorer` scores a file
assembled from the two separate frozen models' stored outputs and must reproduce their published numbers.

**Protocol.** K = 100 % only (a method check, user, 5 November 2026), draw 0, seed 0, all of `EE_2021_rev`. One
cluster job per arm. Pilot on the hub's CPU, its GPU being taken by another process: the frozen arm through
`run.py` on `EE_2021_rev_mini` (`experiment_pilot.yaml`, 2 epochs: training, validation, checkpoint, test,
scoring), and the fine-tuned route through `fit_joint.fit` on two chips per split for one epoch.

**Crop loss weight: 0.1** (`check.py --balance`, 7 November 2026, on `EE_2021_rev_mini`). On the shared U-Net the
crop loss's gradient is 28 times the border loss's at the start (16 batches, 18 to 36) and 3.7 times after 40
steps (1.7 to 7.1). At 0.1 the crop loss pulls 2.8 times as hard as the border loss for the first steps, then about
0.4 times: PRUE's task leads the U-Net, the crop task has a say. The per-epoch probe shows the balance on the full
run.

**Files.** Copied unchanged from `../2026-11-06_prue_tessera`: `prue.py`, `rasters.py`, `tessera_ft.py`,
`score_prue.py`, `panoptic_x.py`; from `../2026-11-07_border_accuracy`: `borders.py`. `data.py` is PRUE's with the
crop target added. New: `joint.py`, `fit_joint.py`, `score_joint.py`, `run.py`, `check.py`, `report.py`, and
`architecture.py`, which draws the figures of [ARCHITECTURE.md](ARCHITECTURE.md): the model step by step on one test
chip, from the two heads' outputs to fields with a crop, and four test chips to check the results.

**How to run.**

    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev_mini
    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev_mini --balance --steps 40 --weight 0.1
    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev --scorer
    poetry run python experiments/2026-11-07_prue_crop_tessera/run.py -e experiments/2026-11-07_prue_crop_tessera/experiment_pilot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --device cpu
    TIME=3-00:00:00 bash scripts/cluster/submit.sh experiments/2026-11-07_prue_crop_tessera/experiment.yaml EE_2021_rev   # cluster
    bash scripts/cluster/pull_results.sh 2026-11-07_prue_crop_tessera EE_2021_rev
    poetry run python experiments/2026-11-07_prue_crop_tessera/run.py -e experiments/2026-11-07_prue_crop_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --summarise

**Results.** `results/2026-11-07_prue_crop_tessera/<set>/<arm>/P100_draw0_seed0/`: `results.json` (with
`crop_metrics`, `parcel_metrics`, `panoptic_metrics`, `border_metrics` and the training history), `metrics.csv`,
`predictions_test.npz`, `borders_per_chip.npz`, `best.pt`; `summary.csv` beside the arms. Figures and tables:
`report.py`.

**Outcome, 11 November 2026** ([REPORT.md](REPORT.md)). With the crop loss weighted 0.01, one network matches PRUE's
borders and beats the MLP's crop map. Fine-tuned it reaches crop pixel / parcel Macro-F1 0.678 / 0.658 against
0.660 / 0.639, border F1 at 1 px 0.870 against 0.868, and class-aware PQ 0.411 against 0.397. Frozen: 0.624 / 0.604
against 0.618 / 0.602, and border F1 0.861 against 0.862. At 0.1 the crop loss pulls on the U-Net three times as hard
as the border loss, so the crop map gains more (+0.021, +0.027 per pixel) but the borders fall by 0.017 to 0.020 F1.
The balanced weight of `../2026-11-09_borders_all_arms` does better still on frozen TESSERA (0.645 / 0.621, border F1
0.868). One draw and one seed, K = 100 % only.
