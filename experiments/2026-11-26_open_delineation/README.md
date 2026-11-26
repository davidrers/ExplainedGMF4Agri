# Open field-delineation models against the two-head workflow, Estonia 2021, 26 November 2026

**Question** (user, 26 November 2026). How does the field delineation of our crop segmentation and field delineation
models compare with top open models, GFM-based or optical, Delineate Anything among them? And, across them: do the
common approaches use temporal data for field delineation?

**What is compared.** Three released models, run zero-shot (no training on our labels) on the main workflow's
validation and test chips of `EE_2021_rev`, chosen so that each uses a different amount of the season:

| Model | Input it reads | Output | Trained on |
|---|---|---|---|
| **Delineate Anything v2** (Lavreniuk et al., ECCV 2026), and v1 | **one date**, RGB | field instances (YOLO) | FBIS-73M (v1: FBIS-22M), 0.25 to 10 m, many sensors |
| **FTW PRUE B7** (Fields of The World v3.1 CC-BY, Muhawenayo et al., CVPR 2026), with FTW v2 one-window and two-window as ablation | **two dates**, RGB + NIR | background, field, boundary | Fields of The World, 24 countries, Estonia 2021 among them |
| **U-TAE + PaPs** (Garnot and Landrieu, ICCV 2021) | **the whole series**, 10 bands | panoptic parcels | PASTIS, France |

No released checkpoint of a geospatial foundation model fine-tuned for field boundaries was found (PRUE's paper
fine-tunes Clay but releases only the EfficientNet U-Nets; AI4Boundaries and PTAViT3D release no weights), so the
GFM side is our own six arms of the two-head workflow, rescored on the same chips from their stored predictions
(`results/2026-11-09_borders_all_arms/`, K = 100, 20 and 5 %).

**Fairness rules.**
- One scorer (`score.py`): the two-head workflow's own functions, read-only, so every model's fields and borders
  are scored as our arms are; rescoring our arms reproduces their stored border F1 and PQ exactly.
- Each external model's variants (month, window pair, recipe, threshold, fold, normalisation) are chosen on the
  validation chips only; for FTW the middle months of its Estonian windows (May and August) are reported beside
  them as an untuned setting.
- **Leakage.** Fields of The World holds Estonia 2021 from the same declarations; 338 of our 1,475 test chips touch
  an FTW chip (`ftw_overlap.py`, `reference/ftw_overlap_table.md`). The headline is `test_clean`, the 1,137 test
  chips that touch none.
- **What a field is.** FTW's Estonian labels call 87 % of our grassland parcel pixels background
  (`ftw_labels.py`, `reference/ftw_label_check.json`). Every model is therefore also scored with grassland parcels
  made void (`no_grass`).

**How to run** (hub, in order; `run_hub.sh` holds the same):

    bash experiments/2026-11-26_open_delineation/fetch_weights.sh        # weights into data/external_models/
    bash experiments/2026-11-26_open_delineation/env.sh                  # ~/.venvs/open_delineation (Delineate Anything)
    bash experiments/2026-11-26_open_delineation/env_ftw312.sh           # /tmp/ftw312/env: ftw-tools' stack, Python 3.12
    poetry run python experiments/2026-11-26_open_delineation/ftw_overlap.py
    poetry run python experiments/2026-11-26_open_delineation/ftw_labels.py
    env -u PYTHONPATH /tmp/ftw312/env/bin/python experiments/2026-11-26_open_delineation/predict_ftw.py
    poetry run python experiments/2026-11-26_open_delineation/predict_utae.py --stage select
    poetry run python experiments/2026-11-26_open_delineation/predict_utae.py --stage final --fold F --norm N
    env -u PYTHONPATH ~/.venvs/open_delineation/bin/python experiments/2026-11-26_open_delineation/predict_delany.py
    poetry run python experiments/2026-11-26_open_delineation/fuse.py
    poetry run python experiments/2026-11-26_open_delineation/score.py
    poetry run python experiments/2026-11-26_open_delineation/check_wiring.py utae
    env -u PYTHONPATH /tmp/ftw312/env/bin/python experiments/2026-11-26_open_delineation/check_wiring.py ftw
    env -u PYTHONPATH /tmp/ftw312/env/bin/python experiments/2026-11-26_open_delineation/check_wiring.py delany
    poetry run python experiments/2026-11-26_open_delineation/report.py

Three environments: the Poetry one (U-TAE + PaPs, the scorer, the report), `~/.venvs/open_delineation`
(Delineate Anything: `ultralytics` brings `opencv-python`, kept out of the Poetry environment) and ftw-tools'
current stack on Python 3.12 (FTW's predictions and the ftw-tools checks: the Poetry environment's
segmentation-models-pytorch 0.5.0 gives 0.5 to 4 % different FTW class maps from ftw-tools, 0.6.0 none). Their
package lists are in `reference/`. Nothing of the core is changed and nothing is trained. There is no `experiment.yaml` or `arms/`: `run_kshot.py` is
not used.

**Results.** `results/2026-11-26_open_delineation/EE_2021_rev/`: `<model>/<variant>/predictions_{val,test}.npz`,
`scores/` (per chip counts, `summary.csv`), `logs/`.

**Report.** [REPORT.md](REPORT.md), with `figures/` from `report.py`.

**Outcome** ([REPORT.md](REPORT.md)). On the 1,137 test chips no FTW chip touches:
- **Our models delineate Estonian fields best.** Border F1 within 1 px: our PRUE baseline 0.857 and TESSERA 0.856,
  against 0.637 for the best open model, FTW's released PRUE B7. Delineate Anything v2 reaches 0.525 (0.598 with
  six dates fused) and U-TAE + PaPs 0.522.
- **FTW's gap is mostly grassland, which its labels call background.** With grassland parcels set aside it reaches
  0.856 against our 0.872, with better fields (PQ 0.467 against 0.423), above our arms trained on 20 % of the
  labels.
- **Temporal data.** The common approaches read one image or two dates; only U-TAE + PaPs and PTAViT3D encode a
  series. Here which dates a model reads matters most: FTW PRUE B7's best pair, April + July, beats the mid-window
  pair, May + August, by 0.10. Averaging six single dates helps the one-image models (Delineate Anything v2 +0.07,
  FTW one-window +0.04) but not the two-date ones (about −0.05 against their best pair).
