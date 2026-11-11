"""The two-head workflow: crop types and field borders from every arm, beside the crop-only K-shot workflow.

Each arm keeps its frozen encoder and gets PRUE's border head beside its crop head (protocol D30); a spec with
``workflow: borders`` (``configs/experiments/kshot_borders.yaml``) runs it through ``scripts/run_kshot.py``. Moved
from ``experiments/2026-11-09_borders_all_arms`` on 11 November 2026, unchanged in behaviour.

* :mod:`~gfm4agri.borders.prue`: PRUE's three-class border target, its log-cosh Dice loss, its U-Net, its fields;
* :mod:`~gfm4agri.borders.networks`: the two network shapes, the crop head and target, the crop-loss balance;
* :mod:`~gfm4agri.borders.data`: the chips of the three inputs (10 m raster, S2 + S1 stack, cached tokens) with both
  targets;
* :mod:`~gfm4agri.borders.fit`: one cell, trained, checkpointed on fields and crop types together, predicted;
* :mod:`~gfm4agri.borders.scoring`: the crop map, the fields (void-rule PQ) and the borders (P, R, F1);
* :mod:`~gfm4agri.borders.runner`: every arm x budget x draw x seed of a spec on one chip set.
"""
