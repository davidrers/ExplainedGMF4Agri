"""Delineate Anything over several dates: its single-date fields fused into one three-class map.

Run in the Poetry environment, after ``predict_delany.py``::

    poetry run python experiments/2026-11-26_open_delineation/fuse.py

Delineate Anything sees one image. The simplest way to give it the season, and the way the multi-date pipelines of
the literature combine single-date predictions (Waldner and Diakogiannis 2020; eo-learn's field delineation), is to
average per-date maps: each month's fields (default confidence 0.15) become a three-class map (background where no
field, boundary on each field's rim as ``prue.boundary_map`` draws it, interior inside), the six months April to
September are averaged, and the fields are built from the mean as from FTW's maps (``prue.parcels(prob, "grow")``).
Saved as the variant ``<recipe>_fused6`` of each model, scored by ``score.py`` like any three-class map.

**The control.** The fused map's fields are built by another rule than the single-date instances' (interiors grown
by watershed, not the model's own instances). Each month alone is therefore also passed through the same path,
``<recipe>_<Mon>_3class`` (its one-hot map, unaveraged), so that one date against six is compared under one field
rule and the gain of fusion splits into the field rule's and the time's.
"""

from __future__ import annotations

import numpy as np

import common as C

CONF = 0.15


def main():
    from gfm4agri.borders import prue

    for model in ("delany_v2", "delany_v1"):
        for recipe in ("official", "ftw"):
            for split in ("test",):
                paths = [C.RESULTS / model / f"{recipe}_{C.MONTHS[m]}" / f"predictions_{split}.npz" for m in C.SEASON]
                if not all(p.exists() for p in paths):
                    continue
                acc = None
                for m, p in zip(C.SEASON, paths):
                    with np.load(p) as z:
                        ids, inst, mask, conf, off = z["ids"], z["inst"], z["mask"], z["conf"], z["conf_offsets"]
                    one = np.zeros((len(ids), 3, 224, 224), np.float32)
                    for i in range(len(ids)):
                        keep = conf[off[i]:off[i + 1]] >= CONF
                        keep[0] = False
                        seg = np.where(keep[inst[i]], inst[i], 0)
                        bnd = prue.boundary_map(seg) & (seg > 0)
                        one[i, prue.BOUNDARY] = bnd
                        one[i, prue.INTERIOR] = (seg > 0) & ~bnd
                        one[i, prue.BACKGROUND] = seg == 0
                    acc = one if acc is None else acc + one
                    single = np.round(255 * one).astype(np.uint8)
                    C.save_prob(model, f"{recipe}_{C.MONTHS[m]}_3class", split, ids, single, mask,
                                {"model": model, "recipe": recipe, "month": C.MONTHS[m], "confidence": CONF,
                                 "rule": "one month's one-hot background / interior / boundary, unaveraged (control)"})
                prob = np.round(255 * acc / len(paths)).astype(np.uint8)
                meta = {"model": model, "recipe": recipe, "fused_months": [C.MONTHS[m] for m in C.SEASON],
                        "confidence": CONF, "rule": "mean over months of one-hot background / interior / boundary"}
                C.save_prob(model, f"{recipe}_fused6", split, ids, prob, mask, meta)
                print(model, recipe, split, "fused", flush=True)


if __name__ == "__main__":
    main()
