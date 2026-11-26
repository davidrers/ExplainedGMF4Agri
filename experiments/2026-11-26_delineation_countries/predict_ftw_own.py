"""FTW's two-window models with each country's own FTW windows: the variant ``own_windows``.

Run in ftw-tools' stack (``env_ftw312.sh`` of the Estonian experiment), after ``predict_ftw.py``::

    env -u PYTHONPATH /tmp/ftw312/env/bin/python experiments/2026-11-26_delineation_countries/predict_ftw_own.py

Fields of The World sets two windows per country (``data_config_<country>.json``, downloaded by ``ftw_overlap.py``
into ``data/external_models/ftw/countries/``), and they differ: Belgium and Denmark take August as window A and April
as window B, Latvia and Slovenia October as window B, Spain September and May. Here each chip reads the middle month
of each of its country's windows, in FTW's order (A then B), from the twelve monthly composites. The four sampled
countries that Fields of The World does not cover (BG, CZ, IE, IT) keep April and July, the pair chosen on the
Estonian validation chips. Everything else is ``predict_ftw.py``'s inference.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import predict_ftw as P

NAMES = {"AT": "austria", "BE": "belgium", "DK": "denmark", "FI": "finland", "FR": "france", "DE": "germany",
         "LV": "latvia", "LT": "lithuania", "NL": "netherlands", "PT": "portugal", "SK": "slovakia",
         "SI": "slovenia", "ES": "spain", "SE": "sweden"}
FALLBACK = (3, 6)                                     # April, July (0-based), chosen on Estonia


def windows() -> dict[str, tuple[int, int]]:
    out = {}
    for cc in C.COUNTRIES:
        f = C.WEIGHTS / "ftw" / "countries" / f"data_config_{NAMES.get(cc, '')}.json"
        if cc not in NAMES or not f.exists():
            out[cc] = FALLBACK
            continue
        s = json.loads(f.read_text())["seasons"]
        s = s[0] if isinstance(s, list) else s
        mids = []
        for w in ("window_a", "window_b"):
            a, b = (dt.date.fromisoformat(s[w][k]) for k in ("start", "end"))
            mids.append((a + (b - a) / 2).month - 1)
        out[cc] = tuple(mids)
    return out


def main():
    import kornia.augmentation as K

    dev = torch.device("cuda")
    up = K.Resize((448, 448)).to(dev)
    down = K.Resize((224, 224), resample="nearest").to(dev)
    win = windows()
    ids = C.split_ids("test")
    for name in ("prue_b7", "v2_multi"):
        rel, nwin, source = P.MODELS[name]
        net, hp = P.load(C.WEIGHTS / rel)
        net = net.to(dev)
        out = np.zeros((len(ids), 3, 224, 224), np.uint8)
        masks = np.zeros((len(ids), 224, 224), np.int16)
        k = 0
        for part, s2, m in C.chips(ids, 8):
            x = torch.from_numpy(s2.astype(np.float32)).to(dev) / P.DIVISOR
            inp = torch.stack([torch.cat([x[i, win[C.country(c)][0]][C.RGBN], x[i, win[C.country(c)][1]][C.RGBN]])
                               for i, c in enumerate(part)])
            with torch.inference_mode():
                p = down(F.softmax(net(up(inp)), dim=1).float())
            out[k:k + len(part)] = (p.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            masks[k:k + len(part)] = m
            k += len(part)
        meta = {"model": name, "source": source, "hyper_parameters": hp, "variant": "own_windows",
                "months_by_country": {cc: [C.MONTHS[a], C.MONTHS[b]] for cc, (a, b) in win.items()},
                "rule": "middle month of each FTW window of the chip's country (data_config_<country>.json), "
                        "A then B; April + July where FTW has no data config"}
        C.save_prob(name, "own_windows", "test", ids, out, masks, meta)
        print(name, "own windows:", {cc: f"{C.MONTHS[a]}+{C.MONTHS[b]}" for cc, (a, b) in win.items()}, flush=True)


if __name__ == "__main__":
    main()
