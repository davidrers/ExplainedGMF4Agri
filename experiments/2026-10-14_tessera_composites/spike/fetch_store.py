"""Read the TESSERA v1.1 store (native per-acquisition series, AWS v1.1 checkpoint) onto the pilot chips' grid.

Uses the main workflow's chip-window reader, pointed at the v1.1 store instead of v1: nearest-neighbour warp
from the zone's UTM grid onto the chip's EPSG:3035 grid, neighbouring zone read near a zone edge. One float16
(128, 224, 224) .npy per chip. Source Cooperative served a chip in about 90 s on 14 October 2026.

    poetry run python experiments/2026-10-14_tessera_composites/spike/fetch_store.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import CHIPSET, OUT, REPO, chip_ids

sys.path.insert(0, str(REPO / "src"))
from gfm4agri.embeddings.tessera import TesseraStore  # noqa: E402

URL = "https://data.source.coop/tessera/tessera/zarr/v1.1-dclimate"


def main() -> None:
    man = json.load(open(CHIPSET / "manifest.json"))
    wanted = set(chip_ids())
    chips = [c for c in man["chips"] if c["chip_id"] in wanted]
    crs = man["grid"]["crs"]
    store = TesseraStore(URL)
    out = OUT / "emb" / "store_v11"
    out.mkdir(parents=True, exist_ok=True)
    print("store model:", store.model, "chips", len(chips), flush=True)

    def one(c):
        p = out / f"{c['chip_id']}.npy"
        if p.exists():
            return c["chip_id"], "skip", 0.0
        t0, err = time.time(), None
        for attempt in range(3):
            try:
                emb, info = store.read_chip(tuple(c["bounds_3035"]), crs, 2021)
                np.save(p, emb.astype(np.float16))
                return c["chip_id"], info.get("nodata_share"), time.time() - t0
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(30 * (attempt + 1))
        return c["chip_id"], f"FAIL {err!r}", time.time() - t0

    with ThreadPoolExecutor(6) as ex:
        for cid, share, sec in ex.map(one, chips):
            print(cid, share, f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
