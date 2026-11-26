"""Are the three external models wired as their authors run them? Three checks, written to ``reference/wiring.json``.

* ``ftw`` (own environment, needs ``ftw-tools``)::

      env -u PYTHONPATH ~/.venvs/open_delineation/bin/python experiments/2026-11-26_open_delineation/check_wiring.py ftw

  Five test chips are written as FTW's eight-band input (R, G, B, NIR of May, then of August), ftw-tools' own
  ``inference.run`` predicts them with the released PRUE B7 checkpoint (``--resize_factor 2 --save_scores``, one
  224 px patch), and its scores are compared with those this experiment stored for the same chips.
* ``delany`` (own environment): ftw-tools' ``DelineateAnything`` wrapper predicts five chips (August) and its
  fields, painted by ``predict_delany.paint``, are compared with the stored ``delany_v2/ftw_Aug``.
* ``utae`` (Poetry environment)::

      poetry run python experiments/2026-11-26_open_delineation/check_wiring.py utae

  Four patches of PASTIS' fold 5 (the test fold of the ``Fold_1`` weights) are read from the PASTIS archive on
  Zenodo by HTTP range requests, predicted as ``predict_utae`` predicts our chips (PASTIS' fold-5 statistics, its
  own dates), and the class-agnostic RQ at IoU > 0.5 against PASTIS' instances is reported: the authors' fold-1 test
  RQ (class-aware) is 0.48, so a value near it shows the weights, the band order and the dates are right.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile

import numpy as np

import common as C

OUT = C.HERE / "reference" / "wiring.json"
CHIPS = 5


def record(key: str, value: dict) -> None:
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = value
    OUT.write_text(json.dumps(d, indent=1))
    print(key, json.dumps(value, indent=1))


def stored(model: str, variant: str, ids: list[str]):
    with np.load(C.RESULTS / model / variant / "predictions_test.npz") as z:
        pos = {str(c): i for i, c in enumerate(z["ids"])}
        key = "prob" if "prob" in z.files else "inst"
        return [z[key][pos[c]] for c in ids]


def check_ftw():
    import rasterio
    from ftw_tools.inference.inference import run

    tmp = C.RESULTS / "wiring"
    tmp.mkdir(parents=True, exist_ok=True)
    ids = C.split_ids("test")[::300][:CHIPS]
    ours = stored("prue_b7", "AMay_BAug", ids)
    rows = []
    for cid, mine in zip(ids, ours):
        s2 = C.read_s2(cid)
        x = np.concatenate([s2[4][C.RGBN], s2[7][C.RGBN]]).astype(np.uint16)
        with rasterio.open(C.CHIP_DIR / f"{cid}_merged.tif") as src:
            prof = src.profile
        prof.update(count=8, dtype="uint16", nodata=None)
        src_tif, out_tif = tmp / f"{cid}_ftw8.tif", tmp / f"{cid}_ftw_scores.tif"
        with rasterio.open(src_tif, "w", **prof) as dst:
            dst.write(x)
        run(input=str(src_tif), model=str(C.WEIGHTS / "ftw" / "prue_efnetb7_ccby_checkpoint.ckpt"),
            out=str(out_tif), resize_factor=2, gpu=0, patch_size=224, batch_size=1, num_workers=0, padding=None,
            overwrite=True, mps_mode=False, save_scores=True)
        with rasterio.open(out_tif) as src:
            theirs = src.read()
        rows.append({"chip": cid, "max_abs_diff_x255": int(np.abs(theirs.astype(int) - mine.astype(int)).max()),
                     "argmax_agreement": float((theirs.argmax(0) == mine.argmax(0)).mean())})
    record("ftw_prue_b7_vs_ftw_tools_run", {"chips": rows,
                                            "mean_argmax_agreement": float(np.mean([r["argmax_agreement"]
                                                                                    for r in rows]))})


def check_delany():
    import torch
    from ftw_tools.inference.models import DelineateAnything

    import predict_delany as P

    ids = C.split_ids("test")[::300][:CHIPS]
    ours = stored("delany_v2", "ftw_Aug", ids)
    import os

    (C.RESULTS / "wiring").mkdir(parents=True, exist_ok=True)
    os.chdir(C.RESULTS / "wiring")              # ultralytics downloads the wrapper's weights into ./weights
    model = DelineateAnything(model="DelineateAnythingV2", patch_size=224, resize_factor=2, max_detections=100,
                              iou_threshold=0.3, conf_threshold=0.05, device="cuda")
    rows = []
    for cid, mine in zip(ids, ours):
        rgb = torch.from_numpy(C.read_s2(cid)[7][C.RGB].astype(np.float32))[None]
        res = model(rgb)
        inst, _ = P.paint(res[0], smooth_masks=False)
        # ids are confidence ranks, so compare the fields: each of ours against its best match in theirs
        best = []
        for j in np.unique(mine[mine > 0]):
            a = mine == j
            cand, cnt = np.unique(inst[a], return_counts=True)
            k = cand[np.argmax(cnt)]
            best.append(0.0 if k == 0 else (a & (inst == k)).sum() / (a | (inst == k)).sum())
        rows.append({"chip": cid, "fields_ftw_tools": int(len(np.unique(inst)) - 1),
                     "fields_ours": int(len(np.unique(mine)) - 1),
                     "field_pixel_agreement": float(((inst > 0) == (mine > 0)).mean()),
                     "share_of_our_fields_matched_iou_0.9": float(np.mean(np.array(best) >= 0.9)),
                     "mean_best_iou": float(np.mean(best))})
    record("delany_v2_ftw_recipe_vs_ftw_tools_wrapper", {"chips": rows})


class HTTPFile(io.RawIOBase):
    """A remote file read by HTTP range requests, enough for ``zipfile``."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or not len(b):
            return 0
        end = min(self.pos + len(b), self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        data = urllib.request.urlopen(req, timeout=300).read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


PASTIS = ("https://zenodo.org/api/records/5012942/files/PASTIS.zip/content", 28760245504)


def check_utae():
    import datetime as dt

    import torch

    import predict_utae as U

    z = zipfile.ZipFile(io.BufferedReader(HTTPFile(*PASTIS), buffer_size=1 << 20))
    meta = json.loads(z.read("PASTIS/metadata.geojson"))
    feats = [f["properties"] for f in meta["features"] if f["properties"]["Fold"] == 5][:4]
    norm = json.loads(U.NORM_PASTIS.read_text())["Fold_5"]
    mean, std = (torch.tensor(norm[k]).float()[None, None, :, None, None].cuda() for k in ("mean", "std"))
    net, _ = U.build(1, torch.device("cuda"))
    tp = fp = fn = 0
    for p in feats:
        pid = p["ID_PATCH"]
        x = np.load(io.BytesIO(z.read(f"PASTIS/DATA_S2/S2_{pid}.npy"))).astype(np.float32)
        inst_t = np.load(io.BytesIO(z.read(f"PASTIS/INSTANCE_ANNOTATIONS/INSTANCES_{pid}.npy")))
        sem_t = np.load(io.BytesIO(z.read(f"PASTIS/ANNOTATIONS/TARGET_{pid}.npy")))[0]
        dates = p["dates-S2"]
        d = dates if isinstance(dates, dict) else json.loads(dates)
        ref = dt.date(2018, 9, 1)
        days = [(dt.date(int(str(v)[:4]), int(str(v)[4:6]), int(str(v)[6:])) - ref).days for v in d.values()]
        xt = ((torch.from_numpy(x)[None].cuda() - mean) / std).contiguous()
        xt = torch.cat([xt, xt])
        pos = torch.tensor(days, dtype=torch.float32).cuda()[None].expand(2, -1)
        with torch.inference_mode():
            out = net(xt, batch_positions=pos, pseudo_nms=True, prune=True)
        pi = out["pano_instance"][0].long().cpu().numpy()
        cls = out["pano_semantic"][0].argmax(0).cpu().numpy()
        keep = [j for j in np.unique(pi[pi > 0]) if np.bincount(cls[pi == j]).argmax() not in U.DROP_CLASSES]
        true = [j for j in np.unique(inst_t[inst_t > 0])
                if np.bincount(sem_t[inst_t == j].astype(int)).argmax() not in U.DROP_CLASSES]
        matched = set()
        for j in keep:
            a = pi == j
            for t in true:
                b = inst_t == t
                inter = (a & b).sum()
                if inter and inter / (a | b).sum() > 0.5:
                    matched.add(t)
                    break
        tp += len(matched)
        fp += len(keep) - len(matched)
        fn += len(true) - len(matched)
    record("utae_paps_fold1_on_pastis_fold5", {"patches": [p["ID_PATCH"] for p in feats], "tp": tp, "fp": fp,
                                               "fn": fn, "rq_agnostic": tp / max(tp + fp / 2 + fn / 2, 1e-9),
                                               "authors_fold1_test": {"PQ": 0.397, "SQ": 0.812, "RQ": 0.484}})


if __name__ == "__main__":
    {"ftw": check_ftw, "delany": check_delany, "utae": check_utae}[sys.argv[1]]()
