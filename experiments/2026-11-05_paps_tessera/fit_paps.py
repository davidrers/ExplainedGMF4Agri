"""One U-Net + PaPs cell: training as utae-paps' ``train_panoptic.py`` does, validation, test predictions, scores.

**Training** (the source's defaults): Adam at 0.01, the rate multiplied by 0.3 after epochs 60 and 80, 100 epochs,
batches of 4 chips (shuffled, last partial batch dropped), the first 5 epochs on the centerness loss alone
(``warmup``), one supervised centre per true parcel (``supmax``), the source's weight initialisation. In
training no panoptic output is built (the source's ``val_metrics_only``). fp32, as the source.

**Validation**, every ``val_every`` epochs after the warm-up and at the last epoch, on the validation chips with
every parcel:

1. a pass with every local maximum as a centre (as the source's validation), giving the validation loss and, for
   each centre in a true parcel's zone, its confidence and the IoU of its mask; the pseudo-NMS confidence threshold
   is then tuned to the F-score of finding masks with IoU > 0.5 (the source's ``autotune``). The source tunes on the
   last validation batch, a side effect of its loss object keeping only the latest one; here the centres of every
   validation chip are pooled. Centres below 0.01 are left out of this pass, which only bounds its cost;
2. a pass with the pseudo-NMS at that threshold; its parcels are scored by the void rule on the whole chip
   (``panoptic_x``, extent ``chip``), and the class-aware PQ chooses the checkpoint, as the source keeps the epoch of
   best validation PQ.

**Test**: the kept model predicts every test chip (``predictions_test.npz``: ``inst``, the parcel id per pixel, 0
for none; ``cls``, its class, 255 for none; ``heatmap`` and ``saliency`` x 255; ``mask``; ``ids``), and
``score_paps.score`` scores the parcels in the three extents.

A ``last.pt`` after every epoch lets a killed job resume where it stopped.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import torch

from data import PapsChips, collate, train_chips
from paps_loss import PaPsLoss
from paps_model import build_model


def _loader(ds, batch, shuffle, workers, drop_last=False):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=False)


def tune_threshold(confidences: np.ndarray, ious: np.ndarray) -> tuple[float, float]:
    """The source's ``tune_threshold``: the confidence cut maximising the F-score of masks with IoU > 0.5."""
    t = ious > 0.5
    p = confidences
    best_score, best_threshold = 0.0, 0.0
    for ct in np.arange(0, 1, 0.01):
        tp = ((p > ct) & t).sum()
        fp = ((p > ct) & ~t).sum()
        fn = ((p < ct) & t).sum()
        score = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
        if score > best_score:
            best_threshold, best_score = float(ct), float(score)
    return best_threshold, best_score


@torch.no_grad()
def predict(model, loader, device) -> dict:
    """``chip -> (inst uint16, cls uint8, heatmap uint8, saliency uint8, mask int16)`` with the pseudo-NMS."""
    model.eval()
    out = {}
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        p = model(x, pseudo_nms=True, prune=True)
        inst = p["pano_instance"].long().cpu().numpy()
        cls = p["pano_semantic"].argmax(1).cpu().numpy()
        heat = torch.round(p["heatmap"][:, 0] * 255).byte().cpu().numpy()
        sal = torch.round(torch.sigmoid(p["saliency"][:, 0]) * 255).byte().cpu().numpy()
        mask = batch["target"][..., 6].numpy().astype(np.int16)
        for i, cid in enumerate(batch["chip"]):
            c = np.where(inst[i] > 0, cls[i], 255).astype(np.uint8)
            out[cid] = (inst[i].astype(np.uint16), c, heat[i], sal[i], mask[i])
    return out


def validate(model, crit, loader, device, chips_dir, class_names, log=print) -> dict:
    import score_paps

    model.eval()
    sums, n = np.zeros(5), 0
    confs, ious = [], []
    keep_conf = model.min_confidence.clone()
    model.min_confidence.fill_(0.01)
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            p = model(x, pseudo_nms=False, prune=True)
            loss = crit(p, y)
            sums += [float(loss), *crit.value]
            n += 1
            if crit.predicted_confidences is not None:
                confs.append(crit.predicted_confidences.view(-1).cpu().numpy())
                ious.append(crit.achieved_ious.view(-1).cpu().numpy())
    model.min_confidence.copy_(keep_conf)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    iou = np.concatenate(ious) if ious else np.zeros(0)
    thr, f = tune_threshold(conf, iou) if len(conf) else (float(keep_conf), 0.0)
    model.min_confidence.fill_(thr)
    preds = predict(model, loader, device)
    s = score_paps.score(preds, chips_dir, class_names, None, extents=("chip",), workers=1)
    v = s["variants"]["chip/paps"]
    res = {"val/loss": sums[0] / n, "val/center_loss": sums[1] / n, "val/size_loss": sums[2] / n,
           "val/shape_loss": sums[3] / n, "val/class_loss": sums[4] / n, "val/min_confidence": thr,
           "val/tune_f": f, "val/tune_centres": int(len(conf)), "val/PQ": v["pq"], "val/SQ": v["sq"],
           "val/RQ": v["rq"], "val/PQ_agnostic": v["pq_agnostic"], "val/RQ_agnostic": v["rq_agnostic"],
           "val/predicted_parcels": v["predicted_parcels"]}
    log(f"  validation: loss {res['val/loss']:.4f}, threshold {thr:.2f} (F {f:.3f}, {len(conf)} centres), "
        f"PQ {v['pq']:.4f}, PQ agnostic {v['pq_agnostic']:.4f}, {v['predicted_parcels']} parcels")
    return res


def fit(cfg: dict, cell_dir: Path, raster: dict, chips: Path, split_dir: Path, worldcover: Path, *,
        num_workers: int, device: str | None = None, log=print) -> dict:
    """Train, validate and test one cell; write ``results.json`` and ``predictions_test.npz`` into ``cell_dir``."""
    from gfm4agri.benchmark.segmentation_data import (draw_support, load_manifest, manifest_sha256,
                                                      split_list, split_protocol)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cell_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    man = load_manifest(chips)
    class_names = [c["name"] for c in man["classes"]]
    lb = cfg["label_budget"]
    keep, support = draw_support(chips, lb["pct"], lb["draw_seed"], split_dir=split_dir)
    ids = {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}
    ids["train"] = train_chips(chips, ids["train"], keep)
    lab_dir = chips / "chips"
    common = dict(raster_dir=raster["dir"], suffix=raster["suffix"], means=raster["means"], stds=raster["stds"],
                  labels_dir=lab_dir, worldcover=worldcover)
    d, o, mcfg, lcfg = cfg["data"], cfg["optim"], cfg["model"], cfg["loss"]
    ds_train = PapsChips(ids["train"], keep=keep, augment=bool(d["augment"]), **common)
    ds_val = PapsChips(ids["val"], **common)
    ds_test = PapsChips(ids["test"], **common)
    log(f"support: {len(keep)} parcels at {lb['pct']} %; chips train {len(ds_train)}, val {len(ds_val)}, "
        f"test {len(ds_test)}")
    train_loader = _loader(ds_train, d["batch_size"], True, num_workers, drop_last=True)
    val_loader = _loader(ds_val, d["batch_size"], False, num_workers)

    model = build_model(len(raster["means"]), len(class_names), mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=o["lr"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(o["milestones"]), gamma=o["gamma"])
    crit = PaPsLoss(l_center=lcfg["l_center"], l_size=lcfg["l_size"], l_shape=lcfg["l_shape"],
                    l_class=lcfg["l_class"], beta=lcfg["beta"])
    epochs, warmup, val_every = int(cfg["epochs"]), int(o["warmup"]), int(o["val_every"])

    last, best_path = cell_dir / "last.pt", cell_dir / "best-pq.pt"
    start, best, history, fit_s = 1, {"pq": -1.0, "epoch": None}, [], 0.0
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        start, best, history, fit_s = st["epoch"] + 1, st["best"], st["history"], st["fit_s"]
        log(f"resumed after epoch {st['epoch']}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        heatmap_only = epoch - 1 < warmup
        model.train()
        sums, n = np.zeros(5), 0
        for batch in train_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            opt.zero_grad()
            p = model(x, zones=y[..., 2], pseudo_nms=False, heatmap_only=heatmap_only)
            loss = crit(p, y, heatmap_only=heatmap_only)
            loss.backward()
            opt.step()
            sums += [float(loss), *crit.value]
            n += 1
        sched.step()
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "heatmap_only": heatmap_only,
               "train/loss": sums[0] / n, "train/center_loss": sums[1] / n, "train/size_loss": sums[2] / n,
               "train/shape_loss": sums[3] / n, "train/class_loss": sums[4] / n, "train_s": time.time() - t0}
        log(f"epoch {epoch}/{epochs}: loss {row['train/loss']:.4f} (centre {row['train/center_loss']:.4f}, "
            f"size {row['train/size_loss']:.4f}, shape {row['train/shape_loss']:.4f}, "
            f"class {row['train/class_loss']:.4f}), {row['train_s']:.0f} s")
        if epoch > warmup and (epoch % val_every == 0 or epoch == epochs):
            tv = time.time()
            row.update(validate(model, crit, val_loader, device, lab_dir, class_names, log=log))
            row["val_s"] = time.time() - tv
            if row["val/PQ"] > best["pq"]:
                best = {"pq": row["val/PQ"], "epoch": epoch, "min_confidence": row["val/min_confidence"]}
                torch.save({"model": model.state_dict(), "epoch": epoch}, best_path)
        fit_s += time.time() - t0
        history.append(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "epoch": epoch, "best": best, "history": history,
                    "fit_s": fit_s}, last)
        _write_history(cell_dir / "metrics.csv", history)

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    st = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model"])
    log(f"test: checkpoint of epoch {st['epoch']}, threshold {float(model.min_confidence):.2f}")
    tt = time.time()
    preds = predict(model, _loader(ds_test, d["batch_size"], False, num_workers), device)
    keys = sorted(preds)
    tmp = cell_dir / "predictions_test.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, ids=np.array(keys), inst=np.stack([preds[k][0] for k in keys]),
                            cls=np.stack([preds[k][1] for k in keys]),
                            heatmap=np.stack([preds[k][2] for k in keys]),
                            saliency=np.stack([preds[k][3] for k in keys]),
                            mask=np.stack([preds[k][4] for k in keys]))
    tmp.replace(cell_dir / "predictions_test.npz")
    test_s = time.time() - tt

    results = {
        "run": cfg["run_name"], "workflow": cfg["workflow"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg, "country": man["country"], "year": man["year"], "dataset": man["dataset"],
        "manifest_sha256": manifest_sha256(chips), "split_protocol": split_protocol(split_dir, False),
        "label_budget": {**lb, "support": support}, "seed": seed,
        "chips": {k: len(v) for k, v in ids.items()}, "params": {"trainable": int(n_params)},
        "raster": raster["record"], "raster_dir": str(raster["dir"]),
        "best_epoch": st["epoch"], "best_val_pq": best["pq"], "min_confidence": float(model.min_confidence),
        "fit_seconds": round(fit_s, 1), "test_seconds": round(test_s, 1), "peak_gpu_gb": round(peak, 2),
        "class_names": class_names, "history": history,
    }
    (cell_dir / "results.json").write_text(json.dumps(results, indent=2))
    last.unlink(missing_ok=True)
    return results


def _write_history(path: Path, history: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in history for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)
