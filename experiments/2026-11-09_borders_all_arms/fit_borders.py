"""One cell of an arm with both heads: training, validation with the checkpoint, test predictions.

Adapted from ``../2026-11-07_prue_crop_tessera/fit_joint.py`` to every arm of the main workflow, under its K-shot
protocol: whole chips, the arm's effective batch (4), 45 epochs, the budget drawn by the core's ``draw_support``.

**Model** (:mod:`heads`): :class:`heads.UNetJoint` on a 10 m input (``source["kind"]`` ``raster`` or ``stack``) or
:class:`heads.TokenJoint` on cached tokens (``token``), built from the core's ``build_cached_model`` so the token
decoder is the main workflow's.

**Optimiser.** AdamW with one parameter group per part, each with its own settings (``cfg["optim"]``): PRUE's for
its U-Net (amsgrad, 3e-3, weight decay 0.01, cosine to 1e-6 over the run, per step); the main workflow arm's for the
token decoder (1e-4, weight decay 0.05, constant, as its TerraTorch task) and for the crop head (the arm's learning
rate and weight decay, on the same schedule as the part it sits on). The loss is PRUE's log-cosh Dice plus the crop
loss at the weight :class:`heads.Balance` keeps (``cfg["balance"]``). bf16 autocast on the GPU. A ``last.pt`` after
every epoch lets a killed job resume.

**Validation**, every epoch, whole chips: PRUE's fields scored as in the PRUE experiment (object F1 and PQ of every
variant over the whole chip) and the crop head's pixel Macro-F1. **The checkpoint** is the epoch of best
``(object F1 of the arm's own fields + crop pixel Macro-F1) / 2``, as in the joint experiment; the epochs each
criterion alone would choose are recorded beside it.

**Test**: ``predictions_test.npz`` with ``prob`` ``(N, 3, H, W)`` uint8 (the border classes, softmax x 255), ``pred``
``(N, H, W)`` uint8 (the crop class), ``mask`` and ``ids``, scored by ``score_joint``.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import heads
import prue
from chipsets import RasterChips, StackChips, TokenChips, collate, train_chips


def _loader(ds, batch, shuffle, workers, drop_last=False):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=False,
                      prefetch_factor=2 if workers else None)


def _macro_f1(cm: np.ndarray) -> float:
    from gfm4agri.benchmark.evaluation import per_class_scores

    return per_class_scores(cm)["macro_f1"]


def build_model(source: dict, n_classes: int, class_names: list[str], cfg: dict) -> torch.nn.Module:
    c = cfg["crop"]
    if source["kind"] in ("raster", "stack"):
        u = cfg["unet"]
        return heads.UNetJoint(source["channels"], n_classes, encoder=u["encoder"], weights=u.get("weights"),
                               hidden=c["hidden"], dropout=float(c["dropout"]),
                               head_dropout=float(c["head_dropout"]))
    from gfm4agri.benchmark.cached import build_cached_model

    base = build_cached_model(source["backbone"], num_classes=n_classes, class_names=class_names,
                              bands=source["bands"], n_timesteps=source["n_timesteps"],
                              channel_list=source["channel_list"], ignore_index=heads.IGNORE_CROP,
                              head_dropout=float(c["head_dropout"]))
    return heads.TokenJoint(base.neck, base.decoder, source["channel_list"], base.out_size, n_classes,
                            width=int(cfg["token"]["width"]), head_dropout=float(c["head_dropout"]))


def datasets(source: dict, ids: dict, keep, lab_dir: Path, worldcover: Path, augment: bool):
    common = dict(labels_dir=lab_dir, worldcover=worldcover)
    if source["kind"] == "raster":
        make = lambda chip_ids, **kw: RasterChips(chip_ids, source["dir"], source["suffix"], source["means"],  # noqa
                                                  source["stds"], **common, **kw)
    elif source["kind"] == "stack":
        make = lambda chip_ids, **kw: StackChips(chip_ids, lab_dir, source["s2_bands"], source["s2_means"],  # noqa
                                                 source["s2_stds"], source["s1_means"], source["s1_stds"],
                                                 source["n_months"], **common, **kw)
    else:
        make = lambda chip_ids, **kw: TokenChips(chip_ids, source["cache_dir"], len(source["channel_list"]),  # noqa
                                                 **common, **kw)
    return (make(ids["train"], keep=keep, augment=augment), make(ids["val"]), make(ids["test"]))


@torch.no_grad()
def predict(net, loader, device, weights, n_classes: int) -> dict:
    """Border ``prob`` uint8 ``(N, 3, H, W)``, crop ``pred`` uint8, ``mask``, ids, both mean losses, crop confusion."""
    net.eval()
    probs, preds, masks, ids, lb, lc = [], [], [], [], [], []
    cm = np.zeros((n_classes, n_classes), np.int64)
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            seg, crop = net(x)
        seg, crop = seg.float(), crop.float()
        y = batch["y"].to(device, non_blocking=True)
        yc = batch["yc"].to(device, non_blocking=True)
        lb.append(float(prue.log_cosh_dice(seg, y, weights)))
        lc.append(float(heads.crop_ce(crop, yc)))
        probs.append(torch.round(torch.softmax(seg, 1) * 255).to(torch.uint8).cpu().numpy())
        p = crop.argmax(1)
        preds.append(p.to(torch.uint8).cpu().numpy())
        sel = yc >= 0
        cm += np.bincount((yc[sel] * n_classes + p[sel]).cpu().numpy(), minlength=n_classes ** 2
                          ).reshape(n_classes, n_classes)
        masks.append(batch["mask"].numpy())
        ids += batch["chip"]
    return {"prob": np.concatenate(probs), "pred": np.concatenate(preds), "mask": np.concatenate(masks), "ids": ids,
            "loss_border": float(np.mean(lb)), "loss_crop": float(np.mean(lc)), "crop_confusion": cm}


def validate(net, loader, device, weights, chips_dir, class_names, worldcover, workers, log=print) -> dict:
    import score_prue

    p = predict(net, loader, device, weights, len(class_names))
    s = score_prue.score(p["prob"], p["mask"], p["ids"], chips_dir, class_names, worldcover, extents=("chip",),
                         workers=workers)
    res = {"val/loss_border": p["loss_border"], "val/loss_crop": p["loss_crop"],
           "val/crop_macro_f1": _macro_f1(p["crop_confusion"])}
    for v in prue.VARIANTS:
        q = s["variants"][f"chip/{v}"]
        res[f"val/{v}/PQ_agnostic"], res[f"val/{v}/RQ_agnostic"] = q["pq_agnostic"], q["rq_agnostic"]
        res[f"val/{v}/SQ_agnostic"], res[f"val/{v}/parcels"] = q["sq_agnostic"], q["predicted_parcels"]
    pxm = s["pixels"]
    res["val/boundary_f1"] = pxm["boundary_argmax"]["f1"]
    res["val/field_iou"] = pxm["field_iou"]
    for c in prue.CLASSES:
        res[f"val/iou_{c}"] = pxm["classes"][c]["iou"]
    res["val/joint"] = 0.5 * (res["val/cc/RQ_agnostic"] + res["val/crop_macro_f1"])
    log(f"  validation: crop Macro-F1 {res['val/crop_macro_f1']:.4f} (loss {p['loss_crop']:.4f}), object F1 (cc) "
        f"{res['val/cc/RQ_agnostic']:.4f}, PQ agnostic grow {res['val/grow/PQ_agnostic']:.4f}, boundary F1 "
        f"{res['val/boundary_f1']:.3f} (loss {p['loss_border']:.5f}); joint {res['val/joint']:.4f}")
    return res


def _schedule(kind: str, total: int, floor: float):
    if kind == "constant":
        return lambda step: 1.0
    if kind == "cosine":
        return lambda step: floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))
    raise ValueError(f"schedule {kind!r}")


def fit(cfg: dict, cell_dir: Path, source: dict, chips: Path, split_dir: Path, worldcover: Path, *,
        num_workers: int, device: str | None = None, log=print) -> dict:
    """Train, validate and test one cell; write ``results.json`` and ``predictions_test.npz`` into ``cell_dir``."""
    from gfm4agri.benchmark.segmentation_data import draw_support, load_manifest, manifest_sha256, split_list, \
        split_protocol

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
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
    d = cfg["data"]
    weights = tuple(cfg["loss"]["class_weights"])
    ds_train, ds_val, ds_test = datasets(source, ids, keep, lab_dir, worldcover, bool(d["augment"]))
    workers = int(d.get("num_workers", num_workers))
    bs, ebs = int(d["batch_size"]), int(d.get("eval_batch_size", d["batch_size"]))
    train_loader = _loader(ds_train, bs, True, workers, drop_last=True)
    val_loader = _loader(ds_val, ebs, False, workers)
    log(f"support: {len(keep)} parcels at {lb['pct']} %; chips train {len(ds_train)}, val {len(ds_val)}, "
        f"test {len(ds_test)}; input {source['kind']}; batch {bs}")

    net = build_model(source, len(class_names), class_names, cfg).to(device)
    groups = net.groups()
    epochs, per_epoch = int(cfg["epochs"]), len(train_loader)
    total = epochs * per_epoch
    params, lambdas = [], []
    for name, ps in groups.items():
        o = cfg["optim"][name]
        params.append({"params": [p for p in ps if p.requires_grad], "lr": float(o["lr"]),
                       "weight_decay": float(o["weight_decay"]), "amsgrad": bool(o.get("amsgrad", False)),
                       "name": name})
        lambdas.append(_schedule(o.get("schedule", "constant"), total,
                                 float(o.get("min_lr", o["lr"])) / float(o["lr"])))
    opt = torch.optim.AdamW(params)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)
    shared = net.shared_parameters()
    b = cfg["balance"]
    balance = heads.Balance(target=float(b["target"]), every=int(b["every"]), smooth=float(b.get("smooth", 0.5)))
    n_params = {k: sum(p.numel() for p in v) for k, v in groups.items()}
    n_params["shared"] = sum(p.numel() for p in shared)

    last, best_path = cell_dir / "last.pt", cell_dir / "best.pt"
    start, history, fit_s, gstep = 1, [], 0.0, 0
    best = {"joint": -1.0, "epoch": None}
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        balance.load_state_dict(st["balance"])
        torch.set_rng_state(st["rng"])
        start, best, history, fit_s, gstep = st["epoch"] + 1, st["best"], st["history"], st["fit_s"], st["gstep"]
        log(f"resumed after epoch {st['epoch']}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log("parameters: " + ", ".join(f"{k} {v / 1e6:.2f} M" for k, v in n_params.items())
        + f"; {per_epoch} steps per epoch, {epochs} epochs")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        net.train()
        sums, n, t_log = np.zeros(3), 0, time.time()
        for step, batch in enumerate(train_loader):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            yc = batch["yc"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                seg, crop = net(x)
            l_b = prue.log_cosh_dice(seg, y, weights)
            l_c = heads.crop_ce(crop, yc)
            if balance.due(gstep):
                balance.update(*heads.shared_grad_norms(shared, l_b, l_c))
            w = balance.weight if balance.weight is not None else 0.0
            loss = l_b + w * l_c
            loss.backward()
            opt.step()
            sched.step()
            gstep += 1
            sums += (loss.item(), l_b.item(), l_c.item())
            n += 1
            if epoch == start and step in (4, 19, 99):
                log(f"  epoch {epoch} step {step + 1}/{per_epoch}: {(time.time() - t_log) / (step + 1):.2f} s per "
                    f"step, peak {torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0:.1f} GB; "
                    f"crop weight {w:.4g}")
        sums /= max(n, 1)
        row = {"epoch": epoch, **{f"lr/{g['name']}": g["lr"] for g in opt.param_groups},
               "train/loss": sums[0], "train/loss_border": sums[1], "train/loss_crop": sums[2],
               **balance.epoch_summary(), "train_s": time.time() - t0}
        pull = row.get("balance/pull")
        log(f"epoch {epoch}/{epochs}: loss {sums[0]:.4f} (border {sums[1]:.5f}, crop {sums[2]:.4f}); crop weight "
            f"{row['balance/crop_weight']:.4g}, pull {pull if pull is None else round(pull, 3)}; "
            f"{row['train_s'] / 60:.1f} min")
        tv = time.time()
        row.update(validate(net, val_loader, device, weights, lab_dir, class_names, worldcover, min(workers, 16),
                            log=log))
        row["val_s"] = time.time() - tv
        if row["val/joint"] > best["joint"]:
            best = {"joint": row["val/joint"], "object_f1": row["val/cc/RQ_agnostic"],
                    "crop_macro_f1": row["val/crop_macro_f1"], "epoch": epoch}
            torch.save({"net": net.state_dict(), "epoch": epoch}, best_path)
        fit_s += time.time() - t0
        history.append(row)
        tmp = last.with_suffix(".pt.tmp")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "balance": balance.state_dict(), "rng": torch.get_rng_state(), "epoch": epoch, "best": best,
                    "history": history, "fit_s": fit_s, "gstep": gstep}, tmp)
        tmp.replace(last)

    keys = sorted({k for r in history for k in r}, key=lambda k: (k != "epoch", k))
    with open(cell_dir / "metrics.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(history)
    alone = {"object_f1": max(history, key=lambda r: r["val/cc/RQ_agnostic"])["epoch"],
             "crop_loss": min(history, key=lambda r: r["val/loss_crop"])["epoch"],
             "crop_macro_f1": max(history, key=lambda r: r["val/crop_macro_f1"])["epoch"],
             "pq_agnostic_grow": max(history, key=lambda r: r["val/grow/PQ_agnostic"])["epoch"]}

    net.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["net"])
    test_loader = _loader(ds_test, ebs, False, workers)
    t1 = time.time()
    p = predict(net, test_loader, device, weights, len(class_names))
    np.savez_compressed(cell_dir / "predictions_test.npz", prob=p["prob"], pred=p["pred"], mask=p["mask"],
                        ids=np.array(p["ids"]))
    log(f"test: {len(p['ids'])} chips predicted, crop Macro-F1 {_macro_f1(p['crop_confusion']):.4f}, losses border "
        f"{p['loss_border']:.5f} / crop {p['loss_crop']:.4f}, {time.time() - t1:.0f} s")

    results = {
        "run": cfg["run_name"], "workflow": cfg["workflow"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg, "country": man["country"], "year": man["year"],
        "chip_manifest_sha256": manifest_sha256(chips), "split": split_protocol(split_dir, False),
        "label_budget": lb, "seed": seed, "support": support, "class_names": class_names,
        "classes": list(prue.CLASSES), "n_chips": {k: len(v) for k, v in
                                                   (("train", ds_train), ("val", ds_val), ("test", ds_test))},
        "params": n_params, "input": {k: v for k, v in source.items() if k not in ("means", "stds", "s2_means",
                                                                                  "s2_stds", "s1_means", "s1_stds")},
        "checkpoint_rule": "best validation (object F1 of the arm's own fields + crop pixel Macro-F1) / 2",
        "best_epoch": best["epoch"], "best_val_joint": best["joint"], "best_val_object_f1": best.get("object_f1"),
        "best_val_crop_macro_f1": best.get("crop_macro_f1"), "epoch_by_single_criterion": alone,
        "final_crop_weight": balance.weight,
        "test_loss_border": p["loss_border"], "test_loss_crop": p["loss_crop"], "fit_seconds": fit_s,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        "history": history,
    }
    (cell_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    last.unlink(missing_ok=True)
    return results
