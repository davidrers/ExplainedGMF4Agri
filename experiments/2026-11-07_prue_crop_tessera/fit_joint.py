"""One cell of the joint model on TESSERA: training, validation with the checkpoint, test predictions.

Adapted from ``../2026-11-06_prue_tessera/fit_prue.py``: the same data, encoder routes, border target, loss, optimiser
and schedule for the U-Net, with the crop-type head of :mod:`joint` added and trained in the same steps.

**Model.** :class:`joint.JointNet`: PRUE's U-Net (smp, EfficientNet-B7, ImageNet) on TESSERA v2 Medium's 128
dimensions, its three-class border head, and the main workflow's crop MLP on the embedding and the U-Net's last
decoder features. ``encoder: frozen`` reads the cached raster; ``encoder: finetuned`` runs the pretrained student on
the chip's composites in every step and trains it with both heads, as PRUE's fine-tuned arm.

**Training.** The loss is PRUE's log-cosh Dice plus ``crop.loss_weight`` times the crop head's cross-entropy
(:func:`joint.losses`). Three AdamW groups: the U-Net with PRUE's settings (amsgrad, ``optim.lr``, weight decay
``optim.weight_decay``), the crop head with the main workflow's (``crop.lr``, ``crop.weight_decay``, no amsgrad, as
TerraTorch's AdamW), both on one cosine over the run, per step, to ``min_lr / lr`` of their rate; and, fine-tuned,
the encoder at ``encoder_lr`` after a linear warm-up, constant after. PRUE's D4, 128 x 128 windows, batches of 8,
bf16. Once per epoch, on its first batch, the norms of the two losses' gradients over the U-Net's parameters are
logged (:func:`joint.shared_grad_norms`), so the balance between the tasks is seen through the run. A ``last.pt``
after every epoch lets a killed job resume.

**Validation**, every epoch, whole chips: PRUE's fields scored as in the PRUE experiment (object F1 and PQ of every
variant over the whole chip), and the crop head's pixel Macro-F1 over the labelled pixels, with both losses. **The
checkpoint** is the epoch of best ``(object F1 of PRUE's own fields + crop pixel Macro-F1) / 2``: one model, one
epoch, chosen on both tasks. The epochs each criterion alone would have chosen (PRUE's object F1; the main
workflow's validation loss, here the crop loss) are recorded beside it.

**Test**: the kept model predicts every test chip, whole: ``predictions_test.npz`` holds ``prob``
``(N, 3, H, W)`` uint8 (border classes, softmax x 255, as PRUE stored them), ``pred`` ``(N, H, W)`` uint8 (the crop
class, as the main workflow stores it), ``mask`` and ``ids``.
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

import joint
import prue
from data import CompositeChips, RasterChips, collate_raster, make_collate_composites, train_chips


def _loader(ds, batch, shuffle, workers, collate, drop_last=False):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=False,
                      prefetch_factor=2 if workers else None)


def _macro_f1(cm: np.ndarray) -> float:
    from gfm4agri.benchmark.evaluation import per_class_scores

    return per_class_scores(cm)["macro_f1"]


@torch.no_grad()
def predict(net, loader, device, weights, n_classes: int) -> dict:
    """Border ``prob`` uint8 ``(N, 3, H, W)``, crop ``pred`` uint8, ``mask``, ids, both mean losses, crop confusion."""
    net.eval()
    probs, preds, masks, ids, lb, lc = [], [], [], [], [], []
    cm = np.zeros((n_classes, n_classes), np.int64)
    for batch in loader:
        seg, crop = net(batch, device)
        seg, crop = seg.float(), crop.float()
        y = batch["y"].to(device, non_blocking=True)
        yc = batch["yc"].to(device, non_blocking=True)
        lb.append(float(prue.log_cosh_dice(seg, y, weights)))
        lc.append(float(joint.crop_ce(crop, yc)))
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


def validate(net, loader, device, weights, crop_weight, chips_dir, class_names, worldcover, workers,
             log=print) -> dict:
    import score_prue

    p = predict(net, loader, device, weights, len(class_names))
    s = score_prue.score(p["prob"], p["mask"], p["ids"], chips_dir, class_names, worldcover, extents=("chip",),
                         workers=workers)
    res = {"val/loss": p["loss_border"] + crop_weight * p["loss_crop"], "val/loss_border": p["loss_border"],
           "val/loss_crop": p["loss_crop"], "val/crop_macro_f1": _macro_f1(p["crop_confusion"])}
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


def fit(cfg: dict, cell_dir: Path, raster: dict | None, chips: Path, split_dir: Path, worldcover: Path, *,
        num_workers: int, device: str | None = None, log=print) -> dict:
    """Train, validate and test one cell; write ``results.json`` and ``predictions_test.npz`` into ``cell_dir``."""
    import tessera_ft
    from gfm4agri.benchmark.segmentation_data import draw_support, load_manifest, manifest_sha256, split_list, \
        split_protocol
    from gfm4agri.embeddings.tessera_v2 import month_doy

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
    d, m, o, c = cfg["data"], cfg["model"], cfg["optim"], cfg["crop"]
    weights = tuple(cfg["loss"]["class_weights"])
    crop_weight = float(c["loss_weight"])
    finetuned = cfg["encoder"] == "finetuned"

    student, stats = None, None
    if finetuned:
        student = tessera_ft.load_student().to(device).eval()
        st_path = cell_dir / "embedding_stats.json"
        if st_path.exists():
            stats = json.loads(st_path.read_text())
        else:
            t0 = time.time()
            stats = tessera_ft.sample_stats(student, lab_dir, ids["train"], man["imagery"]["band_names"],
                                            man["imagery"]["n_months"], man["year"],
                                            n_chips=int(d.get("stats_chips", 128)))
            st_path.write_text(json.dumps(stats, indent=1))
            log(f"standardisation of the student's output from {stats['n_chips']} training chips, "
                f"{time.time() - t0:.0f} s")
        means, stds = stats["means"], stats["stds"]
        common = dict(chips_dir=lab_dir, band_names=man["imagery"]["band_names"],
                      n_months=man["imagery"]["n_months"], labels_dir=lab_dir, worldcover=worldcover)
        Chips, collate = CompositeChips, make_collate_composites(month_doy(man["year"], man["imagery"]["n_months"]))
    else:
        means, stds = raster["means"], raster["stds"]
        common = dict(raster_dir=raster["dir"], suffix=raster["suffix"], means=means, stds=stds,
                      labels_dir=lab_dir, worldcover=worldcover)
        Chips, collate = RasterChips, collate_raster
    ds_train = Chips(ids["train"], keep=keep, augment=bool(d["augment"]), crop=d.get("crop"), **common)
    ds_val, ds_test = Chips(ids["val"], **common), Chips(ids["test"], **common)
    log(f"support: {len(keep)} parcels at {lb['pct']} %; chips train {len(ds_train)}, val {len(ds_val)}, "
        f"test {len(ds_test)}; encoder {cfg['encoder']}; crop {d.get('crop')}; crop loss weight {crop_weight:g}")
    workers = int(d.get("num_workers", num_workers))
    bs, ebs = int(d["batch_size"]), int(d.get("eval_batch_size", d["batch_size"]))
    train_loader = _loader(ds_train, bs, True, workers, collate, drop_last=True)
    val_loader = _loader(ds_val, ebs, False, workers, collate)

    unet = prue.build_unet(len(means), m["unet_encoder"], m.get("unet_weights"))
    head = joint.CropHead(len(means) + joint.decoder_channels(unet), len(class_names), hidden=c["hidden"],
                          dropout=float(c["dropout"]), head_dropout=float(c["head_dropout"]))
    net = joint.JointNet(unet, head, means, stds, student=student,
                         trainable=finetuned and bool(m.get("trainable", True)),
                         chunk=int(m.get("encoder_chunk", 16384))).to(device)
    n_unet = sum(p.numel() for p in unet.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    n_enc = sum(p.numel() for p in student.parameters() if p.requires_grad) if student is not None else 0
    groups = [{"params": list(unet.parameters()), "lr": float(o["lr"]), "weight_decay": float(o["weight_decay"]),
               "amsgrad": True, "name": "unet"},
              {"params": list(head.parameters()), "lr": float(c["lr"]), "weight_decay": float(c["weight_decay"]),
               "amsgrad": False, "name": "crop_head"}]
    if n_enc:
        groups.append({"params": [p for p in student.parameters() if p.requires_grad],
                       "lr": float(m["encoder_lr"]), "weight_decay": float(o["weight_decay"]), "amsgrad": True,
                       "name": "encoder"})
    opt = torch.optim.AdamW(groups)
    epochs, val_every = int(cfg["epochs"]), int(o["val_every"])
    per_epoch = len(train_loader)
    total = epochs * per_epoch
    floor = float(o["min_lr"]) / float(o["lr"])
    warm = max(1, round(float(m.get("encoder_warmup_epochs", 1)) * per_epoch))

    def cosine(step):
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))

    lambdas = [cosine, cosine] + ([lambda s: min(1.0, (s + 1) / warm)] if n_enc else [])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)

    last, best_path = cell_dir / "last.pt", cell_dir / "best.pt"
    start, history, fit_s = 1, [], 0.0
    best = {"joint": -1.0, "epoch": None}
    if last.exists():
        st = torch.load(last, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng"])
        start, best, history, fit_s = st["epoch"] + 1, st["best"], st["history"], st["fit_s"]
        log(f"resumed after epoch {st['epoch']}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log(f"parameters: U-Net {n_unet / 1e6:.1f} M, crop head {n_head / 1e6:.2f} M, encoder trainable "
        f"{n_enc / 1e6:.2f} M; {per_epoch} steps per epoch, {epochs} epochs")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        net.train()
        sums, n, t_log = np.zeros(3), 0, time.time()
        probe = None
        for step, batch in enumerate(train_loader):
            y = batch["y"].to(device, non_blocking=True)
            yc = batch["yc"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            seg, crop = net(batch, device)
            if step == 0:
                probe = joint.shared_grad_norms(net, seg, crop, y, yc, weights, retain=True)
            loss, l_b, l_c = joint.losses(seg, crop, y, yc, weights, crop_weight)
            loss.backward()
            opt.step()
            sched.step()
            sums += (loss.item(), l_b.item(), l_c.item())
            n += 1
            if epoch == start and step in (4, 19, 99):
                log(f"  epoch {epoch} step {step + 1}/{per_epoch}: {(time.time() - t_log) / (step + 1):.2f} s per "
                    f"step, peak {torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0:.1f} GB")
        sums /= max(n, 1)
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "crop_lr": opt.param_groups[1]["lr"],
               "encoder_lr": opt.param_groups[2]["lr"] if n_enc else 0.0,
               "train/loss": sums[0], "train/loss_border": sums[1], "train/loss_crop": sums[2],
               "grad_unet/border": probe[0], "grad_unet/crop": probe[1],
               "grad_unet/crop_weighted_over_border": crop_weight * probe[1] / max(probe[0], 1e-30),
               "train_s": time.time() - t0}
        log(f"epoch {epoch}/{epochs}: loss {sums[0]:.4f} (border {sums[1]:.5f}, crop {sums[2]:.4f}); pull on the "
            f"U-Net, crop (weighted) / border {row['grad_unet/crop_weighted_over_border']:.2f}; "
            f"{row['train_s'] / 60:.1f} min")
        if epoch % val_every == 0 or epoch == epochs:
            tv = time.time()
            row.update(validate(net, val_loader, device, weights, crop_weight, lab_dir, class_names, worldcover,
                                min(workers, 16), log=log))
            row["val_s"] = time.time() - tv
            if row["val/joint"] > best["joint"]:
                best = {"joint": row["val/joint"], "object_f1": row["val/cc/RQ_agnostic"],
                        "crop_macro_f1": row["val/crop_macro_f1"], "epoch": epoch}
                torch.save({"net": net.state_dict(), "epoch": epoch}, best_path)
        fit_s += time.time() - t0
        history.append(row)
        tmp = last.with_suffix(".pt.tmp")
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": torch.get_rng_state(), "epoch": epoch, "best": best, "history": history,
                    "fit_s": fit_s}, tmp)
        tmp.replace(last)

    keys = sorted({k for r in history for k in r}, key=lambda k: (k != "epoch", k))
    with open(cell_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    val_rows = [r for r in history if "val/joint" in r]
    alone = {"object_f1": max(val_rows, key=lambda r: r["val/cc/RQ_agnostic"])["epoch"],
             "crop_loss": min(val_rows, key=lambda r: r["val/loss_crop"])["epoch"],
             "crop_macro_f1": max(val_rows, key=lambda r: r["val/crop_macro_f1"])["epoch"]}

    net.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["net"])
    test_loader = _loader(ds_test, ebs, False, workers, collate)
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
        "params": {"unet": n_unet, "crop_head": n_head, "encoder_trainable": n_enc},
        "embedding_stats": stats["computed_on"] if stats else (raster or {}).get("record", {}).get("created"),
        "checkpoint_rule": "best validation (object F1 of PRUE's own fields + crop pixel Macro-F1) / 2",
        "best_epoch": best["epoch"], "best_val_joint": best["joint"], "best_val_object_f1": best.get("object_f1"),
        "best_val_crop_macro_f1": best.get("crop_macro_f1"), "epoch_by_single_criterion": alone,
        "test_loss_border": p["loss_border"], "test_loss_crop": p["loss_crop"], "fit_seconds": fit_s,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        "history": history,
    }
    (cell_dir / "results.json").write_text(json.dumps(results, indent=2))
    last.unlink(missing_ok=True)
    return results
