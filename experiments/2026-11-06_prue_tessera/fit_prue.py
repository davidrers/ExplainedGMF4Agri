"""One PRUE cell on TESSERA: training, validation with the checkpoint, test predictions.

**Model.** :func:`prue.build_unet` (smp U-Net, EfficientNet-B7, ImageNet) on TESSERA v2 Medium's 128 dimensions at
10 m. ``encoder: frozen`` reads the cached raster; ``encoder: finetuned`` runs the pretrained student on the chip's
composites in every step and trains it with the U-Net, end to end, as a second AdamW group at ``encoder_lr`` after a
linear warm-up and constant after (the convention of ``../2026-10-15_kshot_finetune``), the student held in eval mode
(it has no dropout or batch norm in use). Its output is standardised with fixed statistics
(:func:`tessera_ft.sample_stats`), so the U-Net sees the same scale as on the frozen raster.

**Training** (PRUE's recipe where the paper or ftw-prue states it, RESEARCH.md): log-cosh Dice with the class
weights ``[0.05, 0.20, 0.75]``; AdamW with amsgrad (ftw-prue's ``configure_optimizers``); the U-Net at ``lr``, decayed
by a cosine to ``min_lr`` over the run, per step; D4 augmentation (FTW's rotations and flips; its brightness and
channel shuffle are for raw reflectance and date order, which an embedding does not have); bf16 autocast; batches of
``batch_size`` chips, each a random ``crop`` window when set. A ``last.pt`` after every epoch lets a killed job
resume.

**Validation**, every ``val_every`` epochs and at the last: the class probabilities of every validation chip, whole;
the validation loss; the fields of every variant over the whole chip, scored by the void rule
(``score_prue.score``, extent ``chip``, class-agnostic). The checkpoint is the epoch of best validation object F1
(class-agnostic RQ at IoU > 0.5) of PRUE's own fields (``cc``), as ftw-prue monitors ``val/object_f1``.

**Test**: the kept model predicts every test chip, whole: ``predictions_test.npz`` holds ``prob`` ``(N, 3, H, W)``
uint8 (softmax x 255), ``mask`` and ``ids``.
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

import prue
from data import (CompositeChips, RasterChips, collate_raster, make_collate_composites, train_chips)


def _loader(ds, batch, shuffle, workers, collate, drop_last=False):
    from torch.utils.data import DataLoader

    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      collate_fn=collate, pin_memory=True, persistent_workers=False,
                      prefetch_factor=2 if workers else None)


class Net(torch.nn.Module):
    """The U-Net on the TESSERA embedding: from the raster's batch, or from the student on the composites."""

    def __init__(self, unet, means, stds, student=None, trainable: bool = False, chunk: int = 16384):
        super().__init__()
        self.unet, self.student, self.trainable, self.chunk = unet, student, bool(trainable), int(chunk)
        self.register_buffer("emb_mean", torch.as_tensor(means, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("emb_std", torch.as_tensor(stds, dtype=torch.float32).view(1, -1, 1, 1))
        if student is not None:
            for p in student.parameters():
                p.requires_grad_(self.trainable)

    def embed(self, batch, device) -> torch.Tensor:
        if "image" in batch:
            return batch["image"].to(device, non_blocking=True)             # already standardised
        import tessera_ft

        e = tessera_ft.encode(self.student, batch["prep"], grad=self.trainable and torch.is_grad_enabled(),
                              chunk=self.chunk)
        h, w = batch["hw"]
        e = e.view(batch["n"], h, w, -1).permute(0, 3, 1, 2)
        return (e - self.emb_mean) / self.emb_std

    def forward(self, batch, device) -> torch.Tensor:
        x = self.embed(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return self.unet(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.student is not None:
            self.student.eval()
        return self


@torch.no_grad()
def predict(net, loader, device, weights) -> tuple[np.ndarray, np.ndarray, list, float]:
    """``prob`` uint8 ``(N, 3, H, W)``, ``mask`` int16, ids, and the mean loss over the batches."""
    net.eval()
    probs, masks, ids, losses = [], [], [], []
    for batch in loader:
        logits = net(batch, device).float()
        y = batch["y"].to(device, non_blocking=True)
        losses.append(float(prue.log_cosh_dice(logits, y, weights)))
        probs.append(torch.round(torch.softmax(logits, 1) * 255).to(torch.uint8).cpu().numpy())
        masks.append(batch["mask"].numpy())
        ids += batch["chip"]
    return np.concatenate(probs), np.concatenate(masks), ids, float(np.mean(losses))


def validate(net, loader, device, weights, chips_dir, class_names, worldcover, workers, log=print) -> dict:
    import score_prue

    prob, mask, ids, loss = predict(net, loader, device, weights)
    s = score_prue.score(prob, mask, ids, chips_dir, class_names, worldcover, extents=("chip",), workers=workers)
    res = {"val/loss": loss}
    for v in prue.VARIANTS:
        q = s["variants"][f"chip/{v}"]
        res[f"val/{v}/PQ_agnostic"], res[f"val/{v}/RQ_agnostic"] = q["pq_agnostic"], q["rq_agnostic"]
        res[f"val/{v}/SQ_agnostic"], res[f"val/{v}/parcels"] = q["sq_agnostic"], q["predicted_parcels"]
    pxm = s["pixels"]
    res["val/boundary_f1"] = pxm["boundary_argmax"]["f1"]
    res["val/field_iou"] = pxm["field_iou"]
    for c in prue.CLASSES:
        res[f"val/iou_{c}"] = pxm["classes"][c]["iou"]
    log(f"  validation: loss {loss:.4f}, object F1 (cc) {res['val/cc/RQ_agnostic']:.4f}, PQ agnostic cc "
        f"{res['val/cc/PQ_agnostic']:.4f} / grow {res['val/grow/PQ_agnostic']:.4f} / field "
        f"{res['val/field/PQ_agnostic']:.4f}, boundary F1 {res['val/boundary_f1']:.3f}, field IoU "
        f"{res['val/field_iou']:.3f}, {res['val/cc/parcels']} fields")
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
    d, m, o = cfg["data"], cfg["model"], cfg["optim"]
    weights = tuple(cfg["loss"]["class_weights"])
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
        f"test {len(ds_test)}; encoder {cfg['encoder']}; crop {d.get('crop')}")
    workers = int(d.get("num_workers", num_workers))
    bs, ebs = int(d["batch_size"]), int(d.get("eval_batch_size", d["batch_size"]))
    train_loader = _loader(ds_train, bs, True, workers, collate, drop_last=True)
    val_loader = _loader(ds_val, ebs, False, workers, collate)

    unet = prue.build_unet(len(means), m["unet_encoder"], m.get("unet_weights"))
    net = Net(unet, means, stds, student=student, trainable=finetuned and bool(m.get("trainable", True)),
              chunk=int(m.get("encoder_chunk", 16384))).to(device)
    n_unet = sum(p.numel() for p in unet.parameters())
    n_enc = sum(p.numel() for p in student.parameters() if p.requires_grad) if student is not None else 0
    groups = [{"params": list(unet.parameters()), "lr": float(o["lr"]), "name": "unet"}]
    if n_enc:
        groups.append({"params": [p for p in student.parameters() if p.requires_grad],
                       "lr": float(m["encoder_lr"]), "name": "encoder"})
    opt = torch.optim.AdamW(groups, weight_decay=float(o["weight_decay"]), amsgrad=True)
    epochs, val_every = int(cfg["epochs"]), int(o["val_every"])
    per_epoch = len(train_loader)
    total = epochs * per_epoch
    floor = float(o["min_lr"]) / float(o["lr"])
    warm = max(1, round(float(m.get("encoder_warmup_epochs", 1)) * per_epoch))

    def unet_lr(step):
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))

    lambdas = [unet_lr] + ([lambda s: min(1.0, (s + 1) / warm)] if n_enc else [])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambdas)

    last, best_path = cell_dir / "last.pt", cell_dir / "best.pt"
    start, best, history, fit_s = 1, {"f1": -1.0, "epoch": None}, [], 0.0
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
    log(f"parameters: U-Net {n_unet / 1e6:.1f} M, encoder trainable {n_enc / 1e6:.2f} M; {per_epoch} steps per "
        f"epoch, {epochs} epochs")

    for epoch in range(start, epochs + 1):
        t0 = time.time()
        net.train()
        loss_sum, n, t_log = 0.0, 0, time.time()
        for step, batch in enumerate(train_loader):
            y = batch["y"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = net(batch, device)
            loss = prue.log_cosh_dice(logits, y, weights)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += float(loss)
            n += 1
            if epoch == start and step in (4, 19, 99):
                log(f"  epoch {epoch} step {step + 1}/{per_epoch}: {(time.time() - t_log) / (step + 1):.2f} s per "
                    f"step, peak {torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else 0:.1f} GB")
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"],
               "encoder_lr": opt.param_groups[1]["lr"] if n_enc else 0.0,
               "train/loss": loss_sum / max(n, 1), "train_s": time.time() - t0}
        log(f"epoch {epoch}/{epochs}: loss {row['train/loss']:.4f}, {row['train_s'] / 60:.1f} min")
        if epoch % val_every == 0 or epoch == epochs:
            tv = time.time()
            row.update(validate(net, val_loader, device, weights, lab_dir, class_names, worldcover, min(workers, 16),
                                log=log))
            row["val_s"] = time.time() - tv
            if row["val/cc/RQ_agnostic"] > best["f1"]:
                best = {"f1": row["val/cc/RQ_agnostic"], "pq": row["val/cc/PQ_agnostic"], "epoch": epoch}
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

    net.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["net"])
    test_loader = _loader(ds_test, ebs, False, workers, collate)
    t1 = time.time()
    prob, mask, tids, test_loss = predict(net, test_loader, device, weights)
    np.savez_compressed(cell_dir / "predictions_test.npz", prob=prob, mask=mask, ids=np.array(tids))
    log(f"test: {len(tids)} chips predicted, loss {test_loss:.4f}, {time.time() - t1:.0f} s")

    results = {
        "run": cfg["run_name"], "workflow": cfg["workflow"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg, "country": man["country"], "year": man["year"],
        "chip_manifest_sha256": manifest_sha256(chips), "split": split_protocol(split_dir, False),
        "label_budget": lb, "seed": seed, "support": support, "class_names": class_names,
        "classes": list(prue.CLASSES), "n_chips": {k: len(v) for k, v in
                                                   (("train", ds_train), ("val", ds_val), ("test", ds_test))},
        "params": {"unet": n_unet, "encoder_trainable": n_enc},
        "embedding_stats": stats["computed_on"] if stats else (raster or {}).get("record", {}).get("created"),
        "best_epoch": best["epoch"], "best_val_object_f1": best["f1"], "best_val_pq_agnostic": best.get("pq"),
        "test_loss": test_loss, "fit_seconds": fit_s,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0,
        "history": history,
    }
    (cell_dir / "results.json").write_text(json.dumps(results, indent=2))
    last.unlink(missing_ok=True)
    return results
