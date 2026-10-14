"""One K-shot cell of a TESSERA v2 arm: per-pixel fit, encoder frozen or trained, on the pixel store.

The label budget is the core's ``draw_support``. Each epoch draws up to ``pixels_per_parcel`` pixels of every
support parcel, a fresh subset seeded by (seed, epoch), the same in both modes. ``trainable: false`` trains the
head on the frozen encoder's cached embeddings; ``trainable: true`` passes the gradients through the encoder too.
In both modes the encoder stays in evaluation mode, so v2's one dropout (0.2, in its fusion MLP) is off in both and
the switch toggles gradients only. The checkpoint is the epoch of lowest validation loss; the test scores every
labelled test pixel with the core's ``confusion`` and ``per_class_scores``.

The cell is written in the core runner's layout: ``results.json``, ``logs/version_0/metrics.csv``,
``predictions_test.npz`` (labelled pixels only, -1 elsewhere) and ``checkpoints/best.pt``.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pixels import load_split
from tessera_v2 import encode_batch, load_student

VAL_SEED = 20261014


def head(n_classes: int, hidden=(512, 256), dropout: float = 0.2) -> nn.Sequential:
    """The TESSERA arm's per-pixel MLP: Linear, ReLU, batch norm, dropout per hidden layer, then the classifier."""
    layers, c = [], 128
    for h in hidden:
        layers += [nn.Linear(c, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
        c = h
    return nn.Sequential(*layers, nn.Linear(c, n_classes))


def sample_per_parcel(parcel: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of up to ``cap`` pixels per parcel, chosen at random, in random order."""
    if len(parcel) == 0:
        return np.zeros(0, dtype=np.int64)
    key = rng.random(len(parcel))
    order = np.lexsort((key, parcel))
    p = parcel[order]
    start = np.r_[0, np.flatnonzero(p[1:] != p[:-1]) + 1]
    rank = np.arange(len(p)) - np.repeat(start, np.diff(np.r_[start, len(p)]))
    keep = order[rank < cap]
    return keep[rng.permutation(len(keep))]


class Encoder:
    """The pixels' 128-d input to the head: from the frozen cache, or through the trainable encoder."""

    def __init__(self, cfg: dict, store: Path, cache: Path | None, weights: Path, device: torch.device) -> None:
        m, self.precision, self.device = cfg["model"], cfg["train"]["precision"], device
        self.trainable = bool(m["trainable"])
        model = load_student(m["student"], weights, m["checkpoint_sha256"])
        self.n_params = sum(p.numel() for p in model.parameters())
        if self.trainable:
            self.model = model.to(device).eval()
            self.data = {s: load_split(store, s, ("s2", "s1")) for s in ("train", "val", "test")}
        else:
            self.model = None
            self.cache = {s: np.load(cache / f"{s}.npy") for s in ("train", "val", "test")}

    def __call__(self, split: str, idx: np.ndarray, grad: bool) -> torch.Tensor:
        if not self.trainable:
            return torch.from_numpy(self.cache[split][idx].astype(np.float32)).to(self.device)
        d = self.data[split]
        with torch.set_grad_enabled(grad):
            return encode_batch(self.model, d["s2"][idx], d["s1"][idx], self.device, self.precision)


def _scores(cm: np.ndarray) -> dict:
    from gfm4agri.benchmark.evaluation import per_class_scores

    return per_class_scores(cm)


@torch.no_grad()
def evaluate(enc: Encoder, net: nn.Module, split: str, idx: np.ndarray, y: np.ndarray, n_classes: int,
             batch: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy, confusion matrix and predictions over ``idx`` of ``split``."""
    net.eval()
    loss_sum, pred = 0.0, np.empty(len(idx), dtype=np.int16)
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        logits = net(enc(split, j, grad=False))
        t = torch.from_numpy(y[j].astype(np.int64)).to(logits.device)
        loss_sum += float(nn.functional.cross_entropy(logits, t, reduction="sum"))
        pred[i:i + batch] = logits.argmax(1).cpu().numpy()
    from gfm4agri.benchmark.evaluation import confusion

    return loss_sum / max(len(idx), 1), confusion(pred, y[idx], n_classes), pred


def fit_cell(cfg: dict, *, chips: Path, split_dir: Path, store: Path, cache: Path | None, weights: Path,
             cell_dir: Path, pct: float, draw: int, seed: int, epochs: int, store_rec: dict) -> dict:
    from gfm4agri.benchmark.segmentation_data import (chips_with_parcels, draw_support, load_manifest,
                                                      manifest_sha256, split_protocol)

    t0 = time.time()
    device = torch.device("cuda")
    tr = cfg["train"]
    manifest = load_manifest(chips)
    names = [c["name"] for c in manifest["classes"]]
    n_classes = len(names)
    cell_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    # budget, and the pixels each split contributes
    support, report = draw_support(chips, pct, draw, "train", split_dir)
    lab = {s: load_split(store, s, ("y", "parcel", "chip", "pos")) for s in ("train", "val", "test")}
    train_pool = np.flatnonzero(np.isin(lab["train"]["parcel"], support))
    val_idx = np.sort(sample_per_parcel(lab["val"]["parcel"], tr["val_pixels_per_parcel"],
                                        np.random.default_rng(VAL_SEED)))
    test_idx = np.arange(len(lab["test"]["y"]))

    torch.manual_seed(seed)
    enc = Encoder(cfg, store, cache, weights, device)
    net = head(n_classes, cfg["model"]["head_hidden"], cfg["model"]["head_dropout"]).to(device)
    groups = [{"params": net.parameters(), "lr": tr["lr_head"], "name": "head"}]
    if enc.trainable:
        groups.append({"params": enc.model.parameters(), "lr": tr["lr_encoder"], "name": "encoder"})
    opt = torch.optim.AdamW(groups, weight_decay=tr["weight_decay"])
    batch = int(tr["batch_pixels"])
    steps_per_epoch = math.ceil(_expected(lab["train"]["parcel"][train_pool], tr["pixels_per_parcel"]) / batch)
    warmup = max(1, int(tr.get("warmup_epochs", 1) * steps_per_epoch))

    log_dir = cell_dir / "logs" / "version_0"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows, best, step = [], (math.inf, -1, None), 0
    for epoch in range(epochs):
        net.train()
        rng = np.random.default_rng([seed, epoch])
        order = train_pool[sample_per_parcel(lab["train"]["parcel"][train_pool], tr["pixels_per_parcel"], rng)]
        loss_sum, n_seen = 0.0, 0
        for i in range(0, len(order), batch):
            j = np.sort(order[i:i + batch])
            if len(j) < 2:          # batch norm needs two samples
                continue
            if enc.trainable:
                for g in opt.param_groups:
                    if g["name"] == "encoder":
                        g["lr"] = tr["lr_encoder"] * min(1.0, (step + 1) / warmup)
            logits = net(enc("train", j, grad=enc.trainable))
            t = torch.from_numpy(lab["train"]["y"][j].astype(np.int64)).to(device)
            loss = nn.functional.cross_entropy(logits, t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(j)
            n_seen += len(j)
            step += 1
        if enc.trainable:
            enc.model.eval()
        v_loss, v_cm, _ = evaluate(enc, net, "val", val_idx, lab["val"]["y"], n_classes, 4 * batch)
        v = _scores(v_cm)
        rows.append({"epoch": epoch, "step": step, "train/loss": loss_sum / max(n_seen, 1), "val/loss": v_loss,
                     "val/F1_Score": v["macro_f1"], "val/mIoU": v["miou"], "train/pixels": n_seen})
        print(f"    epoch {epoch:2d}: train loss {rows[-1]['train/loss']:.4f}, val loss {v_loss:.4f}, "
              f"val Macro-F1 {v['macro_f1']:.4f} ({n_seen:,} pixels, {time.time() - t0:.0f} s)", flush=True)
        if v_loss < best[0]:
            state = {"head": copy.deepcopy(net.state_dict())}
            if enc.trainable:
                state["encoder"] = copy.deepcopy(enc.model.state_dict())
            best = (v_loss, epoch, state)
    with open(log_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # test with the checkpoint of lowest validation loss
    net.load_state_dict(best[2]["head"])
    if enc.trainable:
        enc.model.load_state_dict(best[2]["encoder"])
    (cell_dir / "checkpoints").mkdir(exist_ok=True)
    torch.save(best[2], cell_dir / "checkpoints" / "best.pt")
    t_loss, t_cm, t_pred = evaluate(enc, net, "test", test_idx, lab["test"]["y"], n_classes, 4 * batch)
    s = _scores(t_cm)
    metrics = {"test/loss": t_loss, "test/F1_Score": s["macro_f1"], "test/mIoU": s["miou"],
               "test/Pixel_Accuracy": s["accuracy"]}
    for k, c in enumerate(names):
        metrics[f"test/IoU_{c}"] = float(s["iou"][k])
        metrics[f"test/F1_{c}"] = float(s["f1"][k])
    _write_predictions(cell_dir, store_rec["chips"]["test"], lab["test"], t_pred)

    train_chips = chips_with_parcels(chips, store_rec["chips"]["train"], support)
    result = {
        "run": cfg["_name"],
        "workflow": ("per-pixel: TESSERA v2 encoder trained end to end with the head"
                     if enc.trainable else "per-pixel: frozen TESSERA v2 embeddings cached once, head trained "
                     "from cache"),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "country": manifest["country"], "year": manifest["year"], "dataset": manifest["dataset"],
        "manifest_sha256": manifest_sha256(chips),
        "split_protocol": split_protocol(split_dir, test_is_val=False),
        "label_budget": {"mode": "pct", "pct": pct, "draw_seed": draw,
                         "labelled_pixel_fraction_train": round(len(train_pool) / max(len(lab["train"]["y"]), 1), 4),
                         "support": report},
        "seed": seed, "effective_batch": batch,
        "chips": {"train": len(train_chips), "val": len(store_rec["chips"]["val"]),
                  "test": len(store_rec["chips"]["test"])},
        "pixels": {"train_pool": int(len(train_pool)), "val": int(len(val_idx)), "test": int(len(test_idx)),
                   "pixels_per_parcel": tr["pixels_per_parcel"], "steps": step},
        "backbone": cfg["model"]["backbone"], "frozen_encoder": not enc.trainable,
        "params": {"encoder_total": enc.n_params, "encoder_trainable": enc.n_params if enc.trainable else 0,
                   "head_total": sum(p.numel() for p in net.parameters()),
                   "head_trainable": sum(p.numel() for p in net.parameters())},
        "best_checkpoint": str(cell_dir / "checkpoints" / "best.pt"), "best_epoch": best[1],
        "fit_seconds": round(time.time() - t0, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "test_metrics": metrics, "class_names": names,
    }
    (cell_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _expected(parcel: np.ndarray, cap: int) -> int:
    """Pixels the sampler draws per epoch: the sum over parcels of min(cap, pixels in the parcel)."""
    _, counts = np.unique(parcel, return_counts=True)
    return int(np.minimum(counts, cap).sum())


def _write_predictions(cell_dir: Path, test_ids: list[str], lab: dict, pred: np.ndarray) -> None:
    n = len(test_ids)
    P = np.full((n, 224 * 224), -1, dtype=np.int16)
    M = np.full((n, 224 * 224), -1, dtype=np.int16)
    P[lab["chip"], lab["pos"]] = pred
    M[lab["chip"], lab["pos"]] = lab["y"]
    tmp = cell_dir / "predictions_test.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, ids=np.array(test_ids), pred=P.reshape(n, 224, 224), mask=M.reshape(n, 224, 224))
    tmp.replace(cell_dir / "predictions_test.npz")
