"""The checks of DESIGN.md for the TESSERA arms on the labelled-pixel route (``check.py`` calls these).

1. Equivalence: at step 0 the student's embeddings of the labelled pixels, computed in the training loop's way
   (``pixelft.prepare`` and ``encode``), equal the core's raster cache at the same pixels.
2. The switch: a few optimiser steps move the student in the trainable arm and none exists in the frozen control,
   the MLP moving in both; two AdamW groups at the designed rates; the student in eval mode in every training step.
3. Cost: seconds per labelled training pixel and per labelled validation pixel, seconds per test chip for the full-chip
   predictions, peak GPU memory; with them, the estimate for the arm's three cells on ``EE_2021_rev``.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

#: Labelled pixels and chips of EE_2021_rev (split blocks4_buf1600_seed0__6b0eb4cb, draw 0), from
#: chip_parcels.parquet and draw_support: training pixels of the support per K, all validation and test pixels.
EE_REV_PX = {"train": {100: 46_560_824, 20: 9_351_257, 5: 2_304_641}, "val": 9_966_900, "test": 19_379_279,
             "test_chips": 1475, "epochs": 15}


def _say(msg: str) -> None:
    import datetime as dt

    print(f"[check {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def raster_cache(arm: dict, chips: Path, split_dir: Path, mach: dict, machine: str, scratch: Path | None) -> Path:
    """The core's raster cache of the arm's backbone on ``chips``: reused if found and matching, else built."""
    from gfm4agri.pipeline.config import encode_batch_size
    from gfm4agri.pipeline.raster_cache import build_raster_cache, resolve_raster_cache

    repo = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
    root = repo / mach["cache_root"]
    decision = resolve_raster_cache(root, scratch or root, arm["model"]["backbone"], chips, split_dir)
    _say(f"raster cache {decision.action} {decision.cache_dir} ({decision.reason})")
    if decision.action == "stop":
        raise SystemExit(f"raster cache: {decision.reason} {decision.mismatches}")
    if decision.action == "compute":
        build_raster_cache(arm["model"]["backbone"], chips, split_dir, decision.cache_dir,
                           batch=encode_batch_size(arm, mach, machine))
    return decision.cache_dir


def _cfg(arm: dict, chips: Path, split_dir: Path, cache_dir: Path, *, trainable: bool, workers: int) -> dict:
    from gfm4agri.pipeline.config import repo_relative

    m = {**arm["model"], "trainable": trainable}
    return {"data": {"root": repo_relative(chips), "split": repo_relative(split_dir),
                     "normalisation": arm["data"]["normalisation"], "batch_size": arm["data"]["batch_size"],
                     "num_workers": workers, "augment": False, "image_root": repo_relative(cache_dir)},
            "model": m}


def check_equivalence(arm: dict, chips: Path, split_dir: Path, cache_dir: Path, batches: int = 8) -> dict:
    """Embeddings of the labelled training pixels (K = 100 %): the student in the loop against the cache."""
    import torch

    from fit_pixels import build_pixel_cell, pixel_loaders
    from gfm4agri.benchmark.segmentation_data import draw_support
    from pixelft import encode, load_student

    keep, _ = draw_support(chips, 100, 0, split_dir=split_dir)
    cfg = _cfg(arm, chips, split_dir, cache_dir, trainable=False, workers=0)
    dm, _, _ = build_pixel_cell(cfg, keep, stages=("fit",))
    student = load_student(arm["model"]).cuda().eval()
    live = pixel_loaders(dm, source="composites", batch_size=4, num_workers=0, splits=("train",))["train"]
    ref = pixel_loaders(dm, source="cache", batch_size=4, num_workers=0, splits=("train",))["train"]
    # The same chips in the same order, on both sides.
    for dl in (live, ref):
        dl.dataset.ds.image_files = dl.dataset.ds.image_files[: 4 * batches]
        dl.dataset.ds.segmentation_mask_files = dl.dataset.ds.segmentation_mask_files[: 4 * batches]
    cos, rel, n = [], 0.0, 0
    from torch.utils.data import DataLoader

    mk = lambda dl: DataLoader(dl.dataset, batch_size=4, shuffle=False, collate_fn=dl.collate_fn)  # noqa: E731
    for a, b in zip(mk(live), mk(ref), strict=True):
        if a["chips"] != b["chips"] or not torch.equal(a["y"], b["y"]):
            raise AssertionError("the two loaders disagree on chips or labels")
        with torch.no_grad():
            e = encode(student, {"groups": [tuple(x.cuda() for x in g) for g in a["prep"]["groups"]],
                                 "empty": a["prep"]["empty"].cuda(), "n": a["prep"]["n"]}, grad=False).cpu()
        r = b["emb"].float()
        c = torch.nn.functional.cosine_similarity(e.double(), r.double(), dim=1)
        cos.append(c.numpy())
        rel = max(rel, float((e - r).abs().max() / (r.abs().max() + 1e-12)))
        n += len(r)
    cos = np.concatenate(cos)
    out = {"pixels": n, "min_cosine": float(cos.min()), "p001_cosine": float(np.quantile(cos, 0.001)),
           "median_cosine": float(np.median(cos)), "max_abs_diff_over_max_abs": rel,
           "passed": float(np.quantile(cos, 0.001)) > 0.999}
    _say(f"1. equivalence over {n:,} labelled pixels: cosine min {out['min_cosine']:.6f}, 0.1 % quantile "
         f"{out['p001_cosine']:.6f}, median {out['median_cosine']:.7f}; max |diff| / max |ref| {rel:.2e}")
    del student
    gc.collect()
    torch.cuda.empty_cache()
    return out


def run_steps(arm: dict, chips: Path, split_dir: Path, cache_dir: Path, *, trainable: bool, batches: int,
              val_batches: int, workers: int) -> dict:
    """Train ``batches`` steps at K = 100 % and validate ``val_batches``: what moved, how fast, how much memory."""
    import lightning.pytorch as pl
    import torch

    from fit_pixels import build_pixel_cell, pixel_loaders
    from gfm4agri.benchmark.segmentation_data import draw_support

    pl.seed_everything(0, workers=True)
    torch.cuda.reset_peak_memory_stats()
    keep, _ = draw_support(chips, 100, 0, split_dir=split_dir)
    cfg = _cfg(arm, chips, split_dir, cache_dir, trainable=trainable, workers=workers)
    dm, task, source = build_pixel_cell(cfg, keep, stages=("fit",))
    loaders = pixel_loaders(dm, source=source, batch_size=arm["data"]["batch_size"], num_workers=workers,
                            splits=("train", "val"))
    student = dict(task.student.named_parameters()) if task.student is not None else {}
    rest = {n: p for n, p in task.named_parameters() if not n.startswith("student.")}
    before_s = {n: p.detach().cpu().clone() for n, p in student.items()}
    before_r = {n: p.detach().cpu().clone() for n, p in rest.items()}

    class Timing(pl.Callback):
        def __init__(self):
            self.train, self.val, self.px, self.vpx, self.lrs, self.modes = [], [], [], [], [], []
            self._t, self._h = None, None

        def on_fit_start(self, trainer, m):
            if m.student is not None:
                self._h = m.student.s2_backbone.register_forward_pre_hook(
                    lambda mod, args: self.modes.append(mod.training) if m.training else None)

        def on_fit_end(self, trainer, m):
            if self._h is not None:
                self._h.remove()

        def _tick(self, store):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if self._t is not None:
                store.append(now - self._t)
            self._t = now

        def on_train_epoch_start(self, trainer, m):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_train_batch_end(self, trainer, m, outputs, batch, batch_idx):
            self._tick(self.train)
            self.px.append(int(batch["y"].numel()))
            self.lrs.append({g.get("name", str(i)): g["lr"] for i, g in enumerate(trainer.optimizers[0].param_groups)})

        def on_validation_epoch_start(self, trainer, m):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_validation_batch_end(self, trainer, m, outputs, batch, batch_idx, dataloader_idx=0):
            self._tick(self.val)
            self.vpx.append(int(batch["y"].numel()))

    timing = Timing()
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=arm["trainer"]["precision"], max_epochs=1,
                         limit_train_batches=batches, limit_val_batches=val_batches, num_sanity_val_steps=0,
                         logger=False, enable_checkpointing=False, enable_progress_bar=False, callbacks=[timing])
    trainer.fit(task, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    opt = trainer.optimizers[0]
    groups = {g.get("name", str(i)): g for i, g in enumerate(opt.param_groups)}
    moved_s = sum(int(not torch.equal(before_s[n], p.detach().cpu())) for n, p in student.items())
    moved_r = sum(int(not torch.equal(before_r[n], p.detach().cpu())) for n, p in rest.items())
    skip = 2
    # Time per labelled pixel: the steps' wall time over their pixels, past the warm-up steps.
    s_px = sum(timing.train[skip:]) / max(1, sum(timing.px[skip:]))
    s_vpx = sum(timing.val[1:]) / max(1, sum(timing.vpx[1:]))
    out = {
        "trainable": trainable, "steps": len(timing.train), "batch_chips": arm["data"]["batch_size"],
        "pixels_per_step_median": float(np.median(timing.px)) if timing.px else None,
        "student_tensors": len(student), "student_tensors_moved": moved_s,
        "other_tensors": len(rest), "other_tensors_moved": moved_r,
        "student_params": sum(p.numel() for p in student.values()),
        "groups": {k: {"tensors": len(g["params"]), "initial_lr": g.get("initial_lr", g["lr"]), "lr_now": g["lr"]}
                   for k, g in groups.items()},
        "encoder_group_is_student": ({id(p) for p in groups["encoder"]["params"]} == {id(p) for p in student.values()})
        if "encoder" in groups else None,
        "encoder_lr_per_step": [r.get("encoder") for r in timing.lrs],
        "student_forwards_in_training": len(timing.modes),
        "student_in_train_mode_any_step": any(timing.modes),
        "s_per_train_pixel": s_px, "s_per_val_pixel": s_vpx,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    del task, trainer, dm, before_s, before_r
    gc.collect()
    torch.cuda.empty_cache()
    return out


def time_full_chips(arm: dict, chips: Path, split_dir: Path, cache_dir: Path, n: int = 6) -> float:
    """Seconds per test chip for the full-chip prediction (every pixel through the student), as ``FtFitPredictor``."""
    import torch

    from fit_pixels import build_pixel_cell, pixel_loaders
    from gfm4agri.embeddings.tessera_v2 import month_doy
    from pixelft import make_collate

    cfg = _cfg(arm, chips, split_dir, cache_dir, trainable=True, workers=0)
    dm, task, _ = build_pixel_cell(cfg, None, stages=("test",))
    task = task.cuda().eval()
    px = pixel_loaders(dm, source="composites", batch_size=1, num_workers=0, splits=("test",))["test"].dataset
    px.all_pixels = True
    collate = make_collate(month_doy(dm.manifest["year"], dm.manifest["imagery"]["n_months"]))
    ts = []
    for i in range(min(n, len(px))):
        t0 = time.perf_counter()
        b = collate([px[i]])
        b["prep"] = {"groups": [tuple(x.cuda() for x in g) for g in b["prep"]["groups"]],
                     "empty": b["prep"]["empty"].cuda(), "n": b["prep"]["n"]}
        b["y"] = b["y"].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            task.pixel_forward(b).output.argmax(1).cpu()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    del task
    gc.collect()
    torch.cuda.empty_cache()
    return float(np.median(ts[1:] or ts))


def estimate(s_px: float, s_vpx: float, s_chip: float) -> dict:
    """Hours for the arm's three cells on EE_2021_rev: epochs of training and validation, two tests, the predictions."""
    e = EE_REV_PX
    per_cell = {k: round((s_px * n + s_vpx * e["val"]) * e["epochs"] / 3600 + s_vpx * e["test"] * 2 / 3600
                         + s_chip * e["test_chips"] / 3600, 2) for k, n in e["train"].items()}
    return {"hours_total": round(sum(per_cell.values()), 1), "hours_per_cell": per_cell}
