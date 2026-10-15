"""One TESSERA v2 cell on the labelled-pixel route, encoder trained or frozen (``pixelft``).

The core's ``fit_end_to_end`` for the ``tessera_v2_medium`` arm, with:

* the core datamodule built as there (raster cache in ``image_root``, label budget, split), its datasets wrapped in
  ``pixelft.LabelledPixels`` and loaded four chips per step as the core's loaders (training shuffled, last partial
  batch dropped);
* the task of ``pixelft.build_pixel_task``: the core's MLP and head, the cache's normalisation statistics fixed, and,
  when trainable, the TESSERA student as a second AdamW group at ``encoder_lr`` after a linear warm-up;
* the second checkpoint and the ``results.json`` fields of ``fit_ft.fit_finetune``, and a ``pixels`` block.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import yaml

from fit_ft import _best_epoch
from pixelft import PIXEL_WORKFLOW, LabelledPixels, build_pixel_task, load_student, make_collate

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

__all__ = ["build_pixel_cell", "fit_pixels", "pixel_loaders"]


def pixel_loaders(dm, *, source: str, batch_size: int, num_workers: int, splits=("train", "val", "test")) -> dict:
    """``split -> DataLoader`` of labelled pixels over the core datamodule's datasets (``setup`` already run)."""
    from torch.utils.data import DataLoader

    from gfm4agri.embeddings.tessera_v2 import month_doy

    man = dm.manifest
    collate = make_collate(month_doy(man["year"], man["imagery"]["n_months"]))
    out = {}
    for split in splits:
        ds = getattr(dm, f"{split}_dataset")
        px = LabelledPixels(ds, dm.root / "chips", source=source, band_names=man["imagery"]["band_names"],
                            n_months=man["imagery"]["n_months"], ignore_index=man["ignore_index"],
                            img_suffix=dm.img_suffix)
        out[split] = DataLoader(px, batch_size=batch_size, shuffle=split == "train", drop_last=split == "train",
                                num_workers=num_workers, collate_fn=collate,
                                persistent_workers=num_workers > 0 and split != "test")
    return out


def build_pixel_cell(cfg: dict, keep=None, *, stages=("fit", "test")):
    """The core datamodule of the cell and its pixel task, the student loaded when the encoder runs."""
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

    d, m = cfg["data"], cfg["model"]
    if not d.get("image_root"):
        raise ValueError("the labelled-pixel route reads the raster cache's statistics: data.image_root is unset")
    dm = EuroCropsSegDataModule(REPO / d["root"], m["backbone"], normalisation=d["normalisation"],
                                batch_size=d["batch_size"], num_workers=d["num_workers"], augment=False,
                                keep_parcel_ids=keep, split_dir=REPO / d["split"], image_root=REPO / d["image_root"])
    for s in stages:
        dm.setup(s)
    trainable = bool(m["trainable"])
    task = build_pixel_task(num_classes=len(dm.class_names), class_names=dm.class_names, bands=dm.bands,
                            ignore_index=dm.manifest["ignore_index"], lr=m["lr"], weight_decay=m["weight_decay"],
                            head_dropout=m["head_dropout"], loss=m["loss"], trainable=trainable,
                            encoder_lr=float(m.get("encoder_lr", 1e-5)),
                            encoder_warmup_epochs=float(m.get("encoder_warmup_epochs", 1)),
                            encoder_chunk=int(m.get("encoder_chunk", 16384)),
                            means=dm.norm_means, stds=dm.norm_stds,
                            student=load_student(m) if trainable else None)
    return dm, task, ("composites" if trainable else "cache")


def fit_pixels(cfg: dict, *, fast_dev_run: bool = False) -> dict:
    """Train and evaluate one cell described by ``cfg``; return and write its result record."""
    import lightning.pytorch as pl
    import torch

    from gfm4agri.benchmark.fit import budget_tag, labelled_fraction
    from gfm4agri.benchmark.segmentation import count_params
    from gfm4agri.benchmark.segmentation_data import draw_support, manifest_sha256, split_protocol

    pl.seed_everything(cfg["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    root = REPO / cfg["data"]["root"]
    budget = cfg["label_budget"]
    out = REPO / cfg["output_root"] / cfg["run_name"] / f"{budget_tag(budget)}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d, m, t = cfg["data"], cfg["model"], cfg["trainer"]
    split_dir = REPO / d["split"]
    if budget["mode"] != "pct":
        raise ValueError(f"label_budget.mode {budget['mode']!r}: the K-shot workflow draws percentages")
    keep, support = draw_support(root, budget["pct"], budget["draw_seed"], split_dir=split_dir)
    drawn = sum(v["drawn"] for v in support.values())
    avail = sum(v["available"] for v in support.values())
    print(f"support: {drawn} of {avail} training parcels, {budget['pct']} % per class, draw {budget['draw_seed']}")

    dm, task, source = build_pixel_cell(cfg, keep)
    labelled = labelled_fraction(dm.train_dataset)
    print(f"train chips {len(dm.train_dataset)}, val chips {len(dm.val_dataset)}, "
          f"labelled pixel fraction in train {labelled:.3f}; encoder input: {source}")
    loaders = pixel_loaders(dm, source=source, batch_size=d["batch_size"], num_workers=d["num_workers"])
    trainable = bool(m["trainable"])
    params = count_params(task)
    params["student_total"] = sum(p.numel() for p in task.student.parameters()) if task.student is not None else 0
    params["student_trainable"] = sum(p.numel() for p in task.encoder_parameters())
    print("parameters:", json.dumps(params))

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/loss",
                                        mode="min", filename="best-loss", save_weights_only=True)
    ckpt_f1 = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/F1_Score",
                                           mode="max", filename="best-f1", save_weights_only=True)
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=t["precision"],
                         max_epochs=t["max_epochs"], log_every_n_steps=t["log_every_n_steps"],
                         accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
                         callbacks=[ckpt, ckpt_f1], logger=pl.loggers.CSVLogger(out, name="logs"),
                         default_root_dir=out, fast_dev_run=fast_dev_run, deterministic="warn")
    t0 = time.time()
    trainer.fit(task, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
    fit_s = time.time() - t0
    best = None if fast_dev_run else ckpt.best_model_path
    best_f1 = None if fast_dev_run else ckpt_f1.best_model_path
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    test_f1 = trainer.test(task, dataloaders=loaders["test"], ckpt_path=best_f1)[0] if best_f1 else {}
    if best_f1:
        Path(best_f1).unlink()
    test = trainer.test(task, dataloaders=loaders["test"], ckpt_path=best or None)[0]
    metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"

    manifest = dm.manifest
    results = {
        "run": cfg["run_name"],
        "workflow": PIXEL_WORKFLOW[trainable],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fast_dev_run": fast_dev_run,
        "config": cfg,
        "country": manifest["country"], "year": manifest["year"],
        "dataset": manifest["dataset"], "manifest_sha256": manifest_sha256(root),
        "split_protocol": split_protocol(split_dir, dm.test_is_val),
        "label_budget": {**budget, "labelled_pixel_fraction_train": round(labelled, 4), "support": support},
        "seed": cfg["seed"],
        "effective_batch": d["batch_size"] * t.get("accumulate_grad_batches", 1),
        "chips": {"train": len(dm.train_dataset), "val": len(dm.val_dataset), "test": len(dm.test_dataset)},
        "pixels": {"train_steps": int(trainer.global_step), "layout": "labelled pixels of each batch, (1, 128, N, 1)",
                   "dropped_metrics": ["Boundary_mIoU"]},
        "backbone": m["backbone"], "frozen_encoder": not trainable,
        "encoder_training": {"trainable": trainable, "encoder_lr": float(m.get("encoder_lr", 1e-5)) if trainable else None,
                             "warmup_epochs": float(m.get("encoder_warmup_epochs", 1)) if trainable else None,
                             "encoder_mode": "eval", "encoder_input": source,
                             "encoder_precision": "bf16 autocast, as the raster cache",
                             "decoder_lr": m["lr"], "weight_decay": m["weight_decay"]},
        "representation": dm.representation,
        "normalisation": {"mode": dm.normalisation, "means": dm.norm_means, "stds": dm.norm_stds},
        "params": params,
        "best_checkpoint": best,
        "fit_seconds": round(fit_s, 1), "peak_gpu_gb": round(peak_gb, 2),
        "test_metrics": {k: float(v) for k, v in test.items()},
        "test_metrics_best_f1": {k: float(v) for k, v in test_f1.items()},
        "best_f1_epoch": _best_epoch(metrics_csv, "val/F1_Score", "idxmax"),
        "class_names": dm.class_names,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"results -> {out / 'results.json'}")
    for k in ("test/F1_Score", "test/mIoU", "test/Pixel_Accuracy", "test/loss"):
        if k in results["test_metrics"]:
            extra = results["test_metrics_best_f1"].get(k)
            print(f"  {k} = {results['test_metrics'][k]:.4f}"
                  + (f"   (best-F1 checkpoint {extra:.4f})" if extra is not None else ""))
    return results
