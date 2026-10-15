"""One fit with the encoder inside the training loop, trained or frozen: a copy of the core's ``fit_end_to_end``.

Changed from ``gfm4agri.benchmark.fit.fit_end_to_end``, and nothing else:

* the task is ``finetune.build_ft_task``, with the switch (``model.trainable``), ``encoder_lr`` and
  ``encoder_warmup_epochs`` from the arm;
* a second checkpoint, of highest validation Macro-F1, is scored on test into ``test_metrics_best_f1`` and then
  deleted, since it holds the whole encoder; the protocol's checkpoint (lowest validation loss) gives
  ``test_metrics`` and stays on disk for the test predictions;
* ``results.json`` names the workflow, records ``frozen_encoder`` from the switch and an ``encoder_training``
  block, and the GPU memory peak is reset at the start of the fit, so it is this fit's.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from finetune import WORKFLOW, build_ft_task

REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())

__all__ = ["fit_finetune"]


def _best_epoch(metrics_csv: Path, col: str, pick: str) -> int | None:
    import pandas as pd

    if not metrics_csv.exists():
        return None
    m = pd.read_csv(metrics_csv)
    v = m.dropna(subset=[col]) if col in m else None
    return None if v is None or v.empty else int(v.loc[getattr(v[col], pick)(), "epoch"])


def fit_finetune(cfg: dict, *, fast_dev_run: bool = False) -> dict:
    """Train and evaluate one fit described by ``cfg``; return and write its result record."""
    import lightning.pytorch as pl
    import torch

    from gfm4agri.benchmark.fit import budget_tag, labelled_fraction
    from gfm4agri.benchmark.segmentation import count_params
    from gfm4agri.benchmark.segmentation_data import (
        EuroCropsSegDataModule, draw_support, manifest_sha256, split_protocol)

    pl.seed_everything(cfg["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    root = REPO / cfg["data"]["root"]
    budget = cfg["label_budget"]
    out = REPO / cfg["output_root"] / cfg["run_name"] / f"{budget_tag(budget)}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d = cfg["data"]
    split_dir = REPO / d["split"] if d.get("split") else None
    if budget["mode"] != "pct":
        raise ValueError(f"label_budget.mode {budget['mode']!r}: the K-shot workflow draws percentages")
    keep, support = draw_support(root, budget["pct"], budget["draw_seed"], split_dir=split_dir)
    drawn = sum(v["drawn"] for v in support.values())
    avail = sum(v["available"] for v in support.values())
    print(f"support: {drawn} of {avail} training parcels, {budget['pct']} % per class, "
          f"draw {budget['draw_seed']}")

    dm = EuroCropsSegDataModule(root, cfg["model"]["backbone"], normalisation=d["normalisation"],
                                batch_size=d["batch_size"], num_workers=d["num_workers"],
                                augment=d["augment"], keep_parcel_ids=keep, split_dir=split_dir)
    dm.setup("fit")
    labelled = labelled_fraction(dm.train_dataset)
    print(f"train chips {len(dm.train_dataset)}, val chips {len(dm.val_dataset)}, "
          f"labelled pixel fraction in train {labelled:.3f}")

    m = cfg["model"]
    trainable = bool(m["trainable"])
    enc_lr, warm = float(m.get("encoder_lr", 1e-5)), float(m.get("encoder_warmup_epochs", 1))
    task = build_ft_task(m["backbone"], num_classes=len(dm.class_names), class_names=dm.class_names,
                         bands=dm.bands, n_timesteps=dm.n_timesteps,
                         ignore_index=dm.manifest["ignore_index"], lr=m["lr"],
                         weight_decay=m["weight_decay"], head_dropout=m["head_dropout"],
                         loss=m["loss"], trainable=trainable, encoder_lr=enc_lr,
                         encoder_warmup_epochs=warm)
    params = count_params(task)
    print("parameters:", json.dumps(params))

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/loss",
                                        mode="min", filename="best-loss", save_weights_only=True)
    ckpt_f1 = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/F1_Score",
                                           mode="max", filename="best-f1", save_weights_only=True)
    t = cfg["trainer"]
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=t["precision"],
                         max_epochs=t["max_epochs"], log_every_n_steps=t["log_every_n_steps"],
                         accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
                         callbacks=[ckpt, ckpt_f1], logger=pl.loggers.CSVLogger(out, name="logs"),
                         default_root_dir=out, fast_dev_run=fast_dev_run,
                         deterministic="warn")
    t0 = time.time()
    trainer.fit(task, datamodule=dm)
    fit_s = time.time() - t0
    best = None if fast_dev_run else ckpt.best_model_path
    best_f1 = None if fast_dev_run else ckpt_f1.best_model_path
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    test_f1 = trainer.test(task, datamodule=dm, ckpt_path=best_f1 or None)[0] if best_f1 else {}
    if best_f1:
        Path(best_f1).unlink()
    test = trainer.test(task, datamodule=dm, ckpt_path=best or None)[0]
    metrics_csv = Path(trainer.logger.log_dir) / "metrics.csv"

    manifest = dm.manifest
    results = {
        "run": cfg["run_name"],
        "workflow": WORKFLOW[trainable],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fast_dev_run": fast_dev_run,
        "config": cfg,
        "country": manifest["country"], "year": manifest["year"],
        "dataset": manifest["dataset"], "manifest_sha256": manifest_sha256(root),
        "split_protocol": split_protocol(split_dir, dm.test_is_val),
        "label_budget": {**budget, "labelled_pixel_fraction_train": round(labelled, 4),
                         "support": support},
        "seed": cfg["seed"],
        "effective_batch": d["batch_size"] * t.get("accumulate_grad_batches", 1),
        "chips": {"train": len(dm.train_dataset), "val": len(dm.val_dataset),
                  "test": len(dm.test_dataset) if getattr(dm, "test_dataset", None) else None},
        "backbone": m["backbone"], "frozen_encoder": not trainable,
        "encoder_training": {"trainable": trainable, "encoder_lr": enc_lr if trainable else None,
                             "warmup_epochs": warm if trainable else None, "encoder_mode": "eval",
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
