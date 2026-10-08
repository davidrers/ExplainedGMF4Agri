"""Train and evaluate one frozen-encoder segmentation fit on the EuroCrops chips.

    poetry run python scripts/seg/train.py -c configs/seg/terramind_v1_small_ee_pilot.yaml
    poetry run python scripts/seg/train.py -c ... --fast-dev-run
    poetry run python scripts/seg/train.py -c ... --set label_budget.mode=pct label_budget.pct=5

Writes ``<output_root>/<run_name>/<budget>_seed<seed>/``: the Lightning logs and checkpoint,
``config.yaml`` as resolved, and ``results.json`` carrying the configuration, the label
budget, the country, the split protocol, the parameter counts and the metrics.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _set(cfg: dict, dotted: str) -> None:
    key, value = dotted.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def _budget_tag(b: dict) -> str:
    return "dense" if b["mode"] == "dense" else f"P{b['pct']:g}_draw{b['draw_seed']}"


def _labelled_fraction(dataset) -> float:
    """Mean labelled share of the training masks under the budget, from the label rasters alone.

    Reading the masks rather than the samples keeps this cheap on a full-country chip set,
    where a pass over the samples would read every image.
    """
    import rasterio

    keep = getattr(dataset, "keep_parcel_ids", None)
    fr = []
    for f in dataset.segmentation_mask_files:
        with rasterio.open(f) as src:
            mask = src.read(1)
        if keep is not None:
            with rasterio.open(str(f)[: -len(".mask.tif")] + ".parcels.tif") as src:
                mask = np.where(np.isin(src.read(1), keep), mask, -1)
        fr.append(float((mask >= 0).mean()))
    return float(np.mean(fr))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, dotted keys")
    ap.add_argument("--fast-dev-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for s in args.set:
        _set(cfg, s)

    import lightning.pytorch as pl
    import torch

    from gfm4agri.benchmark.segmentation import build_task, count_params
    from gfm4agri.benchmark.segmentation_data import (
        EuroCropsSegDataModule, draw_support, manifest_sha256, split_protocol)

    pl.seed_everything(cfg["seed"], workers=True)
    root = REPO / cfg["data"]["root"]
    budget = cfg["label_budget"]
    out = REPO / cfg["output_root"] / cfg["run_name"] / f"{_budget_tag(budget)}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d = cfg["data"]
    split_dir = REPO / d["split"] if d.get("split") else None
    keep, support = None, None
    if budget["mode"] == "dense" and split_dir is not None:
        # Under a spatial split the pool mask is never fully dense: the buffer withholds the
        # parcels near held-out blocks, which the 100 % draw leaves out.
        keep, support = draw_support(root, 100, budget.get("draw_seed", 0), split_dir=split_dir)
    elif budget["mode"] == "pct":
        keep, support = draw_support(root, budget["pct"], budget["draw_seed"], split_dir=split_dir)
        drawn = sum(v["drawn"] for v in support.values())
        avail = sum(v["available"] for v in support.values())
        print(f"support: {drawn} of {avail} training parcels, {budget['pct']} % per class, "
              f"draw {budget['draw_seed']}")
    elif budget["mode"] != "dense":
        raise ValueError(f"label_budget.mode must be dense or pct, not {budget['mode']!r}")

    dm = EuroCropsSegDataModule(root, cfg["model"]["backbone"], normalisation=d["normalisation"],
                                batch_size=d["batch_size"], num_workers=d["num_workers"],
                                augment=d["augment"], keep_parcel_ids=keep, split_dir=split_dir)
    dm.setup("fit")
    labelled = _labelled_fraction(dm.train_dataset)
    print(f"train chips {len(dm.train_dataset)}, val chips {len(dm.val_dataset)}, "
          f"labelled pixel fraction in train {labelled:.3f}")

    batch = next(iter(dm.train_dataloader()))
    x = dm.aug(batch)["image"]
    other = tuple(d for d in range(x.dim()) if d != 1)
    layout = "(B, C, T, H, W)" if x.dim() == 5 else "(B, C, H, W)"
    print(f"batch image {tuple(x.shape)} {layout}, mask {tuple(batch['mask'].shape)}")
    print("normalised per-band mean:", np.round(x.mean(other).numpy(), 2).tolist()[:16])
    print("normalised per-band std: ", np.round(x.std(other).numpy(), 2).tolist()[:16])

    m = cfg["model"]
    task = build_task(m["backbone"], num_classes=len(dm.class_names), class_names=dm.class_names,
                      bands=dm.bands, n_timesteps=dm.n_timesteps,
                      ignore_index=dm.manifest["ignore_index"], lr=m["lr"],
                      weight_decay=m["weight_decay"], head_dropout=m["head_dropout"],
                      loss=m["loss"])
    params = count_params(task)
    print("parameters:", json.dumps(params))

    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/loss",
                                        mode="min", filename="best-loss", save_weights_only=True)
    t = cfg["trainer"]
    # Gradient accumulation keeps the effective batch identical across backbones when a
    # larger encoder forces a smaller micro-batch onto the GPU.
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=t["precision"],
                         max_epochs=t["max_epochs"], log_every_n_steps=t["log_every_n_steps"],
                         accumulate_grad_batches=t.get("accumulate_grad_batches", 1),
                         callbacks=[ckpt], logger=pl.loggers.CSVLogger(out, name="logs"),
                         default_root_dir=out, fast_dev_run=args.fast_dev_run,
                         deterministic="warn")
    t0 = time.time()
    trainer.fit(task, datamodule=dm)
    fit_s = time.time() - t0
    best = None if args.fast_dev_run else ckpt.best_model_path
    test = trainer.test(task, datamodule=dm, ckpt_path=best or None)[0]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    manifest = dm.manifest
    results = {
        "run": cfg["run_name"],
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fast_dev_run": args.fast_dev_run,
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
        "backbone": m["backbone"], "frozen_encoder": True,
        "representation": dm.representation,
        "normalisation": {"mode": dm.normalisation, "means": dm.norm_means, "stds": dm.norm_stds},
        "params": params,
        "best_checkpoint": best,
        "fit_seconds": round(fit_s, 1), "peak_gpu_gb": round(peak_gb, 2),
        "test_metrics": {k: float(v) for k, v in test.items()},
        "class_names": dm.class_names,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"results -> {out / 'results.json'}")
    for k in ("test/F1_Score", "test/mIoU", "test/Pixel_Accuracy", "test/loss"):
        if k in results["test_metrics"]:
            print(f"  {k} = {results['test_metrics'][k]:.4f}")


if __name__ == "__main__":
    main()
