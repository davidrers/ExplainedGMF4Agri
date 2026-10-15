"""The checks of DESIGN.md, per arm of this folder, before the run on all of Estonia.

Token-grid arms (route ``end_to_end``), here; TESSERA arms (route ``labelled_pixels``), in ``check_pixels``:

1. Equivalence: at step 0 the fine-tuned model's features at the cache point (the input of ``ChannelBottleneck``)
   equal what the main workflow's ``generate_embeddings`` caches for the same chips.
2. The switch: a few optimiser steps move the encoder in the trainable model and leave it bit-identical in the
   frozen one, the decoder moving in both; two AdamW groups at the designed learning rates, the encoder group
   holding exactly the encoder's trainable parameters; the encoder in eval mode inside every training step.
3. Cost: seconds per training and per validation chip at the arm's batch, and peak GPU memory, on this machine;
   with them, the estimate for the arm's nine cells on ``EE_2021_rev``.

With ``--predictions`` it instead checks the finished cells of the chip set: the pixel accuracy of each
``predictions_test.npz`` must equal the test pixel accuracy the fit logged, so the predictions come from the model
that was tested (for a fine-tuned cell: its own encoder, with the coordinates).

The command line is that of ``run.py``, so the cluster runs it through ``submit.sh`` (one job per arm):

    poetry run python experiments/2026-10-15_kshot_finetune/check.py -e experiments/2026-10-15_kshot_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --arms terramind_v1_large_s2s1_ft
    RUNNER=experiments/2026-10-15_kshot_finetune/check.py bash scripts/cluster/submit.sh experiments/2026-10-15_kshot_finetune/experiment.yaml EE_2021_rev_mini terramind_v1_large_s2s1_ft prithvi_eo_v2_600_tl_ft thor_v1_large_s2s1_ft tessera_v2_medium_ft tessera_v2_medium_lab
    poetry run python experiments/2026-10-15_kshot_finetune/check.py -e ... --chips data/eurocrops_chips/EE_2021_rev_mini --predictions

Each arm's findings go to ``results/<name>/<chip set>/checks/<arm>_<machine>.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

#: Training chips per K on EE_2021_rev (split blocks4_buf1600_seed0__6b0eb4cb, draw 0), as every fit on it
#: records them (e.g. results/2026-10-13_terramind_s1_s2/EE_2021_rev/*/P*_draw0_seed0/results.json), and its
#: validation and test chips.
EE_REV = {"train": {100: 4898, 20: 4213, 5: 2706}, "val": 769, "test": 1475, "seeds": 1, "epochs": 15}


def _say(msg: str) -> None:
    print(f"[check {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _datamodule(arm: dict, chips: Path, split_dir: Path, *, batch_size: int, num_workers: int, augment: bool):
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

    return EuroCropsSegDataModule(chips, arm["model"]["backbone"], normalisation=arm["data"]["normalisation"],
                                  batch_size=batch_size, num_workers=num_workers, augment=augment,
                                  split_dir=split_dir)


def _task(arm: dict, dm, trainable: bool):
    from finetune import build_ft_task

    m = arm["model"]
    return build_ft_task(m["backbone"], num_classes=len(dm.class_names), class_names=dm.class_names,
                         bands=dm.bands, n_timesteps=dm.n_timesteps, ignore_index=dm.manifest["ignore_index"],
                         lr=m["lr"], weight_decay=m["weight_decay"], head_dropout=m["head_dropout"],
                         loss=m["loss"], trainable=trainable, encoder_lr=m["encoder_lr"],
                         encoder_warmup_epochs=m["encoder_warmup_epochs"])


# ------------------------------------------------------------------------------------------------- 1. equivalence

def check_equivalence(arm: dict, chips: Path, split_dir: Path, tmp: Path, batch_size: int, workers: int) -> dict:
    import torch

    from gfm4agri.benchmark.cached import IDENTITY, generate_embeddings, variant_name

    backbone = arm["model"]["backbone"]
    precision = arm["trainer"]["precision"]
    cache = tmp / backbone
    t0 = time.time()
    generate_embeddings(backbone, chips, cache, normalisation=arm["data"]["normalisation"], variants=[IDENTITY],
                        batch_size=batch_size, num_workers=workers, precision=precision, split_dir=split_dir)
    _say(f"core cache of the identity variant in {time.time() - t0:.0f} s")
    meta = json.loads((cache / "cache.json").read_text())

    dm = _datamodule(arm, chips, split_dir, batch_size=batch_size, num_workers=0, augment=False)
    dm.setup("fit")
    dm.setup("test")
    model = _task(arm, dm, trainable=True).model.cuda().eval()
    neck = next(m for m in model.modules() if type(m).__name__ == "ChannelBottleneck")
    seen = {}
    hook = neck.register_forward_pre_hook(lambda mod, args: seen.__setitem__("f", [f.detach() for f in args[0]]))
    from torch.utils.data import DataLoader

    rows, worst_rel, cmin = 0, 0.0, 1.0
    for split, ds in (("train", dm.train_dataset), ("test", dm.test_dataset)):
        for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
            names = [Path(str(f)).name[: -len(dm.img_suffix)] for f in batch["filename"]]
            batch = dm.aug(batch)
            if dm.modality_split:
                batch["image"] = dm.split_modalities(batch["image"])
            x = batch["image"]
            x = {k: v.cuda() for k, v in x.items()} if isinstance(x, dict) else x.cuda()
            rest = {k: batch[k].cuda() for k in ("temporal_coords", "location_coords") if k in batch}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                model(x, **rest)
            for layer, f in enumerate(seen["f"]):
                live = f.float().cpu().numpy().astype(np.float16).astype(np.float32)
                for b, cid in enumerate(names):
                    ref = np.load(cache / variant_name(*IDENTITY) / f"layer_{layer:02d}"
                                  / f"{cid}_merged_embedding.npy").astype(np.float32)
                    if ref.shape != live[b].shape:
                        raise AssertionError(f"{cid} layer {layer}: cached {ref.shape}, live {live[b].shape}")
                    worst_rel = max(worst_rel, float(np.abs(live[b] - ref).max() / (np.abs(ref).max() + 1e-12)))
                    cmin = min(cmin, _cos(live[b].ravel(), ref.ravel()))
                    rows += 1
    hook.remove()
    out = {"chip_layers": rows, "max_abs_diff_over_max_abs": worst_rel, "min_cosine": cmin,
           "identical": worst_rel == 0.0,
           "cache_point_channels": meta.get("channel_list"), "passed": cmin > 0.9999}
    _say(f"1. equivalence over {rows} chip-layers: min cosine {cmin:.7f}, max |diff| / max |ref| {worst_rel:.2e}"
         + (" (identical at the cache's float16)" if worst_rel == 0.0 else ""))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------------------------------- 2 and 3. switch, cost

def _timing_callback():
    import lightning.pytorch as pl
    import torch

    class Timing(pl.Callback):
        """Wall time between consecutive batch ends (data wait included), learning rates and encoder mode."""

        def __init__(self) -> None:
            self.train, self.val, self.lrs, self.modes = [], [], [], []
            self._t = self._hook = None

        def on_fit_start(self, trainer, pl_module):
            # The encoder's mode as its forward runs, during training steps only.
            enc = pl_module.model.encoder
            self._hook = enc.register_forward_pre_hook(
                lambda mod, args: self.modes.append(mod.training) if pl_module.training else None)

        def on_fit_end(self, trainer, pl_module):
            self._hook.remove()

        def _tick(self, store: list) -> None:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if self._t is not None:
                store.append(now - self._t)
            self._t = now

        def on_train_epoch_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self._tick(self.train)
            self.lrs.append({g.get("name", str(i)): g["lr"] for i, g in enumerate(trainer.optimizers[0].param_groups)})

        def on_validation_epoch_start(self, trainer, pl_module):
            torch.cuda.synchronize()
            self._t = time.perf_counter()

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self._tick(self.val)

    return Timing()


def run_steps(arm: dict, chips: Path, split_dir: Path, *, trainable: bool, batches: int, val_batches: int,
              workers: int) -> dict:
    """Train ``batches`` micro-batches (one epoch, cut short) and validate ``val_batches``; what moved, and how fast."""
    import lightning.pytorch as pl
    import torch

    pl.seed_everything(0, workers=True)
    torch.cuda.reset_peak_memory_stats()
    t = arm["trainer"]
    dm = _datamodule(arm, chips, split_dir, batch_size=arm["data"]["batch_size"], num_workers=workers,
                     augment=arm["data"]["augment"])
    task = _task(arm, dm, trainable)
    enc = dict(task.model.encoder.named_parameters())
    rest = {n: p for n, p in task.named_parameters() if not n.startswith("model.encoder.")}
    before_enc = {n: p.detach().cpu().clone() for n, p in enc.items()}
    before_rest = {n: p.detach().cpu().clone() for n, p in rest.items()}
    timing = _timing_callback()
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=t["precision"], max_epochs=1,
                         limit_train_batches=batches, limit_val_batches=val_batches, num_sanity_val_steps=0,
                         accumulate_grad_batches=t.get("accumulate_grad_batches", 1), logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, callbacks=[timing])
    trainer.fit(task, datamodule=dm)
    opt = trainer.optimizers[0]
    groups = {g.get("name", str(i)): g for i, g in enumerate(opt.param_groups)}
    trainable_ids = {id(p) for p in task.parameters() if p.requires_grad}
    enc_trainable = {id(p) for p in task.model.encoder.parameters() if p.requires_grad}
    moved_enc = sum(int(not torch.equal(before_enc[n], p.detach().cpu())) for n, p in enc.items())
    moved_rest = sum(int(not torch.equal(before_rest[n], p.detach().cpu())) for n, p in rest.items())
    bs = arm["data"]["batch_size"]
    skip = 3                                     # the first steps carry compilation and allocator warm-up
    tr = timing.train[skip:] or timing.train
    va = timing.val[1:] or timing.val
    out = {
        "trainable": trainable, "micro_batches": len(timing.train), "batch_size": bs,
        "encoder_tensors": len(enc), "encoder_tensors_moved": moved_enc,
        "other_tensors": len(rest), "other_tensors_moved": moved_rest,
        "encoder_params": sum(p.numel() for p in enc.values()),
        "trainable_params": sum(p.numel() for p in task.parameters() if p.requires_grad),
        "groups": {k: {"tensors": len(g["params"]), "initial_lr": g.get("initial_lr", g["lr"]), "lr_now": g["lr"]}
                   for k, g in groups.items()},
        "encoder_group_is_encoder": ({id(p) for p in groups["encoder"]["params"]} == enc_trainable)
        if "encoder" in groups else None,
        "groups_cover_trainable": trainable_ids <= {id(p) for g in opt.param_groups for p in g["params"]},
        "encoder_lr_per_step": [r.get("encoder") for r in timing.lrs],
        "encoder_forwards_in_training": len(timing.modes),
        "encoder_in_train_mode_any_step": any(timing.modes),
        "s_per_train_chip": float(np.median(tr)) / bs if tr else None,
        "s_per_val_chip": float(np.median(va)) / bs if va else None,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    del task, trainer, dm, before_enc, before_rest
    gc.collect()
    torch.cuda.empty_cache()
    return out


def estimate(s_train: float, s_val: float) -> dict:
    """Hours for the arm's cells on EE_2021_rev (three budgets, one seed) at the measured speeds."""
    e = EE_REV
    train_h = s_train * sum(e["train"].values()) * e["epochs"] * e["seeds"] / 3600
    val_h = s_val * e["val"] * e["epochs"] * len(e["train"]) * e["seeds"] / 3600
    # Three passes over the test chips per cell: the two checkpoints' tests and the predictions.
    test_h = s_val * e["test"] * 3 * len(e["train"]) * e["seeds"] / 3600
    per_cell = {k: round((s_train * n + s_val * e["val"]) * e["epochs"] / 3600 + s_val * e["test"] * 3 / 3600, 2)
                for k, n in e["train"].items()}
    return {"hours_total": round(train_h + val_h + test_h, 1), "hours_train": round(train_h, 1),
            "hours_val": round(val_h, 1), "hours_test": round(test_h, 1), "hours_per_cell": per_cell}


# --------------------------------------------------------------------------------------------------------- TESSERA

def check_tessera(arm: dict, chips: Path, split_dir: Path, mach: dict, args) -> dict:
    """The three checks for a labelled-pixel arm; the trainable one also gets the cost of its full-chip predictions."""
    import torch

    import check_pixels as cp

    rec = {}
    cache = cp.raster_cache(arm, chips, split_dir, mach, args.machine, args.scratch)
    trainable = bool(arm["model"]["trainable"])
    if trainable and "equivalence" not in args.skip:
        rec["equivalence"] = cp.check_equivalence(arm, chips, split_dir, cache)
    key = "trainable" if trainable else "frozen"
    if key not in args.skip:
        try:
            r = cp.run_steps(arm, chips, split_dir, cache, trainable=trainable, batches=args.batches,
                             val_batches=4, workers=mach["num_workers"])
        except torch.cuda.OutOfMemoryError as err:
            r = {"out_of_memory": str(err).split("\n")[0], "passed": False}
            _say(f"2. {key}: out of memory on this card")
        else:
            enc_lr = arm["model"]["encoder_lr"]
            lrs = [x for x in r["encoder_lr_per_step"] if x is not None]
            r["passed"] = bool(
                (r["student_tensors_moved"] == r["student_tensors"] > 0 if trainable else r["student_tensors"] == 0)
                and r["other_tensors_moved"] > 0
                and (not trainable or (r["encoder_group_is_student"] and lrs and lrs[0] < enc_lr
                                       and r["student_forwards_in_training"] > 0
                                       and not r["student_in_train_mode_any_step"]
                                       and abs(r["groups"]["encoder"]["initial_lr"] - enc_lr) < 1e-12
                                       and abs(r["groups"]["decoder"]["lr_now"] - arm["model"]["lr"]) < 1e-12)))
            _say(f"2. {key}: student tensors moved {r['student_tensors_moved']} of {r['student_tensors']}, others "
                 f"moved {r['other_tensors_moved']} of {r['other_tensors']}; groups "
                 + ", ".join(f"{k} {v['initial_lr']:g}" for k, v in r["groups"].items())
                 + f"; encoder lr per step {[f'{x:.2e}' for x in lrs]}; student in train mode in any of its "
                 f"{r['student_forwards_in_training']} training forwards: {r['student_in_train_mode_any_step']} -> "
                 f"{'passed' if r['passed'] else 'FAILED'}")
            _say(f"3. {key}: {r['s_per_train_pixel'] * 1e6:.2f} us per labelled training pixel "
                 f"({r['pixels_per_step_median']:,.0f} per step), {r['s_per_val_pixel'] * 1e6:.2f} us per validation "
                 f"pixel, peak {r['peak_gpu_gb']} GB")
            if trainable:
                r["s_per_test_chip_full"] = cp.time_full_chips(arm, chips, split_dir, cache)
                rec["estimate_EE_2021_rev"] = cp.estimate(r["s_per_train_pixel"], r["s_per_val_pixel"],
                                                          r["s_per_test_chip_full"])
                _say(f"3. full-chip prediction {r['s_per_test_chip_full']:.2f} s per test chip; the cells on "
                     f"EE_2021_rev at this speed: {rec['estimate_EE_2021_rev']}")
        rec[key] = r
    rec["raster_cache"] = str(cache)
    rec["passed"] = all(v.get("passed", True) for v in rec.values() if isinstance(v, dict))
    return rec


# ------------------------------------------------------------------------------------------------- 5. predictions

def check_predictions(out_root: Path, arms: list[str] | None) -> int:
    bad = 0
    for npz in sorted(out_root.glob("*/P*_draw*_seed*/predictions_test.npz")):
        arm = npz.parent.parent.name
        if arms and arm not in arms:
            continue
        r = json.loads((npz.parent / "results.json").read_text())
        p = np.load(npz)
        sel = p["mask"] >= 0
        acc = float((p["pred"][sel] == p["mask"][sel]).mean())
        logged = r["test_metrics"]["test/Pixel_Accuracy"]
        ok = abs(acc - logged) < 1e-3
        bad += not ok
        print(f"  {arm:<26} {npz.parent.name:<18} predictions {acc:.5f}  logged {logged:.5f}  "
              f"{'ok' if ok else 'MISMATCH'}  ({r.get('workflow', 'core')})")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-e", "--experiment", type=Path, required=True)
    ap.add_argument("--chips", type=Path, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--machine", default="hub", choices=["hub", "cluster"])
    ap.add_argument("--cache-root", type=Path, default=None, help="ignored; the run.py command line")
    ap.add_argument("--scratch", type=Path, default=None, help="where the equivalence cache is written")
    ap.add_argument("--predictions", action="store_true", help="check the finished cells' predictions instead")
    ap.add_argument("--skip", nargs="*", default=[], choices=["equivalence", "frozen", "trainable"])
    ap.add_argument("--batches", type=int, default=16, help="training micro-batches of the switch and cost fit")
    args = ap.parse_args()

    import run as entry

    entry.apply_overrides()
    from gfm4agri.pipeline.config import (arm_dirs, encode_batch_size, load_arm, load_experiment, load_machine,
                                          resolve_split)

    exp = load_experiment(args.experiment)
    chips = args.chips if args.chips.is_absolute() else Path.cwd() / args.chips
    out_root = REPO / "results" / exp["name"] / chips.name
    if args.predictions:
        bad = check_predictions(out_root, args.arms)
        raise SystemExit(f"{bad} cells do not reproduce their logged accuracy" if bad else 0)

    split_dir = resolve_split(chips, args.split)
    mach = load_machine(args.machine)
    search = arm_dirs(args.experiment)
    arms = args.arms or list(exp["arms"])
    failed = []
    for name in arms:
        arm = load_arm(name, search=search)
        if arm.get("route") not in entry.ROUTES:
            _say(f"{name}: not an arm of this folder's routes, nothing to check")
            continue
        import torch

        _say(f"{name} on {chips.name}, {args.machine} ({torch.cuda.get_device_name(0)})")
        rec = {"arm": name, "chips": chips.name, "machine": args.machine, "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0), "created": dt.datetime.now().isoformat(timespec="seconds")}
        if arm["route"] == "labelled_pixels":
            rec.update(check_tessera(arm, chips, split_dir, mach, args))
            failed += [] if rec["passed"] else [name]
            dest = out_root / "checks" / f"{name}_{args.machine}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(rec, indent=2))
            _say(f"{name}: {'all checks passed' if rec['passed'] else 'NOT all passed'} -> {dest.relative_to(REPO)}")
            continue
        if "equivalence" not in args.skip:
            tmp = Path(tempfile.mkdtemp(prefix="ftcheck_", dir=args.scratch))
            try:
                rec["equivalence"] = check_equivalence(arm, chips, split_dir, tmp,
                                                       encode_batch_size(arm, mach, args.machine),
                                                       mach["encode_num_workers"])
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        for trainable in (False, True):
            key = "trainable" if trainable else "frozen"
            if key in args.skip:
                continue
            try:
                r = run_steps(arm, chips, split_dir, trainable=trainable, batches=args.batches,
                              val_batches=8, workers=mach["num_workers"])
            except torch.cuda.OutOfMemoryError as err:
                rec[key] = {"out_of_memory": str(err).split("\n")[0]}
                _say(f"2. {key}: out of memory on this card")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            enc_lr = arm["model"]["encoder_lr"]
            lrs = [x for x in r["encoder_lr_per_step"] if x is not None]
            r["passed"] = bool(
                (r["encoder_tensors_moved"] > 0 if trainable else r["encoder_tensors_moved"] == 0)
                and r["other_tensors_moved"] > 0 and r["groups_cover_trainable"]
                and r["encoder_forwards_in_training"] > 0 and not r["encoder_in_train_mode_any_step"]
                and (not trainable or (r["encoder_group_is_encoder"] and lrs and lrs[0] < enc_lr
                                       and abs(r["groups"]["encoder"]["initial_lr"] - enc_lr) < 1e-12
                                       and abs(r["groups"]["decoder"]["lr_now"] - arm["model"]["lr"]) < 1e-12)))
            rec[key] = r
            _say(f"2. {key}: encoder tensors moved {r['encoder_tensors_moved']} of {r['encoder_tensors']}, "
                 f"others moved {r['other_tensors_moved']} of "
                 f"{r['other_tensors']}; groups " + ", ".join(f"{k} {v['initial_lr']:g}"
                                                              for k, v in r["groups"].items())
                 + f"; encoder lr per step {[f'{x:.2e}' for x in lrs]}; encoder in train mode in any of its "
                 f"{r['encoder_forwards_in_training']} training forwards: {r['encoder_in_train_mode_any_step']} -> "
                 f"{'passed' if r['passed'] else 'FAILED'}")
            _say(f"3. {key}: {r['s_per_train_chip']:.3f} s per training chip, {r['s_per_val_chip']:.3f} s per "
                 f"validation chip, peak {r['peak_gpu_gb']} GB")
        tr = rec.get("trainable", {})
        if tr.get("s_per_train_chip"):
            rec["estimate_EE_2021_rev"] = estimate(tr["s_per_train_chip"], tr["s_per_val_chip"])
            _say(f"3. the cells on EE_2021_rev at this speed: {rec['estimate_EE_2021_rev']}")
        ok = all(v.get("passed", True) for k, v in rec.items() if isinstance(v, dict) and "passed" in v)
        rec["passed"] = ok and all("out_of_memory" not in v for v in rec.values() if isinstance(v, dict))
        failed += [] if ok else [name]
        dest = out_root / "checks" / f"{name}_{args.machine}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(rec, indent=2))
        _say(f"{name}: {'all checks passed' if rec['passed'] else 'NOT all passed'} -> {dest.relative_to(REPO)}")
    if failed:
        raise SystemExit(f"checks failed for {failed}")


if __name__ == "__main__":
    main()
