"""Run the two-head workflow on one chip set: every arm x budget x draw x seed, each arm with both heads.

From ``experiments/2026-11-09_borders_all_arms/run.py``, unchanged in behaviour. ``scripts/run_kshot.py`` hands
it a spec with ``workflow: borders`` (``configs/experiments/kshot_borders.yaml``), so ``scripts/cluster/submit.sh``
runs it as one Slurm job per arm, as the crop-only workflow.

An arm file (``arms_borders/<arm>.yaml``, in the spec's folder or else in ``configs/``) names the crop-only arm it
extends (``kshot_arm``, from ``arms/`` likewise: its backbone, normalisation and settings) and its input:

* ``raster``: a 10 m embedding. TESSERA v2's comes from the core's frozen raster cache, reused or computed into the
  scratch root as the crop-only runner does (:mod:`gfm4agri.pipeline.raster_cache`); AlphaEarth's from the chip set,
  with its sidecar's statistics.
* ``stack``: the baseline's monthly S2 + S1 composites from the chip set, under the chip set's statistics.
* ``token``: the core's feature cache of the encoder (:func:`gfm4agri.pipeline.cache.resolve_cache`, computed by
  :func:`gfm4agri.benchmark.cached.generate_embeddings` when missing, every D4 variant of the training chips), and the
  crop-only arm's trainable stack (``build_cached_model``).

Every unfinished cell is fitted (:func:`gfm4agri.borders.fit.fit`), predicted and scored
(:func:`gfm4agri.borders.scoring.score_file`); a finished cell (``results.json`` and ``predictions_test.npz``) is
skipped, and a cell with a ``last.pt`` resumes. The border target needs ESA WorldCover on the chip grid,
``data/worldcover/<chip set>/worldcover.npz`` (``scripts/hub/build_worldcover_chips.py``), or the parent's for a
pilot set. Results: ``<results_root>/<name>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/``, with ``summary.csv``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

from gfm4agri.pipeline.config import (CONFIGS, REPO, arm_dirs, encode_batch_size, load_arm, load_experiment,
                                      load_machine, repo_relative, resolve_split)

__all__ = ["INPUTS", "load", "rescore", "run", "source_for", "summarise", "worldcover_for"]

INPUTS = ("raster", "stack", "token")
WORKFLOW = {"raster": "PRUE's U-Net (EfficientNet-B7, ImageNet) with a border head and a crop head, on a frozen 10 m "
                      "embedding",
            "stack": "PRUE's U-Net (EfficientNet-B7, ImageNet) with a border head and a crop head, trained end to end "
                     "on the monthly S2 + S1 composites",
            "token": "the main workflow's token decoder with a fifth block to 10 m, a border head and a crop head, on "
                     "frozen cached encoder features"}
PREDICTIONS = "predictions_test.npz"
#: Scoring processes: at most this many, so a rescore adds the IoU sums in the order the cluster's jobs did.
SCORING_WORKERS = 16


def _say(msg: str) -> None:
    print(f"[borders {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def load(spec: Path, configs: Path = CONFIGS) -> tuple[dict, dict]:
    """The spec and its arms: each arm file with ``_file`` and its crop-only arm under ``_kshot``."""
    spec = Path(spec)
    exp = load_experiment(spec)
    if exp.get("workflow") != "borders":
        raise ValueError(f"{spec}: not a two-head spec (workflow: {exp.get('workflow')!r})")
    dirs = [d for d in (spec.parent / "arms_borders", Path(configs) / "arms_borders") if d.is_dir()]
    search = arm_dirs(spec, configs)
    arms = {}
    for a in exp["arms"]:
        f = next((d / f"{a}.yaml" for d in dirs if (d / f"{a}.yaml").exists()), None)
        if f is None:
            raise FileNotFoundError(f"two-head arm {a!r} not found in {[str(d) for d in dirs]}")
        arm = yaml.safe_load(f.read_text())
        if arm.get("input") not in INPUTS:
            raise ValueError(f"{f}: input must be one of {INPUTS}")
        arm["_file"] = repo_relative(f)
        arm["_kshot"] = load_arm(arm["kshot_arm"], configs, search)
        arms[a] = arm
    return exp, arms


def worldcover_for(chips: Path) -> Path:
    """``data/worldcover/<chip set>/worldcover.npz``, or the parent chip set's for a pilot set."""
    chips = Path(chips)
    names = [chips.name]
    sub = chips / "subset.json"
    if sub.exists():
        names.append(Path(json.loads(sub.read_text())["parent"]).name)
    for n in names:
        f = REPO / "data" / "worldcover" / n / "worldcover.npz"
        if f.exists():
            return f
    raise SystemExit(f"no WorldCover maps for {chips.name} under {REPO / 'data' / 'worldcover'}: run "
                     f"scripts/hub/build_worldcover_chips.py --chips {names[-1]} on the hub")


def _frozen_raster(backbone: str, chips: Path, split_dir: Path, cache_root: Path, scratch: Path, batch: int,
                   dry_run: bool = False):
    """The core's frozen raster cache of a computed 10 m encoder: ``(dir, suffix, means, stds, record, action)``."""
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.pipeline.raster_cache import build_raster_cache, resolve_raster_cache

    d = resolve_raster_cache(cache_root, scratch, backbone, chips, split_dir)
    if dry_run:
        return d.cache_dir, None, None, None, None, f"{d.action} ({d.reason})"
    if d.action == "stop":
        raise SystemExit(f"{d.cache_dir} does not match the run; pass --scratch")
    if d.action == "compute":
        _say(f"{backbone} raster: encoding into {d.cache_dir} ({d.reason})")
        build_raster_cache(backbone, chips, split_dir, d.cache_dir, batch=batch)
    side = json.loads((d.cache_dir / f"{get_backbone(backbone).representation}.json").read_text())
    record = json.loads((d.cache_dir / "cache.json").read_text())
    return (d.cache_dir, side["file_suffix"], side["normalisation"]["means"], side["normalisation"]["stds"],
            record, d.action)


def source_for(arm: dict, chips: Path, split_dir: Path, root: Path, scratch: Path, mach: dict, machine: str,
               dry_run: bool = False) -> dict:
    """What the arm's chips are read from: a description the fit and the loaders use."""
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.segmentation_data import load_manifest

    core = arm["_kshot"]
    backbone = core["model"]["backbone"]
    spec = get_backbone(backbone)
    if arm["input"] == "raster":
        if spec.computed:
            r = _frozen_raster(backbone, chips, split_dir, root, scratch, encode_batch_size(core, mach, machine),
                               dry_run=dry_run)
            if dry_run:
                return {"kind": "raster", "backbone": backbone, "dir": str(r[0]), "cache": r[5]}
            return {"kind": "raster", "backbone": backbone, "dir": str(r[0]), "suffix": r[1], "means": r[2],
                    "stds": r[3], "channels": len(r[2]), "cache": r[5], "record_created": r[4].get("created")}
        side = json.loads((chips / f"{spec.representation}.json").read_text())
        return {"kind": "raster", "backbone": backbone, "dir": str(chips), "suffix": side["file_suffix"],
                "means": side["normalisation"]["means"], "stds": side["normalisation"]["stds"],
                "channels": len(side["normalisation"]["means"]), "cache": "chip set"}
    if arm["input"] == "stack":
        man = load_manifest(chips)
        s1 = json.loads((chips / "s1_rtc.json").read_text())["normalisation"]
        nb, nm = len(man["imagery"]["band_names"]), int(man["imagery"]["n_months"])
        return {"kind": "stack", "backbone": backbone, "s2_bands": nb, "n_months": nm,
                "s2_means": man["normalisation"]["means"], "s2_stds": man["normalisation"]["stds"],
                "s1_means": s1["means"], "s1_stds": s1["stds"], "channels": (nb + len(s1["means"])) * nm,
                "cache": "chip set"}
    from gfm4agri.benchmark.cached import generate_embeddings
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule
    from gfm4agri.pipeline.cache import resolve_cache

    norm, prec = core["data"]["normalisation"], core["trainer"]["precision"]
    dec = resolve_cache(root, scratch, backbone, chips, split_dir, norm, prec)
    if dry_run:
        return {"kind": "token", "backbone": backbone, "cache_dir": str(dec.cache_dir),
                "cache": f"{dec.action} ({dec.reason})"}
    if dec.action == "stop":
        raise SystemExit(f"{dec.cache_dir} does not match the run; pass --scratch")
    if dec.action == "compute":
        Path(dec.cache_dir).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dec.cache_dir).free / 1e9
        need = float(arm.get("cache_gb", 0))
        if free < need:
            raise SystemExit(f"{backbone}: {free:.0f} GB free under {dec.cache_dir}, the cache needs about "
                             f"{need:.0f} GB")
        _say(f"{backbone}: encoding into {dec.cache_dir} ({dec.reason}; {free:.0f} GB free, about {need:.0f} GB "
             "needed)")
        generate_embeddings(backbone, chips, dec.cache_dir, normalisation=norm,
                            batch_size=encode_batch_size(core, mach, machine),
                            num_workers=mach["encode_num_workers"], precision=prec, split_dir=split_dir)
    cache = json.loads((Path(dec.cache_dir) / "cache.json").read_text())
    probe = EuroCropsSegDataModule(chips, backbone, normalisation=norm, batch_size=1, num_workers=0,
                                   augment=False, split_dir=split_dir)
    return {"kind": "token", "backbone": backbone, "cache_dir": str(dec.cache_dir),
            "channel_list": cache["channel_list"], "feature_size": cache["feature_size"], "bands": probe.bands,
            "n_timesteps": probe.n_timesteps, "cache": dec.action}


def cell_config(arm_name: str, arm: dict, exp: dict, chips: Path, split_dir: Path, pct, draw: int, seed: int,
                out_root: Path, worldcover: Path | None = None) -> dict:
    """The configuration dictionary of one cell, in the layout :func:`gfm4agri.borders.fit.fit` reads."""
    return {"run_name": arm_name, "workflow": WORKFLOW[arm["input"]], "input": arm["input"],
            "kshot_arm": arm["kshot_arm"], "kshot_arm_file": repo_relative(arm["_kshot"]["_file"]),
            "backbone": arm["_kshot"]["model"]["backbone"], "seed": seed, "epochs": int(exp["epochs"]),
            "label_budget": {"mode": "pct", "pct": pct, "draw_seed": draw},
            "data": {"root": repo_relative(chips), "split": repo_relative(split_dir), **arm["data"]},
            "unet": arm.get("unet"), "token": arm.get("token"), "crop": arm["crop"], "optim": arm["optim"],
            "loss": exp["loss"], "balance": exp["balance"],
            "worldcover": None if worldcover is None else repo_relative(worldcover),
            "output_root": repo_relative(out_root)}


def score_cell(cell: Path, chips: Path, worldcover: Path, workers: int) -> dict:
    """Score a fitted cell's test predictions into its ``results.json``; ``borders_per_chip.npz`` beside it."""
    from gfm4agri.borders import scoring

    f = cell / "results.json"
    r = json.loads(f.read_text())
    s = scoring.score_file(cell / PREDICTIONS, chips / "chips", r["class_names"], worldcover, workers=workers,
                           per_chip_out=cell / "borders_per_chip.npz")
    r["crop_metrics"] = s["crop"]["pixel"]
    r["parcel_metrics"] = s["crop"]["parcel"]
    r["panoptic_metrics"] = s["panoptic"]
    r["border_metrics"] = s["borders"]
    tmp = f.with_name(f"results.json.{os.getpid()}")
    tmp.write_text(json.dumps(r, indent=2, default=str))
    tmp.replace(f)
    return r


def headline(r: dict) -> dict:
    """One flat row: the crop map, the fields, the borders."""
    v = r.get("panoptic_metrics", {}).get("variants", {})
    g = v.get("chip/grow", {})
    b = r.get("border_metrics", {}).get("d1", {})
    return {"crop_pixel_macro_f1": r.get("crop_metrics", {}).get("macro_f1"),
            "crop_parcel_macro_f1": r.get("parcel_metrics", {}).get("macro_f1"),
            "pq_agnostic_grow": g.get("pq_agnostic"), "rq_agnostic_grow": g.get("rq_agnostic"),
            "pq_grow": g.get("pq"), "fields_grow": g.get("predicted_parcels"),
            "pq_agnostic_cc": v.get("chip/cc", {}).get("pq_agnostic"),
            "border_precision_d1": b.get("precision"), "border_recall_d1": b.get("recall"),
            "border_f1_d1": b.get("f1"),
            "boundary_pixel_f1": r.get("panoptic_metrics", {}).get("pixels", {}).get("boundary_argmax", {}).get("f1")}


def summarise(out_root: Path):
    """Every finished cell under ``out_root`` as one row; written to ``summary.csv`` and returned."""
    import pandas as pd

    rows = []
    for f in sorted(Path(out_root).glob("*/P*_draw*_seed*/results.json")):
        r = json.loads(f.read_text())
        rows.append({"arm": r["run"], "pct": r["label_budget"]["pct"], "draw": r["label_budget"]["draw_seed"],
                     "seed": r["seed"], "best_epoch": r["best_epoch"], "best_val_joint": r["best_val_joint"],
                     **headline(r), "final_crop_weight": r.get("final_crop_weight"),
                     "fit_h": round(r["fit_seconds"] / 3600, 2),
                     "git_commit": r.get("provenance", {}).get("git_commit")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["arm", "pct"], ascending=[True, False])
    Path(out_root).mkdir(parents=True, exist_ok=True)
    tmp = Path(out_root) / f"summary.csv.{os.getpid()}"
    df.to_csv(tmp, index=False)
    tmp.replace(Path(out_root) / "summary.csv")
    return df


def rescore(out_root: Path, chips: Path, *, machine: str = "hub", every: bool = False,
            configs: Path = CONFIGS) -> list[str]:
    """Score the finished cells that have no border scores yet (``every``: all of them); the cells scored."""
    chips = Path(chips) if Path(chips).is_absolute() else Path.cwd() / chips
    workers = min(int(load_machine(machine, configs)["num_workers"]), SCORING_WORKERS)
    worldcover = None
    done = []
    for f in sorted(Path(out_root).glob("*/P*_draw*_seed*/results.json")):
        scored = "border_metrics" in json.loads(f.read_text())
        if (f.parent / PREDICTIONS).exists() and (every or not scored):
            worldcover = worldcover or worldcover_for(chips)
            _say(f"scoring {f.parent.parent.name}/{f.parent.name}")
            score_cell(f.parent, chips, worldcover, workers)
            done.append(f"{f.parent.parent.name}/{f.parent.name}")
    return done


def run(spec: Path, chips: Path, *, split: str | None = None, arms: list[str] | None = None,
        machine: str = "hub", cache_root: Path | None = None, scratch: Path | None = None,
        device: str | None = None, dry_run: bool = False, configs: Path = CONFIGS,
        results_root: Path = REPO / "results") -> list[dict]:
    """Fit, predict and score every unfinished cell; one row per cell (``arm``, ``cell``, ``state``, ``cache``)."""
    exp, arm_cfgs = load(spec, configs)
    if arms:
        unknown = set(arms) - set(arm_cfgs)
        if unknown:
            raise SystemExit(f"arms {sorted(unknown)} are not in the spec")
        arm_cfgs = {a: arm_cfgs[a] for a in arms}
    chips = Path(chips) if Path(chips).is_absolute() else Path.cwd() / chips
    split_dir = resolve_split(chips, split)
    mach = load_machine(machine, configs)
    out_root = Path(results_root) / exp["name"] / chips.name
    worldcover = worldcover_for(chips)
    workers = int(mach["num_workers"])
    root = Path(cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    scratch = Path(scratch) if scratch else root
    prov = {"experiment": exp["name"], "experiment_file": repo_relative(spec), "workflow": "borders",
            "entry": repo_relative(sys.argv[0]) if sys.argv and sys.argv[0] else None,
            "chip_set": repo_relative(chips), "split": split_dir.name, "worldcover": repo_relative(worldcover),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "machine": machine, "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), arms {list(arm_cfgs)}, budgets {exp['budgets_pct']}, "
         f"{exp['epochs']} epochs, commit {prov['git_commit'][:8]}" + (" (dirty tree)" if prov["git_dirty"] else ""))

    from gfm4agri.borders import fit as fit_cell

    rows = []
    for arm_name, arm in arm_cfgs.items():
        cells = [(p, int(d), int(s)) for p in exp["budgets_pct"] for d in exp["draws"] for s in exp["seeds"]]
        source = None
        for pct, draw, seed in cells:
            cell = out_root / arm_name / f"P{pct:g}_draw{draw}_seed{seed}"
            done = (cell / "results.json").exists() and (cell / PREDICTIONS).exists()
            if source is None and not done:
                source = source_for(arm, chips, split_dir, root, scratch, mach, machine, dry_run=dry_run)
            state = "done" if done else ("resume" if (cell / "last.pt").exists() else "fit")
            cache = source.get("cache") if source else None
            _say(f"{arm_name} {cell.name}: {state}; input {arm['input']} ({arm['kshot_arm']})"
                 + (f", {cache} {source.get('dir') or source.get('cache_dir') or ''}" if source else ""))
            rows.append({"arm": arm_name, "cell": cell.name, "state": state, "cache": cache})
            if dry_run or done:
                continue
            cfg = cell_config(arm_name, arm, exp, chips, split_dir, pct, draw, seed, out_root, worldcover)
            fit_cell.fit(cfg, cell, source, chips, split_dir, worldcover, num_workers=workers, device=device,
                         log=_say)
            f = cell / "results.json"
            res = json.loads(f.read_text())
            res["provenance"] = {**prov, "arm_file": arm["_file"], "kshot_arm_file": cfg["kshot_arm_file"]}
            f.write_text(json.dumps(res, indent=2, default=str))
            _say(f"{arm_name} {cell.name}: scoring")
            h = headline(score_cell(cell, chips, worldcover, min(workers, SCORING_WORKERS)))
            _say(f"{arm_name} {cell.name}: crop pixel / parcel Macro-F1 {h['crop_pixel_macro_f1']:.3f} / "
                 f"{h['crop_parcel_macro_f1']:.3f}; PQ agnostic (grow) {h['pq_agnostic_grow']:.3f}, class-aware "
                 f"{h['pq_grow']:.3f}; border P/R/F1 at 1 px {h['border_precision_d1']:.3f} / "
                 f"{h['border_recall_d1']:.3f} / {h['border_f1_d1']:.3f}")
            rows[-1]["state"] = "done"
    if not dry_run:
        df = summarise(out_root)
        _say(f"summary of {len(df)} cells -> {out_root / 'summary.csv'}")
    return rows
