"""Entry script: every arm x budget x draw x seed of ``experiment.yaml`` on one chip set, each arm with both heads.

Adapted from ``../2026-11-07_prue_crop_tessera/run.py``. It has the command line of scripts/run_kshot.py, so
``scripts/cluster/submit.sh`` runs it as one Slurm job per arm. An arm file (``arms_run/<arm>.yaml``, outside
``arms/`` because the core's loader cannot read this route) names the main workflow's arm it extends
(``kshot_arm``, read from ``configs/arms/`` for its backbone and its settings) and the input route:

* ``raster``: a 10 m embedding. TESSERA v2's comes from the core's frozen raster cache, reused or computed into the
  scratch root as the K-shot runner does (``rasters.frozen``); AlphaEarth's from the chip set, with its sidecar's
  statistics.
* ``stack``: the baseline's monthly S2 + S1 composites from the chip set, under the chip set's statistics.
* ``token``: the core's feature cache of the encoder (``gfm4agri.pipeline.cache.resolve_cache``, computed by
  ``gfm4agri.benchmark.cached.generate_embeddings`` when missing, every D4 variant of the training chips), and the
  core's trainable stack for the arm (``build_cached_model``).

Every unfinished cell is fitted (``fit_borders.fit``), predicted and scored (``score_joint``: the crop types as the
main workflow, the fields as the PRUE experiment, the borders as the border-accuracy experiment); a finished cell
(``results.json`` and ``predictions_test.npz``) is skipped, and a cell with a ``last.pt`` resumes. Results:
``results/<name>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/``, with ``summary.csv``.

    poetry run python experiments/2026-11-09_borders_all_arms/run.py -e experiments/2026-11-09_borders_all_arms/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
    poetry run python experiments/2026-11-09_borders_all_arms/run.py -e ... --chips ... --summarise     # rescore, collect
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

INPUTS = ("raster", "stack", "token")
WORKFLOW = {"raster": "PRUE's U-Net (EfficientNet-B7, ImageNet) with a border head and a crop head, on a frozen 10 m "
                      "embedding",
            "stack": "PRUE's U-Net (EfficientNet-B7, ImageNet) with a border head and a crop head, trained end to end "
                     "on the monthly S2 + S1 composites",
            "token": "the main workflow's token decoder with a fifth block to 10 m, a border head and a crop head, on "
                     "frozen cached encoder features"}


def _say(msg: str) -> None:
    print(f"[borders {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def load(spec: Path) -> tuple[dict, dict]:
    import yaml

    from gfm4agri.pipeline.config import load_arm

    exp = yaml.safe_load(spec.read_text())
    exp.setdefault("name", spec.parent.name if spec.stem == "experiment" else spec.stem)
    arms = {}
    for a in exp["arms"]:
        f = spec.parent / "arms_run" / f"{a}.yaml"
        arm = yaml.safe_load(f.read_text())
        if arm.get("input") not in INPUTS:
            raise SystemExit(f"{f}: input must be one of {INPUTS}")
        arm["_file"] = _rel(f)
        arm["_kshot"] = load_arm(arm["kshot_arm"])
        arms[a] = arm
    return exp, arms


def source_for(arm: dict, chips: Path, split_dir: Path, root: Path, scratch: Path, mach: dict, machine: str,
               dry_run: bool = False) -> dict:
    """What the arm's chips are read from: a description the fit and the loaders use."""
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.segmentation_data import load_manifest
    from gfm4agri.pipeline.config import encode_batch_size

    core = arm["_kshot"]
    backbone = core["model"]["backbone"]
    spec = get_backbone(backbone)
    if arm["input"] == "raster":
        if spec.computed:
            import rasters

            r = rasters.frozen(chips, split_dir, root, scratch, encode_batch_size(core, mach, machine),
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
        import shutil

        Path(dec.cache_dir).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dec.cache_dir).free / 1e9
        need = float(arm.get("cache_gb", 0))
        if free < need:
            raise SystemExit(f"{backbone}: {free:.0f} GB free under {dec.cache_dir}, the cache needs about {need:.0f} GB")
        _say(f"{backbone}: encoding into {dec.cache_dir} ({dec.reason}; {free:.0f} GB free, about {need:.0f} GB needed)")
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
                out_root: Path) -> dict:
    return {"run_name": arm_name, "workflow": WORKFLOW[arm["input"]], "input": arm["input"],
            "kshot_arm": arm["kshot_arm"], "kshot_arm_file": _rel(arm["_kshot"]["_file"]),
            "backbone": arm["_kshot"]["model"]["backbone"], "seed": seed, "epochs": int(exp["epochs"]),
            "label_budget": {"mode": "pct", "pct": pct, "draw_seed": draw},
            "data": {"root": _rel(chips), "split": _rel(split_dir), **arm["data"]},
            "unet": arm.get("unet"), "token": arm.get("token"), "crop": arm["crop"], "optim": arm["optim"],
            "loss": exp["loss"], "balance": exp["balance"], "worldcover": exp["worldcover"],
            "output_root": _rel(out_root)}


def score_cell(cell: Path, chips: Path, worldcover: Path, workers: int) -> dict:
    import score_joint

    f = cell / "results.json"
    r = json.loads(f.read_text())
    s = score_joint.score_file(cell / "predictions_test.npz", chips / "chips", r["class_names"], worldcover,
                               workers=workers, per_chip_out=cell / "borders_per_chip.npz")
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


def summarise(out_root: Path) -> None:
    import pandas as pd

    rows = []
    for f in sorted(out_root.glob("*/P*_draw*_seed*/results.json")):
        r = json.loads(f.read_text())
        rows.append({"arm": r["run"], "pct": r["label_budget"]["pct"], "draw": r["label_budget"]["draw_seed"],
                     "seed": r["seed"], "best_epoch": r["best_epoch"], "best_val_joint": r["best_val_joint"],
                     **headline(r), "final_crop_weight": r.get("final_crop_weight"),
                     "fit_h": round(r["fit_seconds"] / 3600, 2),
                     "git_commit": r.get("provenance", {}).get("git_commit")})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["arm", "pct"], ascending=[True, False])
    tmp = out_root / f"summary.csv.{os.getpid()}"
    df.to_csv(tmp, index=False)
    tmp.replace(out_root / "summary.csv")
    print(df.T.to_string() if len(df) else "no finished cell")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-e", "--experiment", type=Path, required=True)
    ap.add_argument("--chips", type=Path, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--machine", default="hub", choices=["hub", "cluster"])
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise", action="store_true", help="score the finished cells not scored yet; collect")
    ap.add_argument("--rescore", action="store_true", help="with --summarise: score every finished cell again")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_machine, resolve_split

    exp, arms = load(args.experiment)
    if args.arms:
        unknown = set(args.arms) - set(arms)
        if unknown:
            raise SystemExit(f"arms {sorted(unknown)} are not in the experiment")
        arms = {a: arms[a] for a in args.arms}
    chips = args.chips if args.chips.is_absolute() else Path.cwd() / args.chips
    split_dir = resolve_split(chips, args.split)
    mach = load_machine(args.machine)
    out_root = REPO / "results" / exp["name"] / chips.name
    worldcover = REPO / exp["worldcover"]
    workers = int(mach["num_workers"])
    if args.summarise:
        for f in sorted(out_root.glob("*/P*_draw*_seed*/results.json")):
            done = "border_metrics" in json.loads(f.read_text())
            if (f.parent / "predictions_test.npz").exists() and (args.rescore or not done):
                _say(f"scoring {f.parent.parent.name}/{f.parent.name}")
                score_cell(f.parent, chips, worldcover, min(workers, 16))
        summarise(out_root)
        return
    if not worldcover.exists():
        raise SystemExit(f"no WorldCover maps at {worldcover} (experiments/2026-11-05_parcel_extent/fetch_worldcover.py)")
    root = Path(args.cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    scratch = Path(args.scratch) if args.scratch else root
    prov = {"experiment": exp["name"], "experiment_file": _rel(args.experiment), "entry": _rel(__file__),
            "chip_set": _rel(chips), "split": split_dir.name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", str(HERE))), "machine": args.machine,
            "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), arms {list(arms)}, budgets {exp['budgets_pct']}, "
         f"{exp['epochs']} epochs, commit {prov['git_commit'][:8]}" + (" (dirty folder)" if prov["git_dirty"] else ""))

    for arm_name, arm in arms.items():
        cells = [(p, int(d), int(s)) for p in exp["budgets_pct"] for d in exp["draws"] for s in exp["seeds"]]
        source = None
        for pct, draw, seed in cells:
            cell = out_root / arm_name / f"P{pct:g}_draw{draw}_seed{seed}"
            done = (cell / "results.json").exists() and (cell / "predictions_test.npz").exists()
            if source is None and not done:
                source = source_for(arm, chips, split_dir, root, scratch, mach, args.machine, dry_run=args.dry_run)
            state = "done" if done else ("resume" if (cell / "last.pt").exists() else "fit")
            _say(f"{arm_name} {cell.name}: {state}; input {arm['input']} ({arm['kshot_arm']})"
                 + (f", {source.get('cache')} {source.get('dir') or source.get('cache_dir') or ''}" if source else ""))
            if args.dry_run or done:
                continue
            import fit_borders

            cfg = cell_config(arm_name, arm, exp, chips, split_dir, pct, draw, seed, out_root)
            fit_borders.fit(cfg, cell, source, chips, split_dir, worldcover, num_workers=workers,
                            device=args.device, log=_say)
            f = cell / "results.json"
            res = json.loads(f.read_text())
            res["provenance"] = {**prov, "arm_file": arm["_file"]}
            f.write_text(json.dumps(res, indent=2, default=str))
            _say(f"{arm_name} {cell.name}: scoring")
            h = headline(score_cell(cell, chips, worldcover, min(workers, 16)))
            _say(f"{arm_name} {cell.name}: crop pixel / parcel Macro-F1 {h['crop_pixel_macro_f1']:.3f} / "
                 f"{h['crop_parcel_macro_f1']:.3f}; PQ agnostic (grow) {h['pq_agnostic_grow']:.3f}, class-aware "
                 f"{h['pq_grow']:.3f}; border P/R/F1 at 1 px {h['border_precision_d1']:.3f} / "
                 f"{h['border_recall_d1']:.3f} / {h['border_f1_d1']:.3f}")
    if not args.dry_run:
        summarise(out_root)


if __name__ == "__main__":
    main()
