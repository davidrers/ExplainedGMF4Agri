"""Entry script of the PRUE experiment: every arm x budget x draw x seed of ``experiment.yaml`` on one chip set.

It has the command line of scripts/run_kshot.py, so ``scripts/cluster/submit.sh`` runs it as one Slurm job per arm,
but its cells are not the K-shot runner's: each is PRUE's U-Net fitted by ``fit_prue.fit`` on TESSERA
(``arms_run/<arm>.yaml``, outside ``arms/`` because the core's arm loader cannot read this route). For the frozen
arm the core's raster cache is reused or computed once (``rasters.frozen``); the fine-tuned arm runs the encoder in
the loop and needs no raster. Every unfinished cell is fitted, predicted and scored; a finished cell
(``results.json`` and ``predictions_test.npz``) is skipped, and a cell with a ``last.pt`` resumes. Results:
``results/<name>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/``, with ``summary.csv``.

    poetry run python experiments/2026-11-06_prue_tessera/run.py -e experiments/2026-11-06_prue_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
    poetry run python experiments/2026-11-06_prue_tessera/run.py -e ... --chips ... --summarise     # rescore, collect
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

WORKFLOW = {"frozen": "PRUE U-Net (EfficientNet-B7) on the frozen TESSERA v2 Medium raster",
            "finetuned": "PRUE U-Net (EfficientNet-B7) on TESSERA v2 Medium trained end to end"}
HEADLINE = {"chip/cc": "chip_cc", "chip/grow": "chip_grow", "chip/field": "chip_field", "worldcover/grow": "wc_grow",
            "oracle/grow": "oracle_grow"}


def _say(msg: str) -> None:
    print(f"[prue {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


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

    exp = yaml.safe_load(spec.read_text())
    exp.setdefault("name", spec.parent.name if spec.stem == "experiment" else spec.stem)
    arms = {}
    for a in exp["arms"]:
        f = spec.parent / "arms_run" / f"{a}.yaml"
        arm = yaml.safe_load(f.read_text())
        if arm.get("route") != "prue" or arm.get("encoder") not in WORKFLOW:
            raise SystemExit(f"{f}: route must be 'prue' and encoder one of {sorted(WORKFLOW)}")
        arm["_file"] = _rel(f)
        arms[a] = arm
    return exp, arms


def cell_config(arm_name: str, arm: dict, exp: dict, chips: Path, split_dir: Path, pct, draw: int, seed: int,
                out_root: Path) -> dict:
    return {"run_name": arm_name, "workflow": WORKFLOW[arm["encoder"]], "encoder": arm["encoder"], "seed": seed,
            "epochs": int(exp["epochs"]), "label_budget": {"mode": "pct", "pct": pct, "draw_seed": draw},
            "data": {"root": _rel(chips), "split": _rel(split_dir), **arm["data"]},
            "model": arm["model"], "loss": arm["loss"], "optim": arm["optim"],
            "worldcover": exp["worldcover"], "cropmap": arm.get("cropmap"), "output_root": _rel(out_root)}


def score_cell(cell: Path, chips: Path, worldcover: Path, workers: int) -> dict:
    import score_prue

    f = cell / "results.json"
    r = json.loads(f.read_text())
    cm = r["config"].get("cropmap")
    cm = None if not cm else REPO / cm.format(set=chips.name) / "predictions_test.npz"
    r["panoptic_metrics"] = score_prue.score_file(cell / "predictions_test.npz", chips / "chips", r["class_names"],
                                                  worldcover, cropmap=cm, workers=workers)
    tmp = f.with_name(f"results.json.{os.getpid()}")
    tmp.write_text(json.dumps(r, indent=2))
    tmp.replace(f)
    return r["panoptic_metrics"]


def summarise(out_root: Path) -> None:
    import pandas as pd

    rows = []
    for f in sorted(out_root.glob("*/P*_draw*_seed*/results.json")):
        r = json.loads(f.read_text())
        pm = r.get("panoptic_metrics", {})
        v = pm.get("variants", {})
        row = {"arm": r["run"], "pct": r["label_budget"]["pct"], "draw": r["label_budget"]["draw_seed"],
               "seed": r["seed"], "best_epoch": r["best_epoch"], "best_val_object_f1": r["best_val_object_f1"],
               "fit_h": round(r["fit_seconds"] / 3600, 2),
               "boundary_f1": pm.get("pixels", {}).get("boundary_argmax", {}).get("f1"),
               "field_iou": pm.get("pixels", {}).get("field_iou"),
               "git_commit": r.get("provenance", {}).get("git_commit")}
        for k, tag in HEADLINE.items():
            q = v.get(k, {})
            row[f"pq_agn_{tag}"], row[f"rq_agn_{tag}"] = q.get("pq_agnostic"), q.get("rq_agnostic")
            row[f"pq_{tag}"], row[f"parcels_{tag}"] = q.get("pq"), q.get("predicted_parcels")
        rows.append(row)
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
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--summarise", action="store_true", help="score the finished cells not scored yet; collect")
    ap.add_argument("--rescore", action="store_true", help="with --summarise: score every finished cell again")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_machine, resolve_split

    import rasters

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
            done = "panoptic_metrics" in json.loads(f.read_text())
            if (f.parent / "predictions_test.npz").exists() and (args.rescore or not done):
                _say(f"scoring {f.parent.parent.name}/{f.parent.name}")
                score_cell(f.parent, chips, worldcover, workers)
        summarise(out_root)
        return
    if not worldcover.exists():
        raise SystemExit(f"no WorldCover maps at {worldcover} (experiments/2026-11-05_parcel_extent/fetch_worldcover.py)")
    root = Path(args.cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    scratch = Path(args.scratch) if args.scratch else root
    batch = int(mach["encode_batch_size"].get(rasters.BACKBONE, 16384))
    prov = {"experiment": exp["name"], "experiment_file": _rel(args.experiment), "entry": _rel(__file__),
            "chip_set": _rel(chips), "split": split_dir.name, "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--", str(HERE))), "machine": args.machine,
            "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), arms {list(arms)}, budgets {exp['budgets_pct']}, "
         f"{exp['epochs']} epochs, commit {prov['git_commit'][:8]}" + (" (dirty folder)" if prov["git_dirty"] else ""))

    for arm_name, arm in arms.items():
        cells = [(p, int(d), int(s)) for p in exp["budgets_pct"] for d in exp["draws"] for s in exp["seeds"]]
        raster = None
        for pct, draw, seed in cells:
            cell = out_root / arm_name / f"P{pct:g}_draw{draw}_seed{seed}"
            done = (cell / "results.json").exists() and (cell / "predictions_test.npz").exists()
            if arm["encoder"] == "frozen" and raster is None and not done:
                raster = rasters.frozen(chips, split_dir, root, scratch, batch, dry_run=args.dry_run)
            state = "done" if done else ("resume" if (cell / "last.pt").exists() else "fit")
            info = f", raster {raster[0]} {raster[5]}" if (raster and arm["encoder"] == "frozen") else ""
            _say(f"{arm_name} {cell.name}: {state}{info}")
            if args.dry_run or done:
                continue
            import fit_prue

            cfg = cell_config(arm_name, arm, exp, chips, split_dir, pct, draw, seed, out_root)
            r = None
            if arm["encoder"] == "frozen":
                r = {"dir": raster[0], "suffix": raster[1], "means": raster[2], "stds": raster[3],
                     "record": raster[4]}
            fit_prue.fit(cfg, cell, r, chips, split_dir, worldcover, num_workers=workers, log=_say)
            f = cell / "results.json"
            res = json.loads(f.read_text())
            res["provenance"] = {**prov, "arm_file": arm["_file"]}
            f.write_text(json.dumps(res, indent=2))
            _say(f"{arm_name} {cell.name}: scoring")
            pm = score_cell(cell, chips, worldcover, min(workers, 16))
            v = pm["variants"]
            _say(f"{arm_name} {cell.name}: PQ agnostic " + ", ".join(
                f"{k} {v[k]['pq_agnostic']:.3f} (RQ {v[k]['rq_agnostic']:.3f})" for k in HEADLINE if k in v)
                 + f"; boundary F1 {pm['pixels']['boundary_argmax']['f1']:.3f}")
    if not args.dry_run:
        summarise(out_root)


if __name__ == "__main__":
    main()
