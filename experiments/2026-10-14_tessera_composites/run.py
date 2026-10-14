"""Entry script of stage 2: every TESSERA v2 cell of ``experiment.yaml`` on one chip set, frozen or fine-tuned.

The command line is that of ``scripts/run_kshot.py``, so ``scripts/cluster/submit.sh`` runs this file unchanged:

    poetry run python experiments/2026-10-14_tessera_composites/run.py -e experiments/2026-10-14_tessera_composites/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini [--dry-run]

Per arm (``route: tessera_pixel``, in ``arms_run/``, out of the core test's reach since no core backbone serves it): check the chip set; build or reuse the pixel store under
``<cache root>/tessera_pixels/<chip set>/`` (the job's ``--scratch`` on the cluster); for a frozen arm, build or
reuse the embedding cache beside it; then fit every unfinished cell with ``fit_pixel.fit_cell`` and stamp its
provenance as the core runner does. ``summary.csv`` comes from the core's ``summarise``. The core is not edited.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import yaml  # noqa: E402

WEIGHTS = REPO / "data/weights/tessera/v2"


def _say(msg: str) -> None:
    print(f"[tessera {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def load_arm(name: str) -> dict:
    path = HERE / "arms_run" / f"{name}.yaml"
    arm = yaml.safe_load(path.read_text())
    if arm.get("route") != "tessera_pixel":
        raise ValueError(f"{path}: route {arm.get('route')!r}; this entry runs route 'tessera_pixel' only")
    arm["_file"], arm["_name"] = str(path), name
    return arm


def check_chipset(chips: Path, split_dir: Path) -> list[str]:
    from gfm4agri.benchmark.segmentation_data import split_list

    problems = [f"{chips / f} missing" for f in ("manifest.json", "chip_parcels.parquet") if not (chips / f).exists()]
    ids = []
    for s in ("train", "val", "test"):
        f = split_list(chips, s, split_dir)
        ids += f.read_text().split() if f.exists() else []
        if not f.exists():
            problems.append(f"{f} missing")
    present = set(os.listdir(chips / "chips")) if (chips / "chips").is_dir() else set()
    for sfx in ("_merged.tif", "_s1rtc.tif", ".mask.tif", ".parcels.tif"):
        miss = [c for c in ids if c + sfx not in present]
        if miss:
            problems.append(f"{len(miss)} chips lack {sfx}, e.g. {miss[:3]}")
    return problems


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
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment, load_machine, plan_cells, repo_relative, resolve_split
    from gfm4agri.pipeline.runner import _git, _stamp, _state, summarise

    exp = load_experiment(args.experiment)
    chips = args.chips if args.chips.is_absolute() else Path.cwd() / args.chips
    out_root = REPO / "results" / exp["name"] / chips.name
    if args.summarise:
        print(summarise(out_root).to_string(index=False))
        return
    mach = load_machine(args.machine)
    split_dir = resolve_split(chips, args.split)
    cells = plan_cells(exp, args.arms)
    arms = {a: load_arm(a) for a in dict.fromkeys(c.arm for c in cells)}
    problems = check_chipset(chips, split_dir)
    if problems:
        raise SystemExit("chip set check failed:\n  " + "\n  ".join(problems))
    root = Path(args.scratch or args.cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    store = root / "tessera_pixels" / chips.name
    provenance = {"experiment": exp["name"], "experiment_file": repo_relative(args.experiment),
                  "entry": repo_relative(sys.argv[0]),
                  "arm_files": {a: repo_relative(c["_file"]) for a, c in arms.items()},
                  "chip_set": repo_relative(chips), "split": split_dir.name,
                  "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
                  "machine": args.machine, "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), {len(cells)} cells, machine {args.machine}, "
         f"commit {provenance['git_commit'][:8]}" + (" (dirty tree)" if provenance["git_dirty"] else ""))
    _say(f"pixel store {store}")
    if args.dry_run:
        for c in cells:
            print(f"  {c.arm:<22} {c.tag:<18} {_state(out_root / c.arm / c.tag)}")
        return

    import torch

    from fit_pixel import fit_cell
    from pixels import embedding_cache, open_store
    from tessera_v2 import load_student

    rec = open_store(chips, split_dir, store)
    _say("pixel store: " + ", ".join(f"{s} {n:,}" for s, n in rec["pixels"].items()))
    for name, arm in arms.items():
        todo = [c for c in cells if c.arm == name and _state(out_root / name / c.tag) != "done"]
        _say(f"{name}: {len(todo)} cells to fit")
        if not todo:
            continue
        m = arm["model"]
        cache = None
        if not m["trainable"]:
            model = load_student(m["student"], WEIGHTS, m["checkpoint_sha256"]).to(torch.device("cuda"))
            cache = embedding_cache(store, model, m["student"], m["checkpoint_sha256"], arm["train"]["precision"],
                                    torch.device("cuda"))
            _say(f"{name}: embedding cache {cache}")
            del model
            torch.cuda.empty_cache()
        for c in todo:
            cell_dir = out_root / name / c.tag
            _say(f"{name} {c.tag}: fit")
            r = fit_cell(arm, chips=chips, split_dir=split_dir, store=store, cache=cache, weights=WEIGHTS,
                         cell_dir=cell_dir, pct=c.pct, draw=c.draw, seed=c.seed, epochs=int(exp["epochs"]),
                         store_rec=rec)
            _stamp(cell_dir / "results.json", provenance)
            _say(f"{name} {c.tag}: test Macro-F1 {r['test_metrics']['test/F1_Score']:.4f}, best epoch "
                 f"{r['best_epoch']}, {r['fit_seconds'] / 60:.1f} min")
            gc.collect()
            torch.cuda.empty_cache()
    df = summarise(out_root)
    _say(f"summary of {len(df)} cells -> {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
