"""Run a K-shot experiment on one chip set: every arm x budget x draw x seed.

Per arm: check that the chip set has what the arm reads; for a cache-route arm, reuse its
feature cache or compute it (:mod:`gfm4agri.pipeline.cache`); then fit every cell that is not
finished, write its test predictions and stamp its provenance. A cell is finished when it
holds ``results.json`` and ``predictions_test.npz``; a cell with only the first gets its
predictions without a refit, which a job-scoped cache needs because two-stage prediction reads
the cache. Results land in ``<results_root>/<experiment>/<chip set>/<arm>/<cell>/``.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gfm4agri.pipeline.config import (CONFIGS, REPO, compose, load_arm, load_experiment,
                                      load_machine, plan_cells, repo_relative, resolve_split)

__all__ = ["PREDICTIONS", "check_chipset", "run", "summarise"]

PREDICTIONS = "predictions_test.npz"
METRICS = {"test/F1_Score": "macro_f1", "test/mIoU": "miou",
           "test/Pixel_Accuracy": "pixel_accuracy", "test/loss": "test_loss"}


def _say(msg: str) -> None:
    print(f"[kshot {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _lists(chips: Path, split_dir: Path) -> dict[str, list[str]]:
    from gfm4agri.benchmark.segmentation_data import split_list

    out = {}
    for s in ("train", "val", "test"):
        f = split_list(chips, s, split_dir)
        out[s] = f.read_text().split() if f.exists() else []
    return out


def check_chipset(chips: Path, split_dir: Path, arms: dict[str, dict]) -> list[str]:
    """What the arms would read and the chip set does not hold; empty when all is there."""
    from gfm4agri.benchmark.backbones import get_backbone

    problems = [f"{chips / f} missing" for f in ("manifest.json", "chip_parcels.parquet")
                if not (chips / f).exists()]
    lists = _lists(chips, split_dir)
    if not lists["test"]:
        problems.append(f"{split_dir} has no test list")
    ids = sorted({c for v in lists.values() for c in v})
    present = set(os.listdir(chips / "chips")) if (chips / "chips").is_dir() else set()
    for name, arm in arms.items():
        spec = get_backbone(arm["model"]["backbone"])
        if spec.representation == "s2_monthly":
            suffixes = ["_merged.tif"]
        else:
            sidecar = chips / f"{spec.representation}.json"
            if not sidecar.exists():
                problems.append(f"{name}: {sidecar.name} missing")
                continue
            suffixes = [json.loads(sidecar.read_text())["file_suffix"]]
        for sfx in suffixes + [".mask.tif", ".parcels.tif"]:
            miss = [c for c in ids if c + sfx not in present]
            if miss:
                problems.append(f"{name}: {len(miss)} chips lack {sfx}, e.g. {miss[:3]}")
    return problems


def _state(cell_dir: Path) -> str:
    if (cell_dir / "results.json").exists():
        return "done" if (cell_dir / PREDICTIONS).exists() else "predict"
    return "fit"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.strip()


def _stamp(results_path: Path, provenance: dict) -> None:
    r = json.loads(results_path.read_text())
    r["provenance"] = provenance
    results_path.write_text(json.dumps(r, indent=2))


def _predict(cell_dir: Path, test_ids: list[str], num_workers: int) -> None:
    import torch

    from gfm4agri.benchmark.evaluation import FitPredictor

    p = FitPredictor(cell_dir)
    out = p.predict(test_ids, num_workers=num_workers)
    keys = sorted(out)
    tmp = cell_dir / (PREDICTIONS + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, ids=np.array(keys), pred=np.stack([out[k][0] for k in keys]),
                            mask=np.stack([out[k][1] for k in keys]))
    tmp.replace(cell_dir / PREDICTIONS)
    del p
    gc.collect()
    torch.cuda.empty_cache()


def _epochs(cell_dir: Path) -> tuple[int | None, int | None]:
    """Epoch of the lowest validation loss and of the highest validation Macro-F1."""
    logs = sorted(cell_dir.glob("logs/version_*/metrics.csv"))
    if not logs:
        return None, None
    m = pd.read_csv(logs[-1])
    best = []
    for col, pick in (("val/loss", "idxmin"), ("val/F1_Score", "idxmax")):
        v = m.dropna(subset=[col]) if col in m else None
        best.append(None if v is None or v.empty else int(v.loc[getattr(v[col], pick)(), "epoch"]))
    return best[0], best[1]


def _row(results_path: Path) -> dict:
    r = json.loads(results_path.read_text())
    lb, m, prov = r["label_budget"], r["test_metrics"], r.get("provenance", {})
    ckpt, best_f1 = _epochs(results_path.parent)
    return {"arm": r["run"], "pct": lb["pct"], "draw": lb["draw_seed"], "seed": r["seed"],
            **{v: m.get(k) for k, v in METRICS.items()},
            "train_chips": r["chips"]["train"],
            "parcels": sum(v["drawn"] for v in (lb.get("support") or {}).values()),
            "ckpt_epoch": ckpt, "best_val_f1_epoch": best_f1,
            "fit_h": round(r["fit_seconds"] / 3600, 3),
            "machine": prov.get("machine"), "git_commit": prov.get("git_commit")}


def summarise(out_root: Path) -> pd.DataFrame:
    """One row per finished cell under ``out_root``, written to ``summary.csv``."""
    rows = [_row(p) for p in sorted(Path(out_root).glob("*/P*_draw*_seed*/results.json"))]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["arm", "pct", "draw", "seed"], ascending=[True, False, True, True])
    tmp = Path(out_root) / f"summary.csv.{os.getpid()}"   # jobs of other arms may write at once
    df.to_csv(tmp, index=False)
    tmp.replace(Path(out_root) / "summary.csv")
    return df


def run(experiment: Path, chips: Path, *, split: str | None = None,
        arms: list[str] | None = None, machine: str = "hub", cache_root: Path | None = None,
        scratch: Path | None = None, dry_run: bool = False, configs: Path = CONFIGS,
        results_root: Path = REPO / "results") -> list[dict]:
    exp = load_experiment(experiment)
    mach = load_machine(machine, configs)
    chips = Path(chips) if Path(chips).is_absolute() else Path.cwd() / chips
    split_dir = resolve_split(chips, split)
    cells = plan_cells(exp, arms)
    arm_cfgs = {a: load_arm(a, configs) for a in dict.fromkeys(c.arm for c in cells)}
    problems = check_chipset(chips, split_dir, arm_cfgs)
    if problems:
        raise SystemExit("chip set check failed:\n  " + "\n  ".join(problems))

    out_root = Path(results_root) / exp["name"] / chips.name
    root = Path(cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    scratch = Path(scratch) if scratch else root
    test_ids = _lists(chips, split_dir)["test"]
    provenance = {"experiment": exp["name"], "chip_set": repo_relative(chips),
                  "split": split_dir.name, "git_commit": _git("rev-parse", "HEAD"),
                  "git_dirty": bool(_git("status", "--porcelain")), "machine": machine,
                  "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), {len(cells)} cells, "
         f"machine {machine}, commit {provenance['git_commit'][:8]}"
         + (" (dirty tree)" if provenance["git_dirty"] else ""))

    rows = []
    for arm_name, arm in arm_cfgs.items():
        mine = [c for c in cells if c.arm == arm_name]
        todo = [c for c in mine if _state(out_root / arm_name / c.tag) != "done"]
        backbone = arm["model"]["backbone"]
        decision = None
        if arm["route"] == "cache":
            from gfm4agri.pipeline.cache import resolve_cache

            decision = resolve_cache(root, scratch, backbone, chips, split_dir,
                                     arm["data"]["normalisation"], arm["trainer"]["precision"])
            _say(f"{arm_name}: cache {decision.action} {decision.cache_dir} ({decision.reason})"
                 + "".join(f"\n    {m}" for m in decision.mismatches))
            if decision.action == "stop" and todo:
                raise SystemExit(f"{arm_name}: {decision.cache_dir} does not match the run and "
                                 "would be written into; pass --scratch to compute elsewhere")
        _say(f"{arm_name}: {len(mine) - len(todo)} of {len(mine)} cells finished")
        if dry_run:
            rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                      "cache": decision.action if decision else None} for c in mine]
            continue
        if todo and decision is not None and decision.action == "compute":
            from gfm4agri.benchmark.cached import generate_embeddings

            _say(f"{arm_name}: encoding into {decision.cache_dir}")
            generate_embeddings(backbone, chips, decision.cache_dir,
                                normalisation=arm["data"]["normalisation"],
                                batch_size=mach["encode_batch_size"][backbone],
                                num_workers=mach["encode_num_workers"],
                                precision=arm["trainer"]["precision"], split_dir=split_dir)
            gc.collect()
        for c in todo:
            cell_dir = out_root / arm_name / c.tag
            if _state(cell_dir) == "fit":
                cfg = compose(arm_name, arm, exp, mach, chips, split_dir, c, out_root)
                _say(f"{arm_name} {c.tag}: fit")
                if arm["route"] == "cache":
                    from gfm4agri.benchmark.cached import fit_cached

                    fit_cached(cfg, decision.cache_dir, pct=c.pct, draw_seed=c.draw,
                               output_root=out_root, num_workers=mach["num_workers"])
                else:
                    from gfm4agri.benchmark.fit import fit_end_to_end

                    fit_end_to_end(cfg)
                _stamp(cell_dir / "results.json", provenance)
                gc.collect()
            _say(f"{arm_name} {c.tag}: test predictions")
            _predict(cell_dir, test_ids, mach["num_workers"])
        rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                  "cache": decision.action if decision else None} for c in mine]
    if not dry_run:
        df = summarise(out_root)
        _say(f"summary of {len(df)} cells -> {out_root / 'summary.csv'}")
    return rows
