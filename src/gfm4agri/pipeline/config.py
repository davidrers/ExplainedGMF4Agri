"""Compose the configuration of one K-shot cell from an arm, an experiment, a machine and a chip set.

The fit functions take one dictionary in the layout of the earlier per-run YAML files
(``run_name``, ``seed``, ``data``, ``label_budget``, ``model``, ``trainer``, ``output_root``). The
workflow keeps that layout and builds it from three smaller files, so each is written once:

* ``configs/arms/<arm>.yaml``: what is trained, independent of country and budget;
* ``configs/experiments/<name>.yaml``: which arms, budgets, draws, seeds and epochs;
* ``configs/machines/<machine>.yaml``: throughput settings that never change a result.

An experiment outside the main workflow lives in its own folder, ``experiments/<date>_<name>/``,
as ``experiment.yaml`` with its own arm files in ``arms/``; those are found before the core's
``configs/arms/``, so an experiment never has to change the core to vary an arm.

The chip set is a path. Paths inside the repository are stored relative to it, so a result
reads the same on the JupyterHub and on the cluster, whose clone mirrors the hub's layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs"

__all__ = ["CONFIGS", "REPO", "Cell", "arm_dirs", "compose", "encode_batch_size", "load_arm",
           "load_experiment", "load_machine", "plan_cells", "repo_relative", "resolve_split"]


@dataclass(frozen=True)
class Cell:
    """One fit: an arm at a label budget, a support draw and a training seed."""

    arm: str
    pct: float
    draw: int
    seed: int

    @property
    def tag(self) -> str:
        return f"P{self.pct:g}_draw{self.draw}_seed{self.seed}"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def arm_dirs(experiment: Path, configs: Path = CONFIGS) -> list[Path]:
    """Where an experiment's arms are looked up: its own ``arms/`` folder, then the core's."""
    local = Path(experiment).parent / "arms"
    core = Path(configs) / "arms"
    return [local, core] if local.is_dir() and local.resolve() != core.resolve() else [core]


def load_arm(name: str, configs: Path = CONFIGS, search: list[Path] | None = None) -> dict:
    """``<dir>/<name>.yaml`` from the first of ``search`` holding it (default the core's
    ``configs/arms/``), checked against the backbone registry."""
    from gfm4agri.benchmark.backbones import get_backbone

    dirs = search or [Path(configs) / "arms"]
    path = next((d / f"{name}.yaml" for d in dirs if (d / f"{name}.yaml").exists()), None)
    if path is None:
        raise FileNotFoundError(f"arm {name!r} not found in {[str(d) for d in dirs]}")
    arm = _yaml(path)
    arm["_file"] = str(path)
    spec = get_backbone(arm["model"]["backbone"])
    want = "cache" if spec.representation == "s2_monthly" else "raster"
    if arm["route"] != want:
        raise ValueError(f"arm {name}: route {arm['route']!r}, but {spec.name} needs {want!r}")
    return arm


def load_experiment(path: Path) -> dict:
    """An experiment spec. Its ``name`` names the results folder; without one it is the file's
    stem, or the folder's name for an ``experiment.yaml`` in an experiment folder."""
    exp = _yaml(path)
    stem = Path(path).stem
    exp.setdefault("name", Path(path).parent.name if stem == "experiment" else stem)
    if exp.get("checkpoint", "val_loss") != "val_loss":
        raise ValueError(f"checkpoint {exp['checkpoint']!r}: only 'val_loss' is implemented")
    return exp


def load_machine(name: str, configs: Path = CONFIGS) -> dict:
    return _yaml(Path(configs) / "machines" / f"{name}.yaml")


def encode_batch_size(arm: dict, machine: dict, machine_name: str) -> int:
    """The arm's own ``encode_batch_size: {hub: .., cluster: ..}`` if it sets one, else the
    machine profile's entry for its backbone, else 1."""
    own = arm.get("encode_batch_size") or {}
    if machine_name in own:
        return int(own[machine_name])
    return int(machine.get("encode_batch_size", {}).get(arm["model"]["backbone"], 1))


def plan_cells(exp: dict, arms: list[str] | None = None) -> list[Cell]:
    """Every cell of ``exp``, arm by arm, budgets outermost within an arm."""
    unknown = sorted(set(arms or []) - set(exp["arms"]))
    if unknown:
        raise ValueError(f"arms {unknown} are not in the experiment ({exp['arms']})")
    chosen = [a for a in exp["arms"] if arms is None or a in arms]
    return [Cell(a, p, int(d), int(s)) for a in chosen for p in exp["budgets_pct"]
            for d in exp["draws"] for s in exp["seeds"]]


def repo_relative(path) -> str:
    """``path`` relative to the repository when it lies inside it, else absolute."""
    p = Path(path)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def resolve_split(chips: Path, split: str | None = None) -> Path:
    """The chip set's split directory: the only one, or the one named (with or without hash)."""
    base = Path(chips) / "splits"
    splits = sorted(p for p in base.iterdir() if (p / "split.json").exists()) if base.is_dir() else []
    names = [p.name for p in splits]
    if split is not None:
        match = [p for p in splits if split in (p.name, p.name.split("__")[0])]
        if len(match) != 1:
            raise ValueError(f"split {split!r} not found in {base}: {names}")
        return match[0]
    if len(splits) != 1:
        raise ValueError(f"{chips} holds {len(splits)} splits {names}; name one with --split")
    return splits[0]


def _number(x):
    return int(x) if float(x).is_integer() else float(x)


def compose(arm_name: str, arm: dict, exp: dict, machine: dict, chips: Path, split_dir: Path,
            cell: Cell, output_root: Path) -> dict:
    """The configuration dictionary of one cell, in the layout the fit functions read."""
    d, t = arm["data"], arm["trainer"]
    trainer = {"max_epochs": exp["epochs"], "precision": t["precision"],
               "log_every_n_steps": t["log_every_n_steps"]}
    if "accumulate_grad_batches" in t:
        trainer["accumulate_grad_batches"] = t["accumulate_grad_batches"]
    return {
        "run_name": arm_name,
        "seed": cell.seed,
        "data": {"root": repo_relative(chips), "split": repo_relative(split_dir),
                 "normalisation": d["normalisation"], "batch_size": d["batch_size"],
                 "num_workers": machine["num_workers"], "augment": d["augment"]},
        "label_budget": {"mode": "pct", "pct": _number(cell.pct), "draw_seed": cell.draw},
        "model": copy.deepcopy(arm["model"]),
        "trainer": trainer,
        "output_root": repo_relative(output_root),
    }
