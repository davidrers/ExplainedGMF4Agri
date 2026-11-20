"""The chain from a chip's composites to the crop score of each of its fields, and the replacements the methods share.

The model explained is the two-head TESSERA arm of ``experiments/2026-11-09_borders_all_arms`` at K = 100 %
(``results/2026-11-09_borders_all_arms/EE_2021_rev/tessera_v2_medium/P100_draw0_seed0/best.pt``): the frozen TESSERA
v2 Medium student, run on the twelve monthly composites as the core's raster cache runs it
(:mod:`gfm4agri.embeddings.tessera_v2`), the embedding standardised with the training-chip statistics, and PRUE's
U-Net with the crop head (:class:`gfm4agri.borders.networks.UNetJoint`). Nothing of the core is changed; it is
imported.

**The input of one pixel** is held here as ``X`` of shape ``(12 months, 12 features)``: TESSERA's ten Sentinel-2
bands in its order (B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, reflectance x 10000) and Sentinel-1 VV and VH in dB
(NaN where never observed). TESSERA reads no atmospheric band.

**The score** of a field is the log-probability of a class or a species, averaged over the field's interior pixels
(every pixel whose 8 neighbours lie in the same field, as ``docs/research/xai/2026-11-16_crop_signatures``). A
species' probability is the sum of its classes' (winter and spring wheat are "wheat").

**Two replacements.** *Drop*: a month's Sentinel-2 frame set to 0, or its Sentinel-1 frame to NaN, which TESSERA
reads as no observation and leaves out of the sequence (checked: identical to removing the frame). *Mean*: the
removed cells take the value of the training mean, the mean of the 13 species' means over the labelled pixels of the
training chips (``step0.py``), so that no species, grassland above all, is the reference.

**Scope.** *Field*: only the explained field's own pixels are changed, every other pixel keeps its observed input;
one encoder pass serves every field of the chip (TESSERA encodes pixels independently), and the network runs once
per field. *Chip*: every pixel of the chip is changed, one network pass scores every field.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

CELL = REPO / "results/2026-11-09_borders_all_arms/EE_2021_rev/tessera_v2_medium/P100_draw0_seed0"
ARM_FILE = REPO / "configs/arms_borders/tessera_v2_medium.yaml"
STUDENT_SHA = "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e"
NAME = HERE.name
#: The model's own chip set: its statistics and mean field are those of its training chips, whatever chips are explained.
MODEL_CHIPS = "EE_2021_rev"
YEAR = 2021

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURES = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12", "VV", "VH"]
S2, S1 = slice(0, 10), slice(10, 12)
BAND_GROUPS = {"Visible": [0, 1, 2], "Red-edge": [5, 6, 7], "NIR": [3, 4], "SWIR": [8, 9], "VV": [10], "VH": [11]}
WINDOWS = [(i, i + 3) for i in range(10)]                     # Jan-Mar ... Oct-Dec
MIN_INTERIOR = 20


def window_name(w: tuple[int, int]) -> str:
    return f"{MONTHS[w[0]]}-{MONTHS[w[1] - 1]}"


def results_dir(chips: str = MODEL_CHIPS) -> Path:
    return REPO / "results" / NAME / chips


def species_of_classes(class_names: list[str]) -> tuple[list[str], np.ndarray]:
    """The species of every class (``docs/research/transferability/crop_classes.csv``): ``(species, index per class)``."""
    with open(REPO / "docs/research/transferability/crop_classes.csv") as f:
        table = {r["hcat"]: r["species"] for r in csv.DictReader(f)}
    names = [table[c] for c in class_names]
    species = list(dict.fromkeys(names))
    return species, np.array([species.index(n) for n in names])


# ------------------------------------------------------------------------------------------------------ inputs

def read_chip(chips_dir: Path, chip: str) -> dict:
    """One chip: ``X`` ``(P, 12, 12)`` float32, ``mask`` and ``parcels`` ``(H, W)``."""
    import rasterio

    from gfm4agri.embeddings import tessera_v2 as tv

    man = json.loads((Path(chips_dir).parent / "manifest.json").read_text())
    s2, s1, hw = tv.chip_inputs(chips_dir, chip, man["imagery"]["band_names"], 12)
    with rasterio.open(Path(chips_dir) / f"{chip}.mask.tif") as m, \
         rasterio.open(Path(chips_dir) / f"{chip}.parcels.tif") as p:
        mask, parcels = m.read(1).astype(np.int64), p.read(1).astype(np.int64)
    return {"chip": chip, "X": np.concatenate([s2, s1], axis=2), "mask": mask, "parcels": parcels, "hw": hw}


def chip_fields(mask: np.ndarray, parcels: np.ndarray, min_interior: int = MIN_INTERIOR) -> list[dict]:
    """The labelled fields of a chip with ``min_interior`` interior pixels or more: id, class, pixels, interior."""
    from scipy import ndimage

    same = ndimage.minimum_filter(parcels, 3, mode="nearest") == ndimage.maximum_filter(parcels, 3, mode="nearest")
    flat, lab = parcels.ravel(), mask.ravel()
    inner = same.ravel() & (flat > 0)
    out = []
    for fid in np.unique(flat[flat > 0]):
        pix = np.flatnonzero(flat == fid)
        cls = lab[pix]
        cls = cls[cls >= 0]
        if not len(cls):
            continue
        interior = pix[inner[pix]]
        if len(interior) >= min_interior:
            out.append({"field": int(fid), "cls": int(np.bincount(cls).argmax()), "pixels": pix, "interior": interior})
    return out


# ------------------------------------------------------------------------------------------------------ the model

@dataclass
class Model:
    student: object
    net: object
    means: np.ndarray
    stds: np.ndarray
    device: object
    class_names: list[str]
    species: list[str]
    species_idx: np.ndarray
    doy: np.ndarray


def load_model(device=None, stats: Path | None = None) -> Model:
    """The student, the network of ``best.pt`` and the training-chip statistics of ``step0``."""
    import torch
    import yaml

    from gfm4agri.borders.networks import UNetJoint
    from gfm4agri.embeddings import tessera_v2 as tv

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    student = tv.load_student("medium", tv.WEIGHTS, STUDENT_SHA).to(device).eval()
    for p in student.parameters():
        p.requires_grad_(False)
    arm = yaml.safe_load(ARM_FILE.read_text())
    res = json.loads((CELL / "results.json").read_text())
    names = res["class_names"]
    c = arm["crop"]
    net = UNetJoint(128, len(names), encoder=arm["unet"]["encoder"], weights=None, hidden=c["hidden"],
                    dropout=float(c["dropout"]), head_dropout=float(c["head_dropout"]))
    net.load_state_dict(torch.load(CELL / "best.pt", map_location="cpu", weights_only=False)["net"])
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    st = json.loads(Path(stats or results_dir() / "step0" / "embedding_stats.json").read_text())
    species, sidx = species_of_classes(names)
    return Model(student, net, np.asarray(st["means"], np.float32), np.asarray(st["stds"], np.float32), device,
                 names, species, sidx, tv.month_doy(YEAR))


def encode(model: Model, X: np.ndarray, precision: str = "32", batch: int = 65536) -> np.ndarray:
    """``(P, 128)`` standardised embeddings of ``X`` ``(P, 12, 12)``; a pixel without any input enters as 0."""
    from gfm4agri.embeddings import tessera_v2 as tv

    emb = tv.encode_pixels(model.student, X[..., S2], X[..., S1], model.doy, model.device, precision, batch)
    return ((np.nan_to_num(emb, nan=0.0) - model.means) / model.stds).astype(np.float32)


def crop_logp(model: Model, maps, precision: str = "32"):
    """Log-softmax of the crop head ``(B, K, H, W)`` for standardised embedding maps ``(B, 128, H, W)`` (torch)."""
    import torch

    with torch.autocast(model.device.type, dtype=torch.bfloat16,
                        enabled=precision.startswith("bf16") and model.device.type == "cuda"):
        _, crop = model.net(maps)
    return torch.log_softmax(crop.float(), 1)


def field_scores(model: Model, logp, interior: np.ndarray):
    """``(K + S,)``: the mean over ``interior`` (flat indices) of each class's and each species' log-probability,
    from ``logp`` ``(K, H*W)`` (torch). Differentiable."""
    import torch

    lp = logp[:, torch.as_tensor(interior, device=logp.device)]                       # (K, n)
    sidx = torch.as_tensor(model.species_idx, device=lp.device)
    sp = torch.stack([torch.logsumexp(lp[sidx == s], 0) for s in range(len(model.species))])
    return torch.cat([lp.mean(1), sp.mean(1)])


# ------------------------------------------------------------------------------------------------------ replacements

def load_mean_field(chips: str = MODEL_CHIPS) -> np.ndarray:
    """``(12, 12)``: the species-balanced training mean of every month and feature (``step0``)."""
    return np.load(results_dir(chips) / "step0" / "mean_field.npz")["baseline"].astype(np.float32)


def apply(X: np.ndarray, cells: np.ndarray, op: str, mean: np.ndarray | None = None) -> np.ndarray:
    """``X`` ``(n, 12, 12)`` with the ``cells`` ``(12, 12)`` bool replaced: ``mean`` by the mean field, ``drop`` by
    no observation (whole frames only: all ten Sentinel-2 bands of a month, or both Sentinel-1 bands)."""
    X = X.copy()
    if not cells.any():
        return X
    if op == "mean":
        X[:, cells] = mean[cells]
        return X
    if op != "drop":
        raise ValueError(op)
    s2, s1 = cells[:, S2], cells[:, S1]
    if ((s2.any(1) & ~s2.all(1)) | (s1.any(1) & ~s1.all(1))).any():
        raise ValueError("drop removes whole frames only")
    X[:, s2.all(1), :10] = 0.0
    X[:, s1.all(1), 10:] = np.nan
    return X


def cells_of(months=None, features=None) -> np.ndarray:
    """``(12, 12)`` bool: the given months (all if None) times the given feature indices (all if None)."""
    c = np.zeros((12, 12), bool)
    m = slice(None) if months is None else list(months)
    f = slice(None) if features is None else list(features)
    c[np.ix_(np.arange(12)[m], np.arange(12)[f])] = True
    return c


# ------------------------------------------------------------------------------------------------------ evaluation

class Evaluator:
    """Scores a chip's fields under replacements, with the chip's observed embedding held once.

    ``field(per_field_X)``: each field gets its own replaced input on its own pixels, every other pixel observed; one
    encoder pass over the fields' pixels, one network pass per field. ``chip(X_all)``: the whole chip replaced.
    Both return ``(n_fields, K + S)``.
    """

    def __init__(self, model: Model, chip: dict, fields: list[dict], precision: str = "32",
                 net_batch: int = 8) -> None:
        self.m, self.chip, self.fields = model, chip, fields
        self.precision, self.net_batch = precision, int(net_batch)
        self.h, self.w = chip["hw"]
        self.base = encode(model, chip["X"], precision)                                  # (P, 128)
        self.rows = np.concatenate([f["pixels"] for f in fields])                        # the fields' pixels
        self.span = np.cumsum([0] + [len(f["pixels"]) for f in fields])

    def _maps(self, emb: np.ndarray):
        import torch

        return torch.from_numpy(emb.T.reshape(1, 128, self.h, self.w)).to(self.m.device)

    def chip_scores(self, X: np.ndarray):
        import torch

        with torch.no_grad():
            lp = crop_logp(self.m, self._maps(encode(self.m, X, self.precision)), self.precision)[0]
            lp = lp.reshape(lp.shape[0], -1)
            return torch.stack([field_scores(self.m, lp, f["interior"]) for f in self.fields]).cpu().numpy()

    def field_scores(self, Xf: np.ndarray, emb: np.ndarray | None = None):
        """``Xf`` ``(len(rows), 12, 12)``: the replaced input of every field's pixels, in ``rows`` order."""
        import torch

        emb = encode(self.m, Xf, self.precision) if emb is None else emb
        base = self._maps(self.base)
        out = []
        with torch.no_grad():
            for i0 in range(0, len(self.fields), self.net_batch):
                idx = range(i0, min(i0 + self.net_batch, len(self.fields)))
                maps = base.repeat(len(idx), 1, 1, 1).reshape(len(idx), 128, -1)
                for k, i in enumerate(idx):
                    pix = torch.as_tensor(self.fields[i]["pixels"], device=maps.device)
                    maps[k][:, pix] = torch.from_numpy(emb[self.span[i]:self.span[i + 1]].T).to(maps.device)
                lp = crop_logp(self.m, maps.reshape(len(idx), 128, self.h, self.w), self.precision)
                lp = lp.reshape(len(idx), lp.shape[1], -1)
                out += [field_scores(self.m, lp[k], self.fields[i]["interior"]) for k, i in enumerate(idx)]
        return torch.stack(out).cpu().numpy()

    def field_input(self) -> np.ndarray:
        return self.chip["X"][self.rows]

    def per_field(self, cells_per_field: list[np.ndarray], op: str, mean=None) -> np.ndarray:
        """The fields' pixels, each field with its own replaced ``cells``."""
        X = self.field_input()
        for i, c in enumerate(cells_per_field):
            a, b = self.span[i], self.span[i + 1]
            X[a:b] = apply(X[a:b], c, op, mean)
        return X

    def same_for_all(self, cells: np.ndarray, op: str, mean=None) -> np.ndarray:
        return apply(self.field_input(), cells, op, mean)
