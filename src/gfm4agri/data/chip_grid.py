"""The chip grid, chip selection and label rasterisation for pixel-level segmentation.

Chips are square cells of a regular grid in ETRS89-LAEA (EPSG:3035), the same projection
the spatial blocks of :mod:`gfm4agri.data.blocks` use, anchored on the projection origin and
sized ``CHIP_PX * PIXEL_M`` = 2,240 m. The grid is therefore fixed rather than
data-dependent: a chip identifier names the same ground footprint in every country and in
every rebuild, and a chip nests inside a block whenever the block size is a multiple of
2,240 m.

The labels a chip carries follow the annotation-unit versus inference-unit distinction:

* ``<chip>.mask.tif``, int16, the class index ``0 .. n_classes - 1`` of every pixel whose
  centre falls inside an in-scheme parcel, and ``IGNORE_INDEX`` everywhere else. This is the
  **dense** label, the one a test chip is scored against.
* ``<chip>.parcels.tif``, int32, the parcel identifier under every pixel, ``0`` outside any
  parcel. The **sparse** training mask for a budget of K polygons per class is derived from
  this raster at training time, by keeping the mask only where the parcel identifier is one
  of the K selected polygons, so the chips never have to be re-exported per budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CHIP_CRS",
    "CHIP_M",
    "CHIP_PX",
    "IGNORE_INDEX",
    "PIXEL_M",
    "Chip",
    "apply_vector_aliases",
    "chip_of",
    "rasterise_labels",
    "vector_layer_aliases",
    "score_cells",
    "select_chips",
]

CHIP_CRS = "EPSG:3035"
PIXEL_M = 10.0
CHIP_PX = 224
CHIP_M = CHIP_PX * PIXEL_M
#: Label value excluded from the loss and from every metric.
IGNORE_INDEX = -1


@dataclass(frozen=True)
class Chip:
    """One grid cell, addressed by its column and row in the 2,240 m grid."""

    country: str
    col: int
    row: int

    @property
    def chip_id(self) -> str:
        return f"{self.country}_{self.col:05d}_{self.row:05d}"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` in EPSG:3035 metres."""
        x0, y0 = self.col * CHIP_M, self.row * CHIP_M
        return (x0, y0, x0 + CHIP_M, y0 + CHIP_M)

    @property
    def transform(self):
        from affine import Affine

        xmin, _, _, ymax = self.bounds
        return Affine(PIXEL_M, 0.0, xmin, 0.0, -PIXEL_M, ymax)


def chip_of(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grid column and row of EPSG:3035 coordinates."""
    return (np.floor(np.asarray(x) / CHIP_M).astype(np.int64),
            np.floor(np.asarray(y) / CHIP_M).astype(np.int64))


def vector_layer_aliases(scheme: dict) -> dict[str, str]:
    """``source code -> scheme code`` for the EuroCrops vector layer, from the class scheme.

    The scheme's codes are EuroCropsML's (HCAT2); the vector layer the labels are rasterised
    from is EuroCrops v11 (HCAT3), which renames a few of them. ``vector_layer_aliases`` in
    the scheme lists each scheme code with the vector-layer codes that mean the same crop.
    """
    raw = scheme.get("vector_layer_aliases") or {}
    return {str(src): str(tgt) for tgt, srcs in raw.items() for src in srcs}


def apply_vector_aliases(codes: pd.Series, lookup: dict[str, str]) -> pd.Series:
    """Vector-layer HCAT codes with every alias replaced by its scheme code."""
    if not lookup:
        return codes
    return codes.astype(str).map(lambda c: lookup.get(c, c))


def score_cells(parcels, class_col: str, classes: list[str]) -> pd.DataFrame:
    """Per grid cell: the in-scheme labelled area share and the number of distinct classes.

    ``parcels`` is a GeoDataFrame in EPSG:3035. A parcel is assigned to the cell holding its
    representative point, so the share is approximate at cell edges; it is used only to rank
    candidates, never as a label.
    """
    inside = parcels[parcels[class_col].isin(classes)]
    pts = inside.geometry.representative_point()
    col, row = chip_of(pts.x.to_numpy(), pts.y.to_numpy())
    df = pd.DataFrame({"col": col, "row": row, "area": inside.geometry.area.to_numpy(),
                       "cls": inside[class_col].to_numpy()})
    grass = classes[0]
    g = df.groupby(["col", "row"])
    out = pd.DataFrame({
        "label_share": g["area"].sum() / CHIP_M**2,
        "n_classes": g["cls"].nunique(),
        "n_parcels": g.size(),
        "grass_share": df.assign(a=np.where(df["cls"] == grass, df["area"], 0.0))
        .groupby(["col", "row"])["a"].sum() / g["area"].sum(),
    }).reset_index()
    return out


def select_chips(cells: pd.DataFrame, country: str, n: int, *, min_label_share: float,
                 min_sep_m: float, seed: int) -> list[Chip]:
    """Pick ``n`` well-labelled, class-diverse cells, pairwise at least ``min_sep_m`` apart.

    Candidates must clear ``min_label_share``; they are ranked by class count and by the share
    of labelled area not taken by the dominant class, with a seeded jitter to break ties, and
    taken greedily subject to the separation constraint. The separation keeps any two chips,
    and hence any training chip and any validation chip, spatially independent.
    """
    rng = np.random.default_rng(seed)
    c = cells[cells["label_share"] >= min_label_share].copy()
    c["rank"] = (c["n_classes"] + 10.0 * c["label_share"] * (1.0 - c["grass_share"])
                 + rng.uniform(0, 0.5, len(c)))
    c = c.sort_values("rank", ascending=False)
    taken: list[tuple[int, int]] = []
    min_sep_cells = min_sep_m / CHIP_M
    for col, row in zip(c["col"], c["row"]):
        if all(np.hypot(col - a, row - b) >= min_sep_cells for a, b in taken):
            taken.append((int(col), int(row)))
            if len(taken) == n:
                break
    if len(taken) < n:
        raise ValueError(f"only {len(taken)} of {n} chips satisfy the selection constraints")
    return [Chip(country, col, row) for col, row in taken]


def rasterise_labels(chip: Chip, parcels, class_col: str, id_col: str,
                     classes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """The dense class mask and the parcel-identifier raster of one chip.

    A pixel is labelled when its centre falls inside a parcel (``all_touched=False``), so a
    pixel straddling a parcel edge is left out rather than guessed. Where two declared
    polygons overlap, the later one in file order wins; the overlap is negligible in the
    declaration layers and is not worth a rule.
    """
    from rasterio.features import rasterize
    from shapely.geometry import box

    shape = (CHIP_PX, CHIP_PX)
    sub = parcels[parcels.intersects(box(*chip.bounds))]
    lut = {c: i for i, c in enumerate(classes)}
    labelled = sub[sub[class_col].isin(lut)]

    mask = np.full(shape, IGNORE_INDEX, dtype=np.int16)
    if len(labelled):
        mask = rasterize(zip(labelled.geometry, labelled[class_col].map(lut)), out_shape=shape,
                         transform=chip.transform, fill=IGNORE_INDEX, dtype="int16")
    ids = np.zeros(shape, dtype=np.int32)
    if len(sub):
        ids = rasterize(zip(sub.geometry, sub[id_col].astype(np.int64)), out_shape=shape,
                        transform=chip.transform, fill=0, dtype="int32")
    return mask, ids
