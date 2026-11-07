"""PaPs targets for one chip, built from its labels under the label budget and from WorldCover, with no other truth.

PASTIS ships its instance targets precomputed (``INSTANCE_ANNOTATIONS``); utae-paps holds no builder. This module
builds them from what the paper says (Garnot and Landrieu, ICCV 2021, arXiv 2107.07933, section 3.2):

* **Parcels.** The 4-connected pieces of each *drawn* in-scheme parcel inside the chip, the units the scoring uses
  (``panoptic_x.truth``). Under a budget of K % only the drawn parcels are parcels; at validation and test every
  in-scheme parcel is.
* **Centre.** The paper gives each parcel "a centerpoint with integer coordinates" and supervises the detection with
  its bounding box, without saying which pixel. Here it is the pixel of the piece nearest the piece's centroid, so
  the centre always lies on the parcel and the features sampled there describe it.
* **Heatmap.** At each pixel, the largest of the parcels' Gaussian kernels, with deviations 1/20 of the height and
  width of each parcel's bounding box (equation 6), and exactly 1 at every centre.
* **Zones.** Each pixel goes to the parcel whose kernel is largest there (the argmax of equation 6), computed on
  the log-kernels so that no kernel underflows. Supervision keeps one centre per zone (``supmax``).
* **Size, classes.** Per zone pixel, the height and width of its parcel's bounding box and its parcel's class; per
  pixel, its own class (-1 off the parcels).

**What is known.** PASTIS labels every pixel: parcels, background and void. Here the label budget gives the drawn
parcels only, and the truth is not used to say where the other parcels or the non-agricultural land lie. The only
background is ESA WorldCover's, the pixels it does not call grassland (30) or cropland (40), nor leaves without data
(0). So a pixel is *known* when it lies on a drawn parcel or on WorldCover background outside every drawn parcel;
every other pixel (WorldCover grassland or cropland outside the drawn parcels) is *unknown*: it is no negative of the
heatmap, and its zone is 0, which no centre is taken from. The same rule holds at every budget, K = 100 % included,
and for the validation chips, whose drawn parcels are all their in-scheme parcels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

FOUR = ndimage.generate_binary_structure(2, 1)
#: WorldCover classes that are not background: grassland and cropland.
AGRI = (30, 40)
#: Channels of the target, as utae-paps' plus ``known``.
CHANNELS = ("heatmap", "instance", "zone", "height", "width", "object_class", "pixel_class", "known")
#: Deviation of a kernel, as a fraction of the bounding box (the paper's 1/20).
SIGMA_FRACTION = 1 / 20


def pieces(labels: np.ndarray) -> np.ndarray:
    """``labels`` (0 = none) with every label's 4-connected pieces numbered apart, from 1, 0 kept."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int32)
    vals, inv = np.unique(labels, return_inverse=True)
    dense = inv.reshape(labels.shape).astype(np.int32) + (0 if vals[0] == 0 else 1)
    nxt = 1
    for k, sl in enumerate(ndimage.find_objects(dense), start=1):
        if sl is None:
            continue
        sub = dense[sl] == k
        lab, n = ndimage.label(sub, structure=FOUR)
        if n:
            view = out[sl]
            view[sub] = lab[sub] + (nxt - 1)
            nxt += n
    return out


def build(mask: np.ndarray, parcels: np.ndarray, worldcover: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """``(H, W, 8)`` float32 target of one chip (``CHANNELS``).

    ``mask`` holds the class index of every in-scheme parcel pixel (-1 elsewhere), ``parcels`` the parcel id of every
    declared parcel (0 elsewhere), ``worldcover`` the WorldCover class, ``keep`` the drawn parcel ids (``None``: every
    parcel, as for validation and test chips).
    """
    H, W = mask.shape
    lab = (mask >= 0) & (parcels > 0)
    if keep is not None:
        lab &= np.isin(parcels, keep)
    inst = pieces(np.where(lab, parcels, 0))
    n = int(inst.max())
    out = np.zeros((H, W, len(CHANNELS)), np.float32)
    out[..., 1] = inst
    out[..., 6] = np.where(lab, mask, -1)
    background = ~np.isin(worldcover, AGRI) & (worldcover != 0) & ~lab
    known = lab | background
    if n == 0:
        out[..., 7] = known
        return out

    rows, cols = np.indices((H, W))
    objs = ndimage.find_objects(inst)
    best = np.full((H, W), -np.inf, np.float32)
    zone = np.zeros((H, W), np.int32)
    cls = np.zeros(n + 1, np.int64)
    hw = np.zeros((n + 1, 2), np.float32)
    centres = []
    for k, sl in enumerate(objs, start=1):
        sub = inst[sl] == k
        h = int(sub.any(axis=1).sum())
        w = int(sub.any(axis=0).sum())
        rr, cc = np.nonzero(sub)
        rr, cc = rr + sl[0].start, cc + sl[1].start
        d = (rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2
        i = int(np.argmin(d))
        ci, cj = int(rr[i]), int(cc[i])
        centres.append((ci, cj))
        sv, sh = SIGMA_FRACTION * h, SIGMA_FRACTION * w
        logk = -(((rows - ci) ** 2) / (2 * sv ** 2) + ((cols - cj) ** 2) / (2 * sh ** 2))
        better = logk > best
        best = np.where(better, logk, best)
        zone = np.where(better, k, zone)
        cls[k] = np.bincount(mask[inst == k]).argmax()
        hw[k] = (h, w)
    heat = np.exp(best)
    for ci, cj in centres:
        heat[ci, cj] = 1.0
        known[ci, cj] = True
    zone = np.where(known, zone, 0)
    out[..., 0] = heat
    out[..., 2] = zone
    out[..., 3:5] = hw[zone]
    out[..., 5] = cls[zone]
    out[..., 7] = known
    out[zone == 0, 3:6] = 0
    return out


def d4(arrays, k: int, flip: bool):
    """The same rotation by ``k`` quarter turns and optional left-right flip of every ``(..., H, W)`` array."""
    out = []
    for a in arrays:
        a = np.rot90(a, k, axes=(-2, -1))
        if flip:
            a = a[..., ::-1]
        out.append(np.ascontiguousarray(a))
    return out
