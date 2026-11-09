"""PRUE's field boundary segmentation, as Fields of The World trains it, on a TESSERA embedding.

PRUE (Practical Recipe for field boundary segmentation, CVPR 2026, arXiv 2603.27101; code
github.com/fieldsoftheworld/ftw-prue) is a U-Net with an EfficientNet-B7 encoder that labels every pixel
**background**, **field interior** or **field boundary**, trained with a log-cosh Dice loss whose class weights put
omega = 0.75 on the boundary, ``[0.05, 0.95 - omega, omega]``. Its fields are the connected regions of the interior
class: the boundary class keeps adjacent fields apart, the background class keeps fields off forest and villages.
On the FTW benchmark (24 countries, Estonia among them) it beats 18 alternatives, the instance models (Mask2Former,
SAM, Delineate Anything) and eleven frozen GFMs among them (RESEARCH.md).

What this module holds, each piece on its own:

* :func:`target`: the three classes of one chip, from its labels under the budget and from ESA WorldCover, with
  the background of ``../2026-11-05_paps_tessera/targets.py`` so the two methods learn from the same knowledge;
* :func:`log_cosh_dice`: ftw-prue's ``logCoshDice`` (``ftw_tools/torchgeo/losses.py``), the smp multiclass Dice
  per class, weighted by the normalised class weights, averaged, then ``log(cosh(.))``;
* :func:`build_unet`: smp's ``Unet("efficientnet-b7")``, 67.2 M parameters as the paper's 67.1 M, its first
  convolution taking TESSERA's 128 dimensions;
* :func:`parcels`: the fields of one chip from the class probabilities, with no truth: ``cc`` is PRUE's own
  polygonisation (the 4-connected regions of the interior class, those under 500 m2 dropped, ftw-prue's
  ``min_size``), ``grow`` the same regions grown into the pixels of the boundary class by watershed on the boundary
  probability, so that a field gets back the rim its interior gave up, and ``field`` the 4-connected regions of
  predicted field (interior or boundary), nothing split: the floor that says what the boundary class adds.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

#: The classes, in the order of the logits.
CLASSES = ("background", "interior", "boundary")
BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2
#: Pixels that are neither: WorldCover grassland or cropland outside the drawn parcels, and WorldCover's no data.
IGNORE = 3
#: PRUE's weights at omega = 0.75, ``[0.05, 0.95 - omega, omega]`` (the paper's Table 5 and section 4.2).
CLASS_WEIGHTS = (0.05, 0.20, 0.75)
#: WorldCover classes that are not background: grassland and cropland (as the PaPs targets).
AGRI = (30, 40)
#: ftw-prue's ``polygonize`` drops fields under 500 m2: 5 pixels at 10 m.
MIN_PIXELS = 5
FOUR = ndimage.generate_binary_structure(2, 1)
VARIANTS = ("cc", "grow", "field")


# ------------------------------------------------------------------------------------------------------ targets

def boundary_map(parcels: np.ndarray) -> np.ndarray:
    """``True`` on every parcel pixel with a 4-neighbour inside the chip carrying another identifier.

    The boundary target of ``../2026-10-18_parcel_delineation/delineation.py``, the one every delineation experiment
    has scored: between two parcels it is two pixels wide, one on each side, and the chip edge is no boundary.
    """
    p = np.asarray(parcels)
    b = np.zeros(p.shape, dtype=bool)
    v = p[:-1, :] != p[1:, :]
    b[:-1, :] |= v
    b[1:, :] |= v
    h = p[:, :-1] != p[:, 1:]
    b[:, :-1] |= h
    b[:, 1:] |= h
    return b & (p > 0)


def target(mask: np.ndarray, parcels: np.ndarray, worldcover: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """``(H, W)`` uint8 of one chip: 0 background, 1 interior, 2 boundary, 3 ignored.

    ``mask`` holds the class of every in-scheme parcel pixel (-1 elsewhere), ``parcels`` the id of every declared
    parcel (0 elsewhere), ``worldcover`` the WorldCover class, ``keep`` the drawn parcel ids (``None``: every parcel,
    as for the validation and test chips). A drawn in-scheme parcel pixel is boundary when a 4-neighbour inside the
    chip is another parcel or none, interior otherwise. Background is WorldCover's non-agricultural land outside the
    drawn parcels, exactly as for PaPs; the declared layer never says where background is. Everything else is
    ignored: WorldCover grassland and cropland outside the drawn parcels may be an undeclared field or not.
    """
    lab = (mask >= 0) & (parcels > 0)
    if keep is not None:
        lab &= np.isin(parcels, keep)
    y = np.full(mask.shape, IGNORE, np.uint8)
    y[~np.isin(worldcover, AGRI) & (worldcover != 0) & ~lab] = BACKGROUND
    y[lab] = INTERIOR
    y[lab & boundary_map(parcels)] = BOUNDARY
    return y


def d4(arrays, k: int, flip: bool):
    """The same rotation by ``k`` quarter turns and optional left-right flip of every ``(..., H, W)`` array."""
    out = []
    for a in arrays:
        a = np.rot90(a, k, axes=(-2, -1))
        if flip:
            a = a[..., ::-1]
        out.append(np.ascontiguousarray(a))
    return out


# --------------------------------------------------------------------------------------------------------- loss

def log_cosh_dice(logits: torch.Tensor, y: torch.Tensor, weights=CLASS_WEIGHTS, ignore: int = IGNORE,
                  eps: float = 1e-7) -> torch.Tensor:
    """ftw-prue's ``logCoshDice`` on ``(B, C, H, W)`` logits and ``(B, H, W)`` labels.

    As smp's ``DiceLoss(mode="multiclass", from_logits=True, smooth=0, ignore_index=ignore)``: softmax, ignored
    pixels zeroed in prediction and one-hot truth, the soft Dice of every class over the whole batch, ``1 - Dice``
    set to 0 for a class absent from the batch's truth; then ftw-prue's ``aggregate_loss``: weighted by the
    normalised class weights, averaged over the classes, ``log(cosh(.))``. In float32 whatever the autocast.
    """
    c = logits.shape[1]
    p = torch.softmax(logits.float(), dim=1)
    valid = (y != ignore)
    oh = torch.nn.functional.one_hot(torch.where(valid, y, 0).long(), c).permute(0, 3, 1, 2).float()
    v = valid[:, None].float()
    p, oh = p * v, oh * v
    dims = (0, 2, 3)
    inter = (p * oh).sum(dims)
    card = (p + oh).sum(dims)
    dice = 2.0 * inter / card.clamp_min(eps)
    loss = (1.0 - dice) * (oh.sum(dims) > 0).float()
    w = torch.as_tensor(weights, dtype=loss.dtype, device=loss.device)
    w = w / w.sum()
    return torch.log(torch.cosh((loss * w).mean()))


# -------------------------------------------------------------------------------------------------------- model

def build_unet(in_channels: int, encoder: str = "efficientnet-b7", weights: str | None = "imagenet"):
    """smp's U-Net as FTW builds it (``ftw_tools`` ``model: unet``), ``classes`` = 3, logits out.

    With ``in_channels`` other than 3 smp keeps the ImageNet stem and spreads its three input filters cyclically over
    the new channels, rescaled (``patch_first_conv``), as FTW does for its 8 channels.
    """
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=in_channels, classes=len(CLASSES))


# ------------------------------------------------------------------------------------------------------ parcels

def pieces(labels: np.ndarray) -> np.ndarray:
    """``labels`` (0 = none) with every label's 4-connected pieces numbered apart from 1, 0 kept."""
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


def _drop_small(seg: np.ndarray, min_pixels: int) -> np.ndarray:
    if not seg.any():
        return seg
    n = np.bincount(seg.ravel())
    small = n < min_pixels
    small[0] = False
    if small.any():
        lut = np.zeros(len(n), np.int32)
        big = np.nonzero(~small)[0]
        big = big[big > 0]
        lut[big] = np.arange(1, len(big) + 1, dtype=np.int32)
        seg = lut[seg]
    return seg


def parcels(prob: np.ndarray, variant: str, min_pixels: int = MIN_PIXELS) -> np.ndarray:
    """``(H, W)`` int32 field ids (0 = no field) of one chip from its ``(3, H, W)`` class probabilities.

    ``prob`` may be uint8 (probability x 255) or float. No truth is used: the fields cover the whole chip.
    """
    prob = np.asarray(prob)
    cls = prob.argmax(0)
    if variant == "field":
        return _drop_small(pieces((cls != BACKGROUND).astype(np.int32)), min_pixels)
    cc = _drop_small(pieces((cls == INTERIOR).astype(np.int32)), min_pixels)
    if variant == "cc":
        return cc
    if variant == "grow":
        from skimage.segmentation import watershed

        field = cls != BACKGROUND
        if not cc.any():
            return cc
        pb = prob[BOUNDARY].astype(np.float32)
        return watershed(pb, markers=cc, mask=field, connectivity=1).astype(np.int32)
    raise ValueError(f"variant {variant!r}")
