"""The registry of geospatial foundation models a segmentation fit can use.

Each entry states everything that differs between models and nothing that does not: the
TerraTorch backbone arguments, the neck chain that turns its output into a feature pyramid,
the decoder-input resolution group (a declared experimental factor, see
docs/thesis_design.md), and the normalisation statistics the encoder was pretrained under.
The decoder, the loss and the
data are held fixed elsewhere, so adding a model is adding an entry here.

Two things vary beyond the arguments: a model may consume only a subset of the exported
bands (``input_bands``; Prithvi was pretrained on six HLS bands), and it may take the
acquisition date and location as extra inputs (``uses_coords``; the Prithvi ``_tl``
variants). The datamodule reads both from the spec.

THOR is not part of TerraTorch. It registers its backbones through its own extension,
``thor_terratorch_ext``, which ``scripts/env/install_thor.sh`` installs outside the Poetry
lock, and it is imported only when a THOR entry is built, because importing it patches timm's
attention for the whole process.

Precomputed embeddings (TESSERA and AlphaEarth) enter the same registry with an
identity backbone, the ``pixel_raster`` group, and a ``representation`` naming the embedding
rasters to read instead of the Sentinel-2 stack. Their normalisation statistics come from the
training chips, recorded next to the rasters, because the embedding has no pretraining
input distribution to match. TESSERA v2 is a third kind: the same identity backbone and group,
but its raster is computed by the K-shot runner from the chip set's composites with the encoder
named in ``computed``, and cached outside the chip set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

__all__ = ["BACKBONES", "BackboneSpec", "get_backbone"]


@dataclass(frozen=True)
class BackboneSpec:
    """How to build one frozen encoder and bring its features to a decoder."""

    name: str
    #: ``"token_grid"`` for ViT patch tokens (about 160 m at 16 px patches on 10 m input),
    #: ``"pixel_raster"`` for 10 m per-pixel embeddings. Comparisons are made within a group.
    #: ``"token_grid_80m"`` holds THOR on 80 m tokens, an ablation compared with THOR at 160 m.
    resolution_group: str
    #: ``(bands, n_timesteps) -> model_args`` fragment for ``EncoderDecoderFactory``.
    model_args: Callable[[list[str], int], dict]
    #: ``(bands) -> (means, stds)`` per band in the order of ``bands``, as pretrained;
    #: ``None`` for precomputed embeddings, which use training-chip statistics.
    stats: Callable[[list[str]], tuple[list[float], list[float]]] | None
    #: The bands the encoder consumes, in its order; ``None`` for every exported band.
    input_bands: tuple[str, ...] | None = None
    #: Whether the encoder takes ``temporal_coords`` (B, T, 2) and ``location_coords`` (B, 2).
    uses_coords: bool = False
    #: The input rasters: ``s2_monthly`` for the Sentinel-2 stack, otherwise the name of a
    #: precomputed embedding set with its ``<root>/<representation>.json`` sidecar.
    representation: str = "s2_monthly"
    notes: str = ""
    extra: dict = field(default_factory=dict)
    #: Further modalities read beside the Sentinel-2 stack, ``name -> {"suffix", "bands",
    #: "stats"}``: the raster ``<chip><suffix>`` in the same ``(time channels)`` layout, its
    #: band names in the encoder's order, and ``(bands) -> (means, stds)`` as pretrained.
    extra_modalities: dict = field(default_factory=dict)
    #: Whether the encoder takes its further modalities as more bands of one image rather than as
    #: a dictionary of modalities. THOR reads Sentinel-1 as a band group of the same image, so its
    #: datamodule hands over the interleaved stack without splitting it; TerraMind takes one
    #: tensor per modality.
    stack_modalities: bool = False
    #: For a 10 m backbone whose raster the K-shot runner computes rather than reads from the chip
    #: set (TESSERA v2): ``encoder`` (a key of ``raster_cache.ENCODERS``), ``student``,
    #: ``checkpoint_sha256``, ``precision``, ``dims`` and the chip-set rasters it reads, ``inputs``.
    #: Empty for every other backbone.
    computed: dict = field(default_factory=dict)


def _terramind_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    from terratorch.models.backbones.terramind.model import terramind_register as tm

    key = "untok_sen2l2a@224"
    order = tm.PRETRAINED_BANDS[key]
    mean, std = tm.v1_pretraining_mean[key], tm.v1_pretraining_std[key]
    missing = [b for b in bands if b not in order]
    if missing:
        raise KeyError(f"TerraMind S2L2A was not pretrained on {missing}")
    return [float(mean[order.index(b)]) for b in bands], [float(std[order.index(b)]) for b in bands]


def _terramind_s1rtc_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    from terratorch.models.backbones.terramind.model import terramind_register as tm

    key = "untok_sen1rtc@224"
    order = tm.PRETRAINED_BANDS[key]
    mean, std = tm.v1_pretraining_mean[key], tm.v1_pretraining_std[key]
    missing = [b for b in bands if b not in order]
    if missing:
        raise KeyError(f"TerraMind S1RTC was not pretrained on {missing}")
    return [float(mean[order.index(b)]) for b in bands], [float(std[order.index(b)]) for b in bands]


#: Sentinel-1 RTC as TerraMind's second modality: VV then VH in dB, twelve monthly composites
#: in ``<chip>_s1rtc.tif``, written by ``scripts/hub/build_s1_chips.py``.
S1RTC = {"S1RTC": {"suffix": "_s1rtc.tif", "bands": ("VV", "VH"),
                   "stats": _terramind_s1rtc_stats, "sidecar": "s1_rtc.json"}}


def _terramind(variant: str, indices: list[int], bottleneck: int,
               extra_modalities: dict | None = None) -> Callable[[list[str], int], dict]:
    extra_modalities = extra_modalities or {}

    def build(bands: list[str], n_timesteps: int) -> dict:
        # ``bands`` are the Sentinel-2 bands; a further modality brings its own band list.
        multimodal = {"backbone_merge_method": "mean"} if extra_modalities else {}
        return {
            "backbone": variant,
            "backbone_pretrained": True,
            "backbone_modalities": ["S2L2A", *extra_modalities],
            "backbone_bands": {"S2L2A": list(bands),
                               **{m: list(v["bands"]) for m, v in extra_modalities.items()}},
            # With a second modality TerraMind averages the tokens of the two at every
            # position, so the token grid, and hence the neck and decoder, stay exactly those
            # of the Sentinel-2 run and the two are compared at matched capacity.
            **multimodal,
            # TerraMind is single-date; the wrapper runs it per month and concatenates the
            # monthly features, so the decoder sees the months in order.
            "backbone_use_temporal": True,
            "backbone_temporal_pooling": "concat",
            "backbone_temporal_n_timestamps": n_timesteps,
            "necks": [
                {"name": "SelectIndices", "indices": indices},
                {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                # Concatenating 12 months leaves T x embed_dim channels, 12 x 1024 = 12,288
                # for the large variant. Projecting first is what keeps the trainable neck
                # and decoder the same size as Prithvi's, which is the condition under which
                # the two are comparable at all; without it the large variant carries 815 M
                # trainable parameters against Prithvi's 63 M and does not fit on a 16 GB card.
                {"name": "ChannelBottleneck", "out_channels": bottleneck},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    return build


#: Prithvi-EO-2.0 was pretrained on six HLS bands; on Sentinel-2, NIR_NARROW is B8A.
PRITHVI_BANDS = ("BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2")


def _prithvi_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    from terratorch.models.backbones import prithvi_vit as pv

    order = [getattr(b, "value", str(b)) for b in pv.PRETRAINED_BANDS]
    missing = [b for b in bands if b not in order]
    if missing:
        raise KeyError(f"Prithvi-EO-2.0 was not pretrained on {missing}")
    return ([float(pv.PRITHVI_V2_MEAN[order.index(b)]) for b in bands],
            [float(pv.PRITHVI_V2_STD[order.index(b)]) for b in bands])


def _prithvi(variant: str, indices: list[int], bottleneck: int) -> Callable[[list[str], int], dict]:
    def build(bands: list[str], n_timesteps: int) -> dict:
        return {
            "backbone": variant,
            "backbone_pretrained": True,
            "backbone_bands": list(bands),
            # Prithvi is natively multi-temporal: the 12 months enter one ViT as 12 x 196
            # tokens with a 3D positional embedding, so no temporal wrapper is needed.
            "backbone_num_frames": n_timesteps,
            "necks": [
                {"name": "SelectIndices", "indices": indices},
                # Tokens back to maps, months stacked on channels: T x embed_dim channels.
                {"name": "ReshapeTokensToImage", "remove_cls_token": True,
                 "effective_time_dim": n_timesteps},
                # 12 x 1024 = 12,288 channels is too wide for any decoder; project first.
                {"name": "ChannelBottleneck", "out_channels": bottleneck},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    return build


#: THOR's Sentinel-2 bands by native resolution, in its two spectral groups; on Sentinel-2,
#: NIR_BROAD is B08 and NIR_NARROW is B8A. Its third Sentinel-2 group, B01 and B09 at 60 m, is
#: left out: 60 m pixels do not tile the 2,240 m chip, and the THOR authors drop the two
#: atmospheric bands from their own Sentinel-2 configurations as well.
THOR_BANDS_10M = ("BLUE", "GREEN", "RED", "NIR_BROAD")
THOR_BANDS_20M = ("RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3", "NIR_NARROW", "SWIR_1", "SWIR_2")
THOR_BANDS = THOR_BANDS_10M + THOR_BANDS_20M
#: The chip, 224 px of 10 m (``chip_grid.CHIP_PX * PIXEL_M``), and the ground size of a token.
THOR_GROUND_COVER_M = 2240
THOR_TOKEN_M = 160


def _register_thor() -> None:
    """Register the THOR backbones in TerraTorch's registry, once per process.

    Importing ``thor`` replaces ``forward`` of timm's ``Attention`` and ``Block`` for the whole
    process, so that THOR's ALiBi bias can enter as an attention mask; without a mask the
    patched forward is the stock one, which ``tests/test_thor_backbone.py`` checks. It also calls
    ``logging.basicConfig(level=INFO)``, which is undone here so a fit's log stays as it was.
    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    import thor_terratorch_ext  # noqa: F401

    root.setLevel(level)
    root.handlers[:] = handlers


def _thor_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    _register_thor()
    from thor_terratorch_ext.models.backbones import thor_vit as tv

    missing = [b for b in bands if tv.lookup_band.get(b) not in tv.THOR_NORMALIZATION_PARAMS]
    if missing:
        raise KeyError(f"THOR was not pretrained on {missing}")
    # THOR's statistics are in reflectance; the chips store reflectance x 10000.
    params = [tv.THOR_NORMALIZATION_PARAMS[tv.lookup_band[b]] for b in bands]
    return [p["mean"] * 1e4 for p in params], [p["std"] * 1e4 for p in params]


#: Sentinel-1 as THOR reads it: VV and VH, which THOR maps to its 10 m IW group.
THOR_S1_BANDS = ("VV", "VH")


def _thor_s1_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    """THOR's pretraining statistics of its S1 bands at 10 m, in dB, the unit of ``_s1rtc.tif``."""
    _register_thor()
    from thor_terratorch_ext.models.backbones import thor_vit as tv

    # THOR keys its SAR statistics by ground sampling distance: "S1:IW-VV_10" for 10 m.
    keys = [f"{tv.lookup_band.get(b)}_10" for b in bands]
    missing = [b for b, k in zip(bands, keys) if k not in tv.THOR_NORMALIZATION_PARAMS]
    if missing:
        raise KeyError(f"THOR was not pretrained on {missing}")
    params = [tv.THOR_NORMALIZATION_PARAMS[k] for k in keys]
    return [p["mean"] for p in params], [p["std"] for p in params]


#: Sentinel-1 RTC as THOR's third band group: the ``_s1rtc.tif`` TerraMind reads, normalised with
#: THOR's statistics, and stacked behind the Sentinel-2 bands (``stack_modalities``).
THOR_S1RTC = {"S1RTC": {"suffix": "_s1rtc.tif", "bands": THOR_S1_BANDS,
                        "stats": _thor_s1_stats, "sidecar": "s1_rtc.json"}}


def _thor(variant: str, indices: list[int], bottleneck: int, token_m: int = THOR_TOKEN_M,
          merge: str = "concat", s1: bool = False) -> Callable[[list[str], int], dict]:
    gsd = {**dict.fromkeys(THOR_BANDS_10M, 10), **dict.fromkeys(THOR_BANDS_20M, 20),
           **dict.fromkeys(THOR_S1_BANDS, 10)}
    if THOR_GROUND_COVER_M % token_m or token_m % max(gsd.values()):
        raise ValueError(f"{token_m} m tokens do not tile the chip in every band group")

    def build(bands: list[str], n_timesteps: int) -> dict:
        # Every path that builds a model asks for these arguments first, so THOR is in
        # TerraTorch's registry by the time the factory looks it up.
        _register_thor()
        # ``bands`` are the Sentinel-2 bands; S1, when read, follows them in every month, as the
        # datamodule interleaves it.
        model_bands = list(bands) + (list(THOR_S1_BANDS) if s1 else [])
        return {
            "backbone": variant,
            "backbone_pretrained": True,
            "backbone_model_bands": model_bands,
            "backbone_ground_cover": THOR_GROUND_COVER_M,
            # THOR tokenises each band group on its own, at a patch size chosen at inference.
            # Every group gets tokens of the same ground size: at 160 m, 16 px on the 10 m bands
            # (S2 and S1) and 8 px on the 20 m bands, the 14 x 14 grid TerraMind has on the same
            # chip; at 80 m, 8 and 4 px and a 28 x 28 grid, 4 px being the smallest patch THOR
            # was pretrained with.
            "backbone_patch_sizes": {b: token_m // gsd[b] for b in model_bands},
            # THOR hands over maps rather than tokens and needs no reshaping neck. "concat"
            # stacks the groups' maps on the channel axis, as the THOR authors do for dense
            # tasks: groups x embed_dim channels per month. "mean" averages them at every token
            # position, as TerraMind averages its modalities: embed_dim channels per month.
            "backbone_merge_method": merge,
            # THOR is single-date; the wrapper runs it per month and concatenates the
            # monthly features, as for TerraMind.
            "backbone_use_temporal": True,
            "backbone_temporal_pooling": "concat",
            "backbone_temporal_n_timestamps": n_timesteps,
            "necks": [
                {"name": "SelectIndices", "indices": indices},
                # Concatenated: 12 months x 2 groups x 1024 = 24,576 channels per layer for the
                # large variant. Averaged: 12 x 1024 = 12,288, TerraMind's width.
                {"name": "ChannelBottleneck", "out_channels": bottleneck},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    return build


def _precomputed(n_dims: int) -> Callable[[list[str], int], dict]:
    def build(bands: list[str], n_timesteps: int) -> dict:
        # The embedding is the encoder output already: pass it through unchanged.
        return {"backbone": "IdentityBackbone", "backbone_out_channels": [len(bands) or n_dims],
                "necks": []}
    return build


BACKBONES: dict[str, BackboneSpec] = {
    "terramind_v1_small": BackboneSpec(
        name="terramind_v1_small", resolution_group="token_grid",
        model_args=_terramind("terramind_v1_small", [2, 5, 8, 11], bottleneck=768),
        stats=_terramind_stats,
        notes="12 layers, 384 dims; S2L2A patch embedding"),
    "terramind_v1_base": BackboneSpec(
        name="terramind_v1_base", resolution_group="token_grid",
        model_args=_terramind("terramind_v1_base", [2, 5, 8, 11], bottleneck=768),
        stats=_terramind_stats,
        notes="12 layers, 768 dims; S2L2A patch embedding"),
    "terramind_v1_large": BackboneSpec(
        name="terramind_v1_large", resolution_group="token_grid",
        model_args=_terramind("terramind_v1_large", [5, 11, 17, 23], bottleneck=768),
        stats=_terramind_stats,
        notes="24 layers, 1024 dims; S2L2A patch embedding; the largest TerraMind released"),
    "terramind_v1_small_s2s1": BackboneSpec(
        name="terramind_v1_small_s2s1", resolution_group="token_grid",
        model_args=_terramind("terramind_v1_small", [2, 5, 8, 11], bottleneck=768,
                              extra_modalities=S1RTC),
        stats=_terramind_stats, extra_modalities=S1RTC,
        notes="terramind_v1_small on S2L2A and S1RTC, tokens averaged across modalities"),
    "terramind_v1_large_s2s1": BackboneSpec(
        name="terramind_v1_large_s2s1", resolution_group="token_grid",
        model_args=_terramind("terramind_v1_large", [5, 11, 17, 23], bottleneck=768,
                              extra_modalities=S1RTC),
        stats=_terramind_stats, extra_modalities=S1RTC,
        notes="terramind_v1_large on S2L2A and S1RTC, tokens averaged across modalities"),
    "prithvi_eo_v2_300_tl": BackboneSpec(
        name="prithvi_eo_v2_300_tl", resolution_group="token_grid",
        model_args=_prithvi("prithvi_eo_v2_300_tl", [5, 11, 17, 23], bottleneck=768),
        stats=_prithvi_stats, input_bands=PRITHVI_BANDS, uses_coords=True,
        notes="24 layers, 1024 dims, 16 x 16 patches, HLS six-band pretraining, date and "
              "location encodings"),
    "prithvi_eo_v2_300": BackboneSpec(
        name="prithvi_eo_v2_300", resolution_group="token_grid",
        model_args=_prithvi("prithvi_eo_v2_300", [5, 11, 17, 23], bottleneck=768),
        stats=_prithvi_stats, input_bands=PRITHVI_BANDS,
        notes="24 layers, 1024 dims, 16 x 16 patches, HLS six-band pretraining"),
    "prithvi_eo_v2_600_tl": BackboneSpec(
        name="prithvi_eo_v2_600_tl", resolution_group="token_grid",
        model_args=_prithvi("prithvi_eo_v2_600_tl", [7, 15, 23, 31], bottleneck=768),
        stats=_prithvi_stats, input_bands=PRITHVI_BANDS, uses_coords=True,
        notes="32 layers, 1280 dims, 14 x 14 patches, HLS six-band pretraining, date and "
              "location encodings; the largest Prithvi-EO-2.0 released"),
    "thor_v1_base": BackboneSpec(
        name="thor_v1_base", resolution_group="token_grid",
        model_args=_thor("thor_v1_base", [2, 5, 8, 11], bottleneck=768),
        stats=_thor_stats, input_bands=THOR_BANDS,
        notes="12 layers, 768 dims, FlexiViT with 2D ALiBi; 10 m and 20 m band groups "
              "tokenised apart, at 16 and 8 px"),
    "thor_v1_large": BackboneSpec(
        name="thor_v1_large", resolution_group="token_grid",
        model_args=_thor("thor_v1_large", [5, 11, 17, 23], bottleneck=768),
        stats=_thor_stats, input_bands=THOR_BANDS,
        notes="24 layers, 1024 dims, FlexiViT with 2D ALiBi; 10 m and 20 m band groups "
              "tokenised apart, at 16 and 8 px; the largest THOR released"),
    # The same encoder on 80 m tokens: the token-size ablation. Everything trainable is that of
    # thor_v1_large, so the two differ only in the size of the tokens the decoder receives.
    "thor_v1_large_80m": BackboneSpec(
        name="thor_v1_large_80m", resolution_group="token_grid_80m",
        model_args=_thor("thor_v1_large", [5, 11, 17, 23], bottleneck=768, token_m=80),
        stats=_thor_stats, input_bands=THOR_BANDS,
        notes="thor_v1_large on 80 m tokens, 8 and 4 px patches, a 28 x 28 grid"),
    # THOR on Sentinel-2 and Sentinel-1: S1 is a third band group on the same 160 m grid, and the
    # three groups are averaged at every token position, as TerraMind averages its two
    # modalities, so the bottleneck and decoder are TerraMind's (decided 15 October 2026).
    "thor_v1_large_s2s1": BackboneSpec(
        name="thor_v1_large_s2s1", resolution_group="token_grid",
        model_args=_thor("thor_v1_large", [5, 11, 17, 23], bottleneck=768, merge="mean", s1=True),
        stats=_thor_stats, input_bands=THOR_BANDS, extra_modalities=THOR_S1RTC,
        stack_modalities=True,
        notes="thor_v1_large on S2 and S1 RTC (VV, VH), the three band groups averaged per token"),
    "tessera_v1": BackboneSpec(
        name="tessera_v1", resolution_group="pixel_raster", model_args=_precomputed(128),
        stats=None, representation="tessera_v1",
        notes="precomputed annual embedding, 128 dims per 10 m pixel, from the calendar-year "
              "Sentinel-1 and Sentinel-2 series; read from the public Zarr store"),
    # TESSERA v2 from its open weights, frozen, run by the K-shot runner on the chip set's monthly
    # S2 and S1 composites; its 10 m output is cached as an embedding raster
    # (gfm4agri.pipeline.raster_cache) and read like a precomputed one. It replaces tessera_v1,
    # whose store rasters sit about one pixel off the chip grid (decided 15 October 2026).
    "tessera_v2_medium": BackboneSpec(
        name="tessera_v2_medium", resolution_group="pixel_raster", model_args=_precomputed(128),
        stats=None, representation="tessera_v2_medium",
        computed={"encoder": "tessera_v2", "student": "medium",
                  "checkpoint_sha256": "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
                  "precision": "bf16-mixed", "dims": 128, "inputs": ["_merged.tif", "_s1rtc.tif"]},
        notes="TESSERA v2 Medium pixel student (21.0 M parameters), frozen, 128 dims per 10 m pixel "
              "from the 12 monthly S2 and S1 composites; computed and cached by the runner"),
    "alphaearth_v1": BackboneSpec(
        name="alphaearth_v1", resolution_group="pixel_raster", model_args=_precomputed(64),
        stats=None, representation="alphaearth_v1",
        notes="precomputed annual embedding, 64 dims per 10 m pixel (unit vectors), Google's "
              "Satellite Embedding V1; cut from the published COG tiles"),
}


def get_backbone(name: str) -> BackboneSpec:
    try:
        return BACKBONES[name]
    except KeyError:
        raise KeyError(f"unknown backbone {name!r}; registered: {sorted(BACKBONES)}") from None
