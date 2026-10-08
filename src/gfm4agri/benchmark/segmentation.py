"""Frozen-encoder, trainable-decoder segmentation tasks built from the backbone registry."""

from __future__ import annotations

from gfm4agri.benchmark.backbones import get_backbone

__all__ = ["DECODERS", "build_task", "count_params"]

#: One decoder family per resolution group, fixed so that model comparisons within a group
#: are not confounded by decoder choice. Decoder capacity is reported with every result.
DECODERS: dict[str, dict] = {
    "token_grid": {"decoder": "UNetDecoder", "decoder_channels": [512, 256, 128, 64]},
    # The token-size ablation keeps the token-grid decoder, so that only the token size changes.
    "token_grid_80m": {"decoder": "UNetDecoder", "decoder_channels": [512, 256, 128, 64]},
    # Per-pixel MLP on 10 m embedding rasters (the TESSERA authors' crop head). A light
    # spatial UNet is the planned second head for this group.
    "pixel_raster": {"decoder": "PixelMLPDecoder", "decoder_hidden": [512, 256], "decoder_dropout": 0.2},
}


def build_task(backbone: str, *, num_classes: int, class_names: list[str], bands: list[str],
               n_timesteps: int, ignore_index: int = -1, lr: float = 1e-4,
               weight_decay: float = 0.05, head_dropout: float = 0.1, loss: str = "ce",
               decoder: dict | None = None, plot_on_val: bool = False):
    """A ``SemanticSegmentationTask`` with the encoder frozen and the decoder trainable."""
    import terratorch  # noqa: F401  registers backbones, necks and decoders

    import gfm4agri.benchmark.decoders  # noqa: F401  registers the thesis decoders
    import gfm4agri.benchmark.necks  # noqa: F401  registers the thesis necks
    from terratorch.tasks import SemanticSegmentationTask

    spec = get_backbone(backbone)
    model_args = {
        **spec.model_args(bands, n_timesteps),
        **(decoder or DECODERS[spec.resolution_group]),
        "head_dropout": head_dropout,
        "num_classes": num_classes,
    }
    return SemanticSegmentationTask(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        loss=loss,
        lr=lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": weight_decay},
        ignore_index=ignore_index,
        freeze_backbone=True,
        freeze_decoder=False,
        plot_on_val=plot_on_val,
        class_names=class_names,
    )


def count_params(task) -> dict[str, int]:
    """Parameter counts by component, trainable and total."""
    model = task.model
    parts = {"encoder": getattr(model, "encoder", None), "neck": getattr(model, "neck", None),
             "decoder": getattr(model, "decoder", None), "head": getattr(model, "head", None),
             "aux_heads": getattr(model, "aux_heads", None)}
    out = {}
    for name, mod in parts.items():
        if not hasattr(mod, "parameters"):  # absent, or the identity-lambda neck
            continue
        out[f"{name}_total"] = sum(p.numel() for p in mod.parameters())
        out[f"{name}_trainable"] = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    out["trainable_total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    out["all_total"] = sum(p.numel() for p in model.parameters())
    return out
