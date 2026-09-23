#!/usr/bin/env python3
"""Validate a TerraTorch YAML config file for common issues.

Usage:
    python validate_config.py <config.yaml>

Checks:
    - Required sections present (trainer, data, model, optimizer)
    - Backbone exists in registry
    - SelectIndices match model size
    - Band configuration consistency
    - Normalization stats provided
    - Neck chain is valid for ViT backbones
"""
import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: pyyaml is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# Known ViT backbones and their expected SelectIndices
VIT_SELECT_INDICES = {
    "prithvi_eo_v1_100": [2, 5, 8, 11],
    "prithvi_eo_tiny": [2, 5, 8, 11],
    "prithvi_eo_v2_300": [5, 11, 17, 23],
    "prithvi_eo_v2_300_tl": [5, 11, 17, 23],
    "prithvi_eo_v2_600": [7, 15, 23, 31],
    "prithvi_eo_v2_600_tl": [7, 15, 23, 31],
}

VIT_BACKBONES = set(VIT_SELECT_INDICES.keys()) | {
    "satmae", "scalemae", "clay_v1", "clay_v15", "dofa_vit",
    "terramind", "dinov3",
}

VALID_LOSSES_SEG = {"ce", "dice", "jaccard", "focal"}
VALID_LOSSES_REG = {"mse", "rmse", "mae", "huber"}
VALID_LOSSES_CLS = {"ce", "jaccard", "focal"}

VALID_DECODERS = {
    "UNetDecoder", "UperNetDecoder", "FCNDecoder",
    "IdentityDecoder", "LinearDecoder", "MLPDecoder",
}


def warn(msg):
    print(f"  WARNING: {msg}")


def error(msg):
    print(f"  ERROR: {msg}")


def info(msg):
    print(f"  OK: {msg}")


def validate(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    issues = 0
    print(f"\nValidating: {config_path}\n")

    # Check top-level sections
    for section in ["trainer", "data", "model"]:
        if section not in cfg:
            error(f"Missing required section: '{section}'")
            issues += 1
        else:
            info(f"Section '{section}' present")

    if "optimizer" not in cfg:
        warn("No 'optimizer' section — Lightning will use default Adam")

    # Check data section
    data = cfg.get("data", {})
    data_init = data.get("init_args", {})
    if data_init:
        if "means" not in data_init or "stds" not in data_init:
            error("Missing 'means' and/or 'stds' in data.init_args (required for normalization)")
            issues += 1
        else:
            means = data_init["means"]
            stds = data_init["stds"]
            if len(means) != len(stds):
                error(f"means ({len(means)} values) and stds ({len(stds)} values) have different lengths")
                issues += 1
            else:
                info(f"Normalization stats: {len(means)} bands")

        if "num_classes" not in data_init and "Regression" not in data.get("class_path", ""):
            warn("No 'num_classes' in data.init_args")

    # Check model section
    model = cfg.get("model", {})
    model_init = model.get("init_args", {})
    model_args = model_init.get("model_args", {})

    if model_args:
        backbone = model_args.get("backbone", "")
        info(f"Backbone: {backbone}")

        # Check if ViT backbone has necks
        is_vit = any(vit in backbone for vit in VIT_BACKBONES) or "vit" in backbone.lower()
        necks = model_args.get("necks", [])

        if is_vit and not necks:
            error(f"ViT backbone '{backbone}' requires necks (SelectIndices + ReshapeTokensToImage + *ToPyramidal)")
            issues += 1

        # Check SelectIndices match
        if backbone in VIT_SELECT_INDICES and necks:
            for neck in necks:
                if neck.get("name") == "SelectIndices":
                    expected = VIT_SELECT_INDICES[backbone]
                    actual = neck.get("indices", [])
                    if actual != expected:
                        warn(f"SelectIndices {actual} may not match '{backbone}' — expected {expected}")
                    else:
                        info(f"SelectIndices {actual} matches backbone")

        # Check neck chain completeness for ViT
        if is_vit and necks:
            neck_names = [n.get("name", "") for n in necks]
            if "SelectIndices" not in neck_names:
                warn("ViT backbone missing SelectIndices neck")
            if "ReshapeTokensToImage" not in neck_names:
                warn("ViT backbone missing ReshapeTokensToImage neck")

        # Check decoder
        decoder = model_args.get("decoder", "")
        if decoder:
            info(f"Decoder: {decoder}")
        else:
            warn("No decoder specified")

        # Check band consistency
        backbone_bands = model_args.get("backbone_bands", [])
        output_bands = data_init.get("output_bands", [])
        if backbone_bands and output_bands and backbone_bands != output_bands:
            warn(f"backbone_bands ({len(backbone_bands)}) differs from data output_bands ({len(output_bands)})")

    # Check loss
    loss = model_init.get("loss", "")
    task_class = model.get("class_path", "")
    if loss:
        info(f"Loss: {loss}")
    else:
        warn("No loss specified")

    print(f"\nValidation complete: {issues} error(s) found")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Validate a TerraTorch config")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"File not found: {args.config}")
        sys.exit(1)

    issues = validate(args.config)
    sys.exit(1 if issues > 0 else 0)


if __name__ == "__main__":
    main()
