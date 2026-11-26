#!/usr/bin/env bash
# The released weights of the three external models, into data/external_models/ (git-ignored), from their authors'
# release locations, pinned where the release is.
#
#     bash experiments/2026-11-26_open_delineation/fetch_weights.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
W=data/external_models
mkdir -p "$W/delineate_anything" "$W/ftw" "$W/utae_paps"
R=https://github.com/fieldsoftheworld/ftw-baselines/releases/download
get() { [ -s "$2" ] || curl -sSL -o "$2" "$1"; }
get "$R/v3.1/prue_efnetb7_ccby_checkpoint.ckpt"                    "$W/ftw/prue_efnetb7_ccby_checkpoint.ckpt"
get "$R/v2/3_Class_FULL_FTW_Pretrained_v2.ckpt"                    "$W/ftw/3_Class_FULL_FTW_Pretrained_v2.ckpt"
get "$R/v2/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt"       "$W/ftw/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt"
# Delineate Anything: v2 from the authors' Hugging Face repository at the revision ftw-tools pins; v1 (YOLO11x)
# from TorchGeo's mirror, the file ftw-tools uses.
get "https://hf.co/MykolaL/DelineateAnything/resolve/369d0b4c44cf9bec2bd3a27bc81810cadd2c963e/DelineateAnythingv2.pt" \
    "$W/delineate_anything/DelineateAnythingv2.pt"
get "https://hf.co/torchgeo/delineate-anything/resolve/60bea7b2f81568d16d5c75e4b5b06289e1d7efaf/delineate_anything_rgb_yolo11x-88ede029.pt" \
    "$W/delineate_anything/DelineateAnything.pt"
# U-TAE + PaPs, PASTIS panoptic weights (five folds and conf.json).
get "https://zenodo.org/records/5172301/files/UTAE+PAPs_PanopticSeg_weights.zip?download=1" "$W/utae_paps/UTAE_PAPs_weights.zip"
[ -d "$W/utae_paps/UTAE_PAPs" ] || (cd "$W/utae_paps" && unzip -q UTAE_PAPs_weights.zip -x '__MACOSX/*' '*.DS_Store')
sha256sum "$W"/ftw/*.ckpt "$W"/delineate_anything/*.pt "$W"/utae_paps/UTAE_PAPs/Fold_*/model.pth.tar \
    > experiments/2026-11-26_open_delineation/reference/weights.sha256
