# U-TAE + PaPs, vendored

From https://github.com/VSainteuf/utae-paps, commit `987874e26a98bb399277f2d55eb010744d26f881`, MIT License
(`LICENSE`, Copyright (c) 2021 VSainteuf). Garnot and Landrieu, *Panoptic Segmentation of Satellite Image Time
Series with Convolutional Temporal Attention Networks*, ICCV 2021.

| File | Source |
|---|---|
| `utae.py` | `src/backbones/utae.py` |
| `ltae.py` | `src/backbones/ltae.py` |
| `positional_encoding.py` | `src/backbones/positional_encoding.py` |
| `convlstm.py` | `src/backbones/convlstm.py` (imported by `utae.py`) |
| `paps.py` | `src/panoptic/paps.py` |

The only changes: the `src.backbones...` imports are relative, and `paps.py` imports `torch_scatter.scatter_max`
inside the one branch that uses it (centre extraction with training `zones`), so that inference needs no
`torch_scatter`. Two changes to `PaPs.forward`, as in `experiments/2026-11-05_paps_tessera/paps_model.py`: an option `prune`
(used here) drops the candidate centres below `min_confidence` before their masks are assembled, which the pseudo-NMS
never keeps anyway, so the panoptic output is the same and much faster (checked by `check_wiring.py`); and the
candidates' confidences are flattened with `view(-1)` instead of `squeeze()`, so that a chip with a single candidate
does not fail. The networks, the pseudo-NMS and every default are otherwise the source's. The weights are the authors'
PASTIS panoptic weights, https://zenodo.org/record/5172301 (`UTAE+PAPs_PanopticSeg_weights.zip`, five folds and
`conf.json`), unzipped into `data/external_models/utae_paps/UTAE_PAPs/`.
