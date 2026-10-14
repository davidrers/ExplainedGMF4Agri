"""The checks of DESIGN.md, run on the pilot before any fit at scale.

1. The fit's input path (pixel store, ``encode_batch``, float32) reproduces the spike's ``v2_medium`` embeddings,
   which came from TESSERA's own ``encode_pixels`` on the chip rasters.
2. The switch: at the start of training both modes hand the head the same input (frozen cache = trainable encoder's
   output, both bf16), and one optimiser step moves the encoder in full mode and leaves it bit-identical otherwise.
3. A pixel with unobserved S1 months (bucket 8) is encoded exactly as ``encode_pixels`` encodes it.

    poetry run python experiments/2026-10-14_tessera_composites/check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fit_pixel import head  # noqa: E402
from pixels import embedding_cache, load_split, open_store  # noqa: E402
from tessera_v2 import DOY, encode_batch, load_student, s1_to_tessera  # noqa: E402

CHIPS = REPO / "data/eurocrops_chips/EE_2021_rev_mini"
STORE = REPO / "data/embeddings/tessera_pixels/EE_2021_rev_mini"
SPIKE = REPO / "results/2026-10-14_tessera_composites/spike/emb/v2_medium"
WEIGHTS = REPO / "data/weights/tessera/v2"


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main() -> None:
    from gfm4agri.pipeline.config import resolve_split

    dev = torch.device("cuda")
    arm = yaml.safe_load((HERE / "arms_run/tessera_v2m_full.yaml").read_text())
    sha = arm["model"]["checkpoint_sha256"]
    split_dir = resolve_split(CHIPS)
    rec = open_store(CHIPS, split_dir, STORE)
    print("store:", rec["pixels"])
    model = load_student("medium", WEIGHTS, sha).to(dev).eval()

    # 1. input path against the spike, chip by chip over the test split
    lab = load_split(STORE, "test")
    worst, cmin = 0.0, 1.0
    for k, cid in enumerate(rec["chips"]["test"]):
        sel = np.flatnonzero(lab["chip"] == k)
        if not len(sel):
            continue
        with torch.no_grad():
            e = encode_batch(model, lab["s2"][sel], lab["s1"][sel], dev, "32").cpu().numpy()
        ref = np.load(SPIKE / f"{cid}.npy").astype(np.float32).reshape(128, -1)[:, lab["pos"][sel]].T
        worst = max(worst, float(np.abs(e.astype(np.float16).astype(np.float32) - ref).max()))
        cmin = min(cmin, float(cos(e, ref).min()))
    print(f"1. input path vs spike, {len(lab['y']):,} test pixels: max |diff| {worst:.2e} (float16 storage), "
          f"min cosine {cmin:.6f}")
    assert cmin > 0.9999 and worst < 0.02, "the fit's input path does not reproduce the spike"

    # 2. the switch
    cache = embedding_cache(STORE, model, "medium", sha, "bf16-mixed", dev)
    tr = load_split(STORE, "train")
    idx = np.sort(np.random.default_rng(0).choice(len(tr["y"]), 4096, replace=False))
    cached = np.load(cache / "train.npy", mmap_mode="r")[idx].astype(np.float32)
    with torch.no_grad():
        live = encode_batch(model, tr["s2"][idx], tr["s1"][idx], dev, "bf16-mixed").cpu().numpy()
    gap = float(np.abs(live - cached).max())
    print(f"2a. frozen cache vs trainable encoder at step 0 (bf16): max |diff| {gap:.2e}, "
          f"min cosine {cos(live, cached).min():.6f}")
    assert cos(live, cached).min() > 0.999
    for trainable in (False, True):
        enc = load_student("medium", WEIGHTS, sha).to(dev).eval()
        enc.requires_grad_(trainable)
        before = [p.detach().clone() for p in enc.parameters()]
        net = head(20).to(dev)
        params = [{"params": net.parameters(), "lr": 1e-3}]
        if trainable:
            params.append({"params": enc.parameters(), "lr": 1e-5})
        opt = torch.optim.AdamW(params, weight_decay=0.05)
        y = torch.from_numpy(tr["y"][idx].astype(np.int64)).to(dev)
        x = (encode_batch(enc, tr["s2"][idx], tr["s1"][idx], dev, "bf16-mixed") if trainable
             else torch.from_numpy(cached).to(dev))
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        moved = sum(int(not torch.equal(a, b)) for a, b in zip(before, enc.parameters()))
        print(f"2b. trainable={trainable}: {moved} of {len(before)} encoder tensors changed after one step")
        assert (moved > 0) == trainable

    # 3. bucket 8: a pixel with 5 of 12 S1 months unobserved, against TESSERA's encode_pixels
    sys.path.insert(0, str(REPO / "data/external/tessera/tessera_infer_v2/student"))
    from infer import encode_pixels

    s2 = tr["s2"][idx[:64]].astype(np.float32)
    s1 = tr["s1"][idx[:64]].astype(np.float32)
    s1[:, [0, 1, 10, 11, 6], :] = np.nan
    with torch.no_grad():
        ours = encode_batch(model, s2, s1, dev, "32").cpu().numpy()
    theirs = encode_pixels(model, s2, DOY, s1_asc_bands=s1_to_tessera(s1), s1_asc_doys=DOY,
                           s2_masks=np.ones(s2.shape[:2], np.uint8), batch_pixels=4096, device=dev)
    print(f"3. 7 observed S1 months (bucket 8): max |ours - encode_pixels| {np.abs(ours - theirs).max():.2e}")
    assert np.abs(ours - theirs).max() < 1e-4
    print("all checks passed")


if __name__ == "__main__":
    main()
