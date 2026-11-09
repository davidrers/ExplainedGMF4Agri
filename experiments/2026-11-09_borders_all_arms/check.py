"""Checks before the cluster run, on a pilot chip set (by default ``EE_2021_rev_mini``), on the CPU.

1. **Inputs equal the main workflow's.** For a few test chips of each arm, the loader of this folder
   (:mod:`chipsets`) must give exactly the standardised input the core gives the same arm: the core's
   ``EuroCropsSegDataModule`` after its normalisation (and, for the baseline, flattened month-major as the core's
   ``ScratchUNet`` does), or the core's ``CachedFeatureDataset`` at the identity variant for a token arm.
2. **Targets.** Border and crop targets agree with each other (the crop target is labelled exactly where the border
   target is interior or boundary) and, under a budget, only on drawn parcels.
3. **Models.** Every arm's model runs forward and backward on two training items; both heads have the chip's size;
   the crop loss reaches the shared part and not the border head; :class:`heads.Balance` sets a finite weight.

    poetry run python experiments/2026-11-09_borders_all_arms/check.py [--chips EE_2021_rev_mini] [--arms ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402


def core_inputs(arm: dict, source: dict, chips: Path, split_dir: Path, ids: list[str]) -> dict:
    """The core's standardised input of each chip, in this folder's layout ``(C, H, W)``."""
    core = arm["_kshot"]
    backbone = core["model"]["backbone"]
    if source["kind"] == "token":
        from gfm4agri.benchmark.cached import IDENTITY, CachedFeatureDataset

        ds = CachedFeatureDataset(source["cache_dir"], chips / "chips", ids, len(source["channel_list"]),
                                  variants=[IDENTITY])
        return {cid: ds[i]["image"].numpy() for i, cid in enumerate(ids)}
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

    image_root = Path(source["dir"]) if source["kind"] == "raster" and source["dir"] != str(chips) else None
    dm = EuroCropsSegDataModule(chips, backbone, normalisation=core["data"]["normalisation"], batch_size=1,
                                num_workers=0, augment=False, split_dir=split_dir, image_root=image_root)
    dm.setup("test")
    ds = dm.test_dataset
    stems = [Path(f).name[: -len(dm.img_suffix)] for f in ds.image_files]
    out = {}
    for cid in ids:
        item = ds[stems.index(cid)]
        batch = dm.aug({"image": item["image"][None], "mask": item["mask"][None]})
        x = batch["image"]
        if x.dim() == 5:                                    # (B, C, T, H, W) -> month-major, as ScratchUNet
            x = x.transpose(1, 2).flatten(1, 2)
        out[cid] = x[0].numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", default="EE_2021_rev_mini")
    ap.add_argument("-e", "--experiment", type=Path, default=HERE / "experiment.yaml")
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--n", type=int, default=3, help="test chips compared per arm")
    a = ap.parse_args()

    import fit_borders
    import heads
    import prue
    import run
    from gfm4agri.benchmark.segmentation_data import draw_support, split_list
    from gfm4agri.pipeline.config import load_machine, resolve_split

    exp, arms = run.load(a.experiment)
    chips = REPO / "data/eurocrops_chips" / a.chips
    split_dir = resolve_split(chips)
    mach = load_machine("hub")
    root = REPO / mach["cache_root"]
    worldcover = REPO / exp["worldcover"]
    test_ids = split_list(chips, "test", split_dir).read_text().split()[: a.n]
    train_ids = split_list(chips, "train", split_dir).read_text().split()
    keep5, _ = draw_support(chips, 5, 0, split_dir=split_dir)
    names = [c["name"] for c in __import__("json").loads((chips / "manifest.json").read_text())["classes"]]
    fails = []

    def check(ok: bool, msg: str) -> None:
        print(("  ok    " if ok else "  FAIL  ") + msg, flush=True)
        if not ok:
            fails.append(msg)

    for name in (a.arms or list(arms)):
        arm = arms[name]
        print(f"== {name} ({arm['input']}, {arm['kshot_arm']})", flush=True)
        source = run.source_for(arm, chips, split_dir, root, root, mach, "hub")
        ids = {"train": [c for c in train_ids if c in set(fit_borders.train_chips(chips, train_ids, keep5))][:2],
               "val": test_ids, "test": test_ids}
        ds_train, _, ds_test = fit_borders.datasets(source, ids, keep5, chips / "chips", worldcover, augment=True)

        # 1. inputs
        ref = core_inputs(arm, source, chips, split_dir, test_ids)
        for i, cid in enumerate(test_ids):
            mine = ds_test[i]["image"].numpy()
            same = mine.shape == ref[cid].shape and np.allclose(mine, ref[cid], atol=1e-4, rtol=1e-5)
            dmax = float(np.abs(mine - ref[cid]).max()) if mine.shape == ref[cid].shape else float("nan")
            check(same, f"{cid}: input {mine.shape} equals the core's (max |diff| {dmax:.2e})")

        # 2. targets
        for ds, kind in ((ds_test, "test, every parcel"), (ds_train, "train, K = 5 %")):
            it = ds[0]
            y, yc = it["y"].numpy(), it["yc"].numpy()
            check(np.array_equal(yc >= 0, (y == prue.INTERIOR) | (y == prue.BOUNDARY)),
                  f"{kind}: crop target labelled exactly on the border target's interior and boundary "
                  f"({int((yc >= 0).sum())} px)")
        it = ds_train[0]
        check(int((it["yc"] >= 0).sum()) <= int((it["mask"] >= 0).sum()),
              "train under the budget: no more labelled crop pixels than the dense mask")

        # 3. model
        cfg = {"crop": arm["crop"], "unet": {**(arm.get("unet") or {}), "weights": None}, "token": arm.get("token")}
        torch.manual_seed(0)
        net = fit_borders.build_model(source, len(names), names, cfg)
        net.train()
        batch = fit_borders.collate([ds_train[0], ds_train[min(1, len(ds_train) - 1)]])
        seg, crop = net(batch["image"])
        h, w = batch["y"].shape[-2:]
        check(tuple(seg.shape) == (2, 3, h, w) and tuple(crop.shape) == (2, len(names), h, w),
              f"heads: borders {tuple(seg.shape)}, crops {tuple(crop.shape)}")
        lb = prue.log_cosh_dice(seg, batch["y"], tuple(exp["loss"]["class_weights"]))
        lc = heads.crop_ce(crop, batch["yc"])
        shared = net.shared_parameters()
        bal = heads.Balance(**exp["balance"])
        bal.update(*heads.shared_grad_norms(shared, lb, lc))
        check(bal.weight is not None and np.isfinite(bal.weight),
              f"balance: crop weight {bal.weight:.4g} from |g_border| {bal.log[-1][0]:.3g}, |g_crop| "
              f"{bal.log[-1][1]:.3g}")
        net.zero_grad()
        lc.backward(retain_graph=True)
        border_head = net.unet.segmentation_head if hasattr(net, "unet") else net.border_head
        reaches_shared = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in shared)
        reaches_border = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in border_head.parameters())
        check(reaches_shared and not reaches_border, "the crop loss reaches the shared part, not the border head")
        n = {k: sum(p.numel() for p in v) for k, v in net.groups().items()}
        print(f"  parameters: {n}, shared {sum(p.numel() for p in shared):,}", flush=True)

    print(f"\n{len(fails)} failure(s)" + "".join(f"\n  {m}" for m in fails))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
