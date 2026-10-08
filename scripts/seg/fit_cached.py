"""Stage 2 of the two-stage workflow: train the neck and decoder on cached encoder features.

Reads the same fit configuration as ``scripts/seg/train.py`` and the feature cache written
by ``scripts/seg/encode.py``, draws the label budget, trains, tests on the held-out chips and
writes ``<output-root>/<run>/P<pct>_draw<d>_seed<s>/results.json``.

    poetry run python scripts/seg/fit_cached.py -c configs/seg/terramind_v1_large_ee.yaml --pct 5
    poetry run python scripts/seg/fit_cached.py -c ... --pct 100 --max-epochs 1   # timing run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "seg"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--pct", type=float, required=True)
    ap.add_argument("--draw-seed", type=int, default=0)
    ap.add_argument("--cache-root", type=Path, default=REPO / "data" / "embeddings")
    ap.add_argument("--output-root", type=Path, default=REPO / "results" / "seg_cached")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None, help="overrides the config's seed")
    ap.add_argument("--train-variants", nargs="+", default=None,
                    help="cached D4 variants the training draw uses, e.g. k0; "
                         "default all eight, as the end-to-end augmentation")
    args = ap.parse_args()

    from encode import cache_dir_for
    from gfm4agri.benchmark.cached import fit_cached

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    cache_root = args.cache_root if args.cache_root.is_absolute() else REPO / args.cache_root
    r = fit_cached(cfg, cache_dir_for(cfg, cache_root), pct=args.pct, draw_seed=args.draw_seed,
                   output_root=args.output_root, num_workers=args.num_workers,
                   max_epochs=args.max_epochs, train_variants=args.train_variants)
    m = r["test_metrics"]
    print(f"{cfg['run_name']} P{args.pct:g} draw {args.draw_seed}: chips {r['chips']}, "
          f"fit {r['fit_seconds'] / 3600:.2f} h, test F1 {m.get('test/F1_Score', float('nan')):.4f}, "
          f"mIoU {m.get('test/mIoU', float('nan')):.4f}", flush=True)


if __name__ == "__main__":
    main()
