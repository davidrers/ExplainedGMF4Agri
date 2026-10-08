# How many cached D4 variants does the two-stage fit need? 8 October 2026

**Question.** The two-stage fit encodes training chips in all eight D4 symmetries. Would the identity alone, or the
four quarter turns, train as well?

**What ran.** TerraMind v1 large on full Estonia at K = 5 %, draw 0, 15 epochs, seeds 0 to 2, trained from the
identity only (k0), the four quarter turns (rot4) and all eight (all8). `run_d4_ablation.sh`.

**Outcome.** Test Macro-F1 0.304 +- 0.030 (k0), 0.335 +- 0.049 (rot4), 0.358 +- 0.003 (all8). The workflow keeps
all eight.

**Results.** `results/seg_cached/d4_ablation/`.

**Re-run.** `git checkout a81bd9e`, then from the repository root
`setsid nohup bash scripts/seg/run_d4_ablation.sh > results/seg_cached/d4_ablation/run.log 2>&1 &`.
