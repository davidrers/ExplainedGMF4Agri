# TerraMind v1 large on Sentinel-2, Sentinel-1 and both, on all of Estonia, 13 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. `run.py` registers the
Sentinel-1-only backbone and drops the Sentinel-2 input for it.

**Question.** How well does a frozen TerraMind v1 large segment crops from Sentinel-1 alone, and does adding
Sentinel-1 to Sentinel-2 help, at each label budget?

So far no fit in the thesis has given an encoder Sentinel-1 directly. It entered the results only inside the
TESSERA and AlphaEarth embeddings. TerraMind was pretrained with an S1RTC modality, and every chip has monthly S1 RTC
composites.

**What it varies.** The input modalities, and nothing else.

| Arm | Input | Encoder | Trainable |
|---|---|---|---|
| `terramind_v1_large` (core) | 12 monthly S2 composites, 12 bands | TerraMind v1 large, frozen, once per month | `ChannelBottleneck` 40.4 M, UNet decoder 12.9 M |
| `terramind_v1_large_s1` (`arms_run/`, registered by `run.py`) | 12 monthly S1 RTC composites, VV and VH in dB | the same, with the pretrained S1RTC patch embedding | the same |
| `terramind_v1_large_s2s1` (`arms/`, a core backbone) | both; TerraMind averages the tokens of the two modalities at every position | the same | the same |

**Shared by all three arms:**
- **Chips:** `EE_2021_rev`, all of Estonia under the `revised` screening: 4,898 training, 769 validation and 1,475
  test chips.
- **Protocol:** K = 100, 20 and 5 %, draw 0, seeds 0 to 2, 15 epochs, checkpoint of lowest validation loss.

The parcels drawn at a budget are the same for the three arms, so they are compared fit by fit.

**About the S1 input.** The S1 composites are per-pixel monthly medians over every valid acquisition in linear
power, both orbit directions mixed, converted to dB (pipeline.md section 5). Every month is observed. They were
not reviewed the way the S2 composites were.

**How to run.** On the cluster, one job per arm:

    bash scripts/cluster/submit.sh experiments/2026-10-13_terramind_s1_s2/experiment.yaml EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-10-13_terramind_s1_s2 EE_2021_rev
    poetry run python experiments/2026-10-13_terramind_s1_s2/report.py --chips EE_2021_rev

**Checked before submission, on the hub.**
- The dry run lists 27 cells.
- On `EE_2021_rev_mini`:
  - the S1 arm's encoder receives only `S1RTC` (2 bands x 12 months);
  - the S2 + S1 arm receives both;
  - all three build with the same 40.4 M bottleneck and 12.9 M decoder and run a forward pass.
- `pytest` passes.

**Results.** `results/2026-10-13_terramind_s1_s2/EE_2021_rev/`.

**Report.** `report.py` compares the three arms, with the paired differences S2 + S1 minus S2 and S1 minus S2.

**Outcome.** Pending: submitted to the cluster.
