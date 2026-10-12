# TerraMind v1 large on 12 monthly against 24 half-month frames, on a sample of Estonia, 12 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. `run.py` gives the datamodule
the frame count of a half-month chip set, in memory.

**Question.** Does a frozen TerraMind v1 large segment crops better from 24 half-month Sentinel-2 frames than from
12 monthly ones, and does the answer change with the label budget?

- **For 24:** half months resolve short events that a monthly median dilutes, such as the three-week potato canopy
  of notebook 06, and the dates of emergence and harvest.
- **Against 24:** they leave more frames interpolated. On these chips, 545 of 733 have at least one half month
  with no acquisition at all.

**What it varies.** The temporal grid of the input, and with it what follows from the frame count.

| | 12 frames | 24 frames |
|---|---|---|
| Chip set | `EE_2021_sample_rev` | `EE_2021_sample_24` |
| Frame | calendar month | days 1 to 15, day 16 to the month's end |
| Chips, split | the 733 chips of `EE_2021_sample`, 413 training, 88 validation, 207 test, split `blocks4_buf1600_seed0` | the same |
| Screening | `REVISED` | `REVISED` |
| Composite windows | 2 x 2 chip groups | the same groups, so identical resampling |
| Encoder | TerraMind v1 large, frozen, run once per frame (`terramind_v1_large`) | the same |
| `ChannelBottleneck` | 12 x 1,024 channels to 768 per layer, 39 M parameters | 24 x 1,024 to 768, 78 M parameters |
| Decoder | UNet, 12.9 M parameters | the same |
| Cache | about 69 GB | about 139 GB |

The arm, budgets, draws, seeds and epochs are those of `experiment.yaml`: K = 100, 20 and 5 %, draw 0, seeds 0 to
2, 15 epochs, checkpoint of lowest validation loss. The parcels drawn at each budget are identical on both sets,
because the chips, the split and the parcel table are the same, so the two frame grids are compared fit by fit.

The bottleneck doubles with the frame count: it projects the concatenated frames to 768 channels. This is part of
what 24 frames cost in this design, not a separate change. A difference is therefore the effect of the temporal
grid in this architecture, not of the input alone.

The two chip sets are built in
[`../2026-10-12_s2_semimonthly/`](../2026-10-12_s2_semimonthly/README.md).

**How to run.** The data and the code reach the cluster first. `EE_2021_sample_rev` links into `EE_2021_rev`, which
links into `EE_2021`; `EE_2021_sample_24` links into `EE_2021` and `EE_2021_sample`.

    bash scripts/cluster/push_data.sh eurocrops_chips/EE_2021_rev eurocrops_chips/EE_2021_sample_rev eurocrops_chips/EE_2021_sample_24
    ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && git pull --ff-only'
    ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && bash scripts/cluster/submit.sh experiments/2026-10-12_terramind_12v24/experiment.yaml EE_2021_sample_rev'
    ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && bash scripts/cluster/submit.sh experiments/2026-10-12_terramind_12v24/experiment.yaml EE_2021_sample_24'
    bash scripts/cluster/pull_results.sh 2026-10-12_terramind_12v24 EE_2021_sample_rev
    bash scripts/cluster/pull_results.sh 2026-10-12_terramind_12v24 EE_2021_sample_24
    poetry run python experiments/2026-10-12_terramind_12v24/report.py

On the hub the same runs as `run.py -e experiments/2026-10-12_terramind_12v24/experiment.yaml --chips
data/eurocrops_chips/<set>`.

**Checked before submission, on the hub.**
- Both dry runs list the nine cells.
- The datamodule reads T = 24 from the half-month set and T = 12 from the monthly one.
- TerraMind v1 large builds with 24 frames and runs a forward pass on the A4000, peaking at 2.6 GB.
- `pytest` passes.

**Results.** `results/2026-10-12_terramind_12v24/EE_2021_sample_rev/` and `.../EE_2021_sample_24/`.

**Report.** `report.py` reads both sets and reports them as two arms, `12 monthly frames` and
`24 half-month frames`, with the paired difference 24 minus 12. `REPORT.md` is written from its figures.

**Outcome.** Pending: submitted to the cluster.
