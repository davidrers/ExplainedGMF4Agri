# Re-export of the Estonian Sentinel-2 composites under the revised screening, 12 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for this export. `run.py` runs
the core export script, `scripts/hub/build_country_chips.py`, with two of its names swapped in its own process.

**Question.** This is a data product rather than a fit. It replaces the Sentinel-2 composites of `EE_2021` with
composites built under `REVISED`, the screening chosen by the study in
[`../2026-10-12_s2_screening/`](../2026-10-12_s2_screening/REPORT.md), and records how they were made.
[`PROCESS.md`](PROCESS.md) explains the download and the processing step by step, with the reason for each step.

**What it varies.** Only the screening of the monthly composites, the `_merged.tif` of every chip. The new set has:
- the same 7,402 chips as `EE_2021`, the same 12-month layout, the same bands and the same split;
- the same labels, re-rasterised and checked identical;
- Sentinel-1, TESSERA and AlphaEarth linked from the parent.

| | `EE_2021` | `EE_2021_rev` |
|---|---|---|
| Clear observation | SCL not in 0, 1, 3, 8, 9, 10, 11 | SCL 4, 5 or 6, at least 2 px from cloud, shadow, cirrus and snow, passing the Fmask snow test and the haze-optimised transform |
| One observation is | one catalogue item | one overpass: tile copies and reprocessings merged, flagged if any copy is flagged |
| Read | chip by chip | 2 x 2 groups of adjacent chips in one read |
| Everything else | | unchanged: search, warp, BOA offset, per-pixel monthly median, linear gap filling, int16 layout |

**How to run, on the hub.** Detached, because the export takes about a day. The first pass is the 733 chips of
`EE_2021_sample`, so that the 12-month counterpart of the 24-period set in
[`../2026-10-12_s2_semimonthly/`](../2026-10-12_s2_semimonthly/README.md) exists early; the second pass is
everything else, and skips what is on disk.

    cd "/data/private/THESIS - ExplainedGMF4Agri"
    export PATH="$HOME/.local/bin:$PATH"
    setsid nohup bash -c 'poetry run python experiments/2026-10-12_s2_revised_export/run.py --workers 8 --group-size 2 --only-chips-of data/eurocrops_chips/EE_2021_sample && poetry run python experiments/2026-10-12_s2_revised_export/run.py --workers 8 --group-size 2' > data/eurocrops_chips/EE_2021_rev/export.log 2>&1 < /dev/null &
    tail -f data/eurocrops_chips/EE_2021_rev/export.log          # progress and ETA
    poetry run python experiments/2026-10-12_s2_revised_export/check.py   # once finished

A failed chip is listed in `EE_2021_rev/failures.json`. Running the same command again exports only what is
missing.

**Results.** `data/eurocrops_chips/EE_2021_rev/`: `chips/` (composite, labels and report per chip, links for the
rest), `manifest.json` (stamped with the screening and the parent), `export.log`, `failures.json` if any.
`check.json` in this folder holds the post-export checks.

**Report.** `REPORT.md`, once the export and `check.py` have finished: what was exported, how long it took,
failures, the checks, and the gap-filled shares against the parent.

**Outcome.** Pending: the export is running.

**Adding it to the main workflow** is a separate decision of the user. It would mean pointing the workflow at
`EE_2021_rev`, or renaming it into place, and adding a `--screening` option to the export script.
