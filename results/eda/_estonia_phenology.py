"""Estonia-focused phenology: mean NDVI per top-5 HCAT class across 2021.

Samples up to N parcels per class, resamples each NDVI series to weekly bins, then
plots mean ± std per class on the same axis. Output: figures/11_estonia_phenology.png.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")
random.seed(0)
np.random.seed(0)

HERE = Path(__file__).parent.resolve()
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("EUROCROPSML_DATA", REPO / "data" / "eurocropsml")).expanduser() / "preprocess"
RED, NIR = 3, 7  # B04, B08 in 0-indexed S2 order

# Estonia parcels start with "EE"; we read filenames directly — fast.
print("scanning Estonia parcels…")
ee_files = [p for p in DATA_DIR.glob("EE*.npz")]
print(f"  {len(ee_files):,} Estonia .npz")

# class id is the last token before .npz
df = pd.DataFrame({"path": ee_files})
df["cls"] = df.path.map(lambda p: p.name.rsplit("_", 1)[1].split(".")[0])
top5 = df["cls"].value_counts().head(5).index.tolist()
print(f"  top-5 EE classes: {top5}")

PER_CLASS = 400  # sample budget per class
WEEKS = pd.date_range("2021-01-01", "2021-12-31", freq="W").to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
colors = sns.color_palette("tab10", len(top5))

# (a) Phenology mean ± std
for ax_phen, cls, color in zip([axes[0]] * 5, top5, colors):
    sub = df[df.cls == cls]
    pick = sub.sample(min(PER_CLASS, len(sub)), random_state=0)
    weekly = np.full((len(pick), len(WEEKS)), np.nan, dtype=float)
    for i, p in enumerate(pick["path"]):
        with np.load(p) as z:
            data = z["data"].astype(float); dates = pd.to_datetime(z["dates"])
        ndvi = (data[:, NIR] - data[:, RED]) / (data[:, NIR] + data[:, RED] + 1e-9)
        s = pd.Series(ndvi, index=dates).resample("W").mean().reindex(WEEKS)
        weekly[i] = s.values
    mean = np.nanmean(weekly, axis=0)
    std = np.nanstd(weekly, axis=0)
    ax_phen.plot(WEEKS, mean, lw=2, color=color, label=f"HCAT {cls} (n={len(pick)})")
    ax_phen.fill_between(WEEKS, mean - std, mean + std, color=color, alpha=0.15)

axes[0].set_title("Estonia · mean ± 1σ weekly NDVI per top-5 class (2021)")
axes[0].set_ylabel("NDVI"); axes[0].set_ylim(-0.1, 0.95)
axes[0].legend(fontsize=8, loc="lower center")
axes[0].tick_params(axis="x", labelrotation=30)

# (b) Distribution of observation dates (when does Estonia have data?)
all_dates = []
for p in df.sample(min(3000, len(df)), random_state=0)["path"]:
    with np.load(p) as z:
        all_dates.extend(pd.to_datetime(z["dates"]).tolist())
all_dates = pd.Series(all_dates)
axes[1].hist(all_dates, bins=52, color="slategray")
axes[1].set_title("Estonia · observation date distribution (post-cloud-filter)")
axes[1].set_ylabel("# observations")
axes[1].tick_params(axis="x", labelrotation=30)

plt.tight_layout()
out = FIG / "11_estonia_phenology.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
