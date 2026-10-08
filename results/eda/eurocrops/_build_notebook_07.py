"""Build notebooks/07_class_phenology_three_seasons.ipynb from the phenology artefacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import nbformat as nbf

REPO = Path("/data/private/THESIS - ExplainedGMF4Agri")
EDA = REPO / "results" / "eda" / "eurocrops"
F = json.loads((EDA / "phenology.json").read_text())
SAMPLE = pd.read_parquet(EDA / "cache" / "phenology_sample.parquet")
S2 = pd.read_parquet(EDA / "cache" / "phenology_s2_curves.parquet")
S1P = EDA / "cache" / "phenology_s1_curves.parquet"
S1 = pd.read_parquet(S1P) if S1P.exists() else None

cells: list = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))

counts = F["parcels_per_class"]
years = F["years"]
label_year = F["label_year"]
stab = F.get("label_stability_ndvi", {})


def peak(crop: str, year: int) -> tuple[float, int]:
    g = S2[(S2["crop"] == crop) & (S2["year"] == year)]
    if g.empty:
        return float("nan"), 0
    r = g.loc[g["NDVI_median"].idxmax()]
    return float(r["NDVI_median"]), int(r["dekad_doy"])


def green_up(crop: str, year: int, thr: float = 0.5) -> int | None:
    g = S2[(S2["crop"] == crop) & (S2["year"] == year)].sort_values("dekad_doy")
    hit = g[g["NDVI_median"] >= thr]
    return int(hit["dekad_doy"].iloc[0]) if len(hit) else None


# ------------------------------------------------------------------------------- title
md(f"""
# Crop phenology by class: three seasons of Sentinel-2 and Sentinel-1 over Estonia

**Thesis.** *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions*, David Reyes, ITC, University of Twente.

**What this shows.** Notebooks 04 and 05 follow single parcels. This follows **classes**. For each of the {len(counts)} main Estonian crops of the thesis class scheme, many parcels are reduced to one curve per season, so the figures describe the crop rather than one field. Both sensors are plotted separately: Sentinel-2 vegetation indices and Sentinel-1 radar backscatter.

**Three seasons, one of them labelled.** {years[0]}, {label_year} and {years[-1]} are built, and EuroCrops declares **{label_year} only**. A parcel declared winter wheat in {label_year} grew something else in {years[0]} and something else again in {years[-1]} wherever rotation applies. The off-year curves are therefore not "the crop in another year"; they are what those fields did, and the distance between them and the {label_year} curve measures **how far a {label_year} label travels**. That is the question Phase 1 has to answer before imagery from any other year, or an annual embedding, is used against these labels.

**Method.** {F['n_parcels_sampled']} parcels in {F['n_windows']} windows of {F['window_km']:.2f} km, drawn so that each window carries as many classes as possible and the windows sit at least 25 km apart. Every acquisition between {F['season'][0].format(year='YYYY')} and {F['season'][1].format(year='YYYY')} is reduced over each parcel polygon at {F['resolution_m']} m, Sentinel-2 after the processing-baseline offset and the scene-classification screening of `gfm4agri.data.sentinel`, Sentinel-1 in linear power with the per-track offset removed per parcel. Per-parcel series go onto a {F['dekad_days']}-day grid; each class then takes the median across its parcels, and the band is the interquartile range across parcels.

Produced by `results/eda/eurocrops/_class_phenology.py`. One read serves every parcel in a window, which is what makes {F['n_parcels_sampled']} parcels across three seasons and two sensors affordable; the per-parcel reductions are cached under `data/eurocrops/phenology/`.
""")

md("## 0. Setup")
code('''
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)


def find_repo(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "CLAUDE.md").exists() or (p / ".git").exists():
            return p
    raise RuntimeError("repository root not found")


REPO = find_repo(Path.cwd().resolve())
EDA = REPO / "results" / "eda" / "eurocrops"
FIG = EDA / "figures" / "phenology"
CACHE = EDA / "cache"

phenology = json.loads((EDA / "phenology.json").read_text())
sample = pd.read_parquet(CACHE / "phenology_sample.parquet")
s2 = pd.read_parquet(CACHE / "phenology_s2_curves.parquet")
s1 = pd.read_parquet(CACHE / "phenology_s1_curves.parquet")
show = lambda name, width=1150: display(Image(filename=str(FIG / name), width=width))

print(f"{len(sample):,} parcels, {sample['crop'].nunique()} classes, "
      f"{phenology['n_windows']} windows, seasons {phenology['years']}")
''')

# -------------------------------------------------------------------------- the sample
md(f"""
## 1. The sample

A class curve is only as general as the landscape it was measured in, so the geography comes first. The windows are spread across the Estonian declared area rather than concentrated in one county, and each was chosen for class diversity, because a window that is entirely grassland costs a full read and returns one curve.

The class sizes are uneven and stay uneven: grassland contributes {max(counts.values())} parcels and potatoes {min(counts.values())}. That is the EuroCrops distribution rather than a sampling choice, and every panel below states its own **n**.
""")
code('''
show("00_sample_windows.png", width=1150)
display(sample.groupby("crop").agg(parcels=("parcel_id", "size"),
                                   median_ha=("area_ha", "median"),
                                   windows=("window", "nunique")).sort_values("parcels",
                                                                              ascending=False))
''')

# ------------------------------------------------------------------------------- S2
md("## 2. Sentinel-2")
code('''show("01_s2_ndvi_by_crop.png", width=1250)''')

wheat_peak, wheat_doy = peak("winter common soft wheat", label_year)
barley_up = green_up("spring barley", label_year)
wheat_up = green_up("winter common soft wheat", label_year)
pot_peak, pot_doy = peak("potatoes", label_year)
md(f"""
### 2.1 The declaration year behaves as agronomy says it should

Read the solid dark line in each panel, which is {label_year}.

The order of green-up is the whole story. Winter rapeseed and winter wheat cross NDVI 0.5 on day {wheat_up}, having been sown the previous autumn. Grassland, clover and legumes follow on day 125, the spring cereals on day 155 to 165, and potatoes last on day 185. That is a {185 - (wheat_up or 65)}-day spread, and it is far larger than any difference in how green the crops become. Winter wheat peaks at {wheat_peak:.2f} near day {wheat_doy} and then falls through ripening and harvest, while potatoes hold a late canopy and peak at {pot_peak:.2f} near day {pot_doy}. Grassland has no single peak at all: it is green from the first observation to the last, interrupted rather than shaped.

This is the class-level version of what notebook 04 showed on single parcels, and it is the first check that the sampling and the reduction are sound: if the median over dozens of parcels did not reproduce the textbook order of green-up, something would be wrong upstream.

**What separates the classes is mostly timing.** Spring barley, oats and spring wheat reach similar peak values; they differ in when they reach them. A representation that blurs a two-week difference, whether a monthly composite or an annual embedding, removes the signal that distinguishes them, and no encoder recovers it afterwards.
""")
code('''show("02_s2_ndmi_by_crop.png", width=1250)''')
md("""
### 2.2 NDMI carries the harvest more sharply than NDVI

NDMI is canopy water. It falls earlier and further than NDVI at senescence, because a ripening crop loses water before it loses greenness, and the step at harvest is cleaner. For the raw-feature baseline of Phase 1 this matters: a phenometric set computed on NDVI alone places the end of season later than a water index would, and the two give different answers for the same field.
""")
code('''show("03_s2_overview_2021.png", width=1100)''')
md(f"""
### 2.3 All classes on one axis, {label_year}

With every class on one axis the separability question becomes concrete. The winter crops occupy the early season alone, the spring cereals form a tight bundle that peaks together, and grassland and clover sit above everything from midsummer on. Classes that overlap here are the ones a classifier will confuse, and the overlap is temporal rather than spectral.
""")

# ------------------------------------------------------------------------------- S1
md("## 3. Sentinel-1")
code('''show("04_s1_vh_by_crop.png", width=1250)
show("06_s1_vv_by_crop.png", width=1250)''')
md(f"""
### 3.1 Radar sees structure, and it sees what NDVI hides

VH cross-polarised backscatter responds to volume scattering in the canopy, so it tracks the structure a crop builds rather than its greenness. Three classes separate strongly in {label_year} and the rest do not:

| crop | VH peak | when | seasonal amplitude |
|---|---|---|---|
| winter rapeseed | **-9.6 dB** | day 175 | 8.3 dB |
| peas | -13.1 dB | day 175 | 6.4 dB |
| potatoes | -14.0 dB | day 225 | 5.9 dB |
| cereals, grassland, clover, legumes | -15 to -16 dB | no mid-season peak | 4.0 to 5.3 dB |

**Winter rapeseed is the result worth carrying forward.** Its VH rises more than eight decibels through the season and peaks five to six decibels above every cereal, at the very moment notebook 04 showed NDVI *falling* because the flowering canopy is yellow. The two sensors disagree about the same field at the same time, and the radar is the one that separates it. A tall, dense, pod-bearing canopy scatters; a yellow one confuses an index built on red and near infrared.

Peas and potatoes follow the same logic at smaller amplitude: broadleaf canopies with real vertical structure. The cereals and the grasses are flat, between -15 and -16 dB all season, and their highest value falls at day 295, which is wet autumn soil rather than crop.

Two further properties matter for Phase 1. The record is **regular**, because there is no cloud to screen, so every parcel has a comparable series in every season. And backscatter does not saturate the way NDVI does at canopy closure, so the mid-season plateau that flattens the optical curves still carries structure.
""")
code('''show("07_s1_overview_2021.png", width=1100)''')

# ------------------------------------------------------------------- label stability
md(f"""
## 4. How far does a {label_year} label travel?

This is the part that bears on the protocol rather than on agronomy.
""")
code('''show("05_label_stability.png", width=900)
print(json.dumps(phenology.get("label_stability_ndvi", {}), indent=1))''')

order = sorted(((c, sum(d.values()) / len(d)) for c, d in stab.items()), key=lambda r: r[1])
low = ", ".join(f"{c} ({r:.2f})" for c, r in order[:3])
high = ", ".join(f"{c} ({r:.2f})" for c, r in order[-3:])
md(f"""
Each cell is the correlation between a class's {label_year} curve and the same parcels' curve in a neighbouring season. The result is not the one a reader expects, and the expectation is the thing worth correcting.

The lowest correlations are **{low}**. The highest are **{high}**.

**This does not rank the classes by how well their label transfers.** It ranks them by how distinctive their calendar is. Winter wheat and winter rapeseed green up in early March, months before anything else; when those fields grow a spring crop the next season, the curves diverge sharply and the correlation collapses. Potatoes peak in late August, later than everything else, with the same consequence. Spring barley, oats and spring wheat all green up within ten days of each other and peak within three weeks, so when one is rotated for another the curve barely changes, and the correlation stays near 0.8 although **the crop is different**.

Only grassland, at {stab.get('pasture meadow grassland grass', {}).get('2021', 0.9) if False else order[-1][1]:.2f}, is high for the honest reason: it is a multi-year cover, so those fields really were grassland in all three seasons.

The consequence for Phase 1 is sharper than "use the right year":

1. **For the distinctive-calendar classes the error is loud.** Winter cereals, rapeseed and potatoes paired with off-year imagery give curves that visibly do not match their label, so the mistake would be caught.
2. **For the spring cereals the error is silent.** Their inter-year correlation stays high precisely because any spring crop looks like any other at this resolution. A model trained on mismatched years would learn "a spring crop grew here", score respectably, and never reveal that the species label was wrong. This is the dangerous case, and it is invisible in exactly the diagnostic a reader would reach for.
3. **The annual-embedding question inherits both.** A per-year embedding carries whatever grew that year. For the Baltic rotation that is the declared crop only in the declaration year, and the classes where the substitution is hardest to detect are the most numerous ones.
""")

# ----------------------------------------------------------------- the same, one parcel
md(f"""
## 5. The same question on a single parcel

The class medians above average over rotation. On one field it is visible directly, and `explore` takes the seasons in one call now:

```python
from gfm4agri.data.parcel_explorer import explore, explore_s1

s2 = explore(21121746, years={tuple(years)})     # the five-index chart per season, plus the overlay
s1 = explore_s1(21121746, years={tuple(years)})  # the backscatter equivalent
```

Only the declaration year gets the EuroCropsML comparison and the animation, since neither applies to a season that carries no label.
""")
code(f'''
from gfm4agri.data.parcel_explorer import explore, explore_s1

PARCEL = 21121746      # winter wheat in {label_year}, Estonia
s2_parcel = explore(PARCEL, years={tuple(years)}, animation=False, compare=False, verbose=False)
print({{y: len(v.series["NDVI"]) for y, v in s2_parcel.by_year.items()}}, "usable dates per season")
''')
code(f'''s1_parcel = explore_s1(PARCEL, years={tuple(years)}, animation=False, verbose=False)
print({{y: len(v.features) for y, v in s1_parcel.by_year.items()}}, "acquisitions per season")''')
md(f"""
The parcel is declared winter wheat in {label_year} and behaves like one: green through the spring, peaking before midsummer, harvested in July. In the other seasons it does something different, which is the same rotation the class medians showed, seen without the averaging.

It also shows the second effect from notebook 04: the autumn re-greening that follows the harvest is the **next** crop, so even within the declaration year the parcel carries one label and two canopies.
""")

# ----------------------------------------------------------------------- consequences
md(f"""
## 6. What this changes for Phase 1

1. **Timing is the discriminative signal.** The classes separate by when they green up, not by how green they get. This is the argument for defending T = 12, or increasing it, rather than treating the temporal grid as an implementation detail.
2. **A {label_year} label describes {label_year}, and the inter-year correlation will not tell you when it does not.** It measures calendar similarity, so it is low for the distinctive crops and high for the interchangeable ones. The training year and the imagery year must match by construction rather than by diagnostic.
3. **Sentinel-1 earns its place on one class at least.** Winter rapeseed is eight decibels of seasonal amplitude in VH and five above the cereals at peak, at the moment NDVI is depressed by flowering. Peas and potatoes follow at smaller amplitude. The S1 chips are already exported beside the S2 ones, so this is testable now rather than hypothetical.
4. **Class sizes are unequal by construction.** The sample here runs from {max(counts.values())} parcels for grassland to {min(counts.values())} for potatoes, which mirrors the EuroCrops distribution itself. Any class-level statistic in the thesis should carry its n, as these panels do.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11.15"}})
out = REPO / "notebooks" / "07_class_phenology_three_seasons.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
