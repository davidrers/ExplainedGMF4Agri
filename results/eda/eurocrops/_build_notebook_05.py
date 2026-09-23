"""Build notebooks/05_explore_parcel.ipynb: the per-parcel explorer, S2 and S1."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import nbformat as nbf

REPO = Path("/data/private/THESIS - ExplainedGMF4Agri")
PARCELS = pd.read_parquet(REPO / "results/eda/eurocrops/cache/gallery_parcels.parquet")
SHOWCASE = int(PARCELS.loc[PARCELS["crop"] == "winter_common_soft_wheat", "parcel_id"].iloc[0])

cells: list = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md(f"""
# Explore one parcel: Sentinel-2 and Sentinel-1 by identifier and year

**Thesis.** *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions*, David Reyes, ITC, University of Twente.

**What this is.** A working surface, not a report. Give it a EuroCrops parcel identifier and it pulls that parcel's season from the archive, charts it, animates it and puts it beside the EuroCropsML series. Change the number in the first cell of section 2 and everything below follows.

    from gfm4agri.data.parcel_explorer import explore, explore_s1, explore_both

    s2 = explore({SHOWCASE})            # chart, animation, comparison
    s1 = explore_s1({SHOWCASE})         # backscatter chart and animation
    s2, s1 = explore_both({SHOWCASE})   # both, optical overlaid on the radar chart

**Where it comes from.** The vendored `space_time_deepsearch` package ships
`vis.parcel_timeseries.explore`, which does exactly this for Austrian GSA parcels: it looks a parcel up by `KENNUNG` in a GeoPackage, joins the ISV inspection register, and draws five indices with Moran's I plus an NDVI animation. There is **no `explore_s1`** upstream; the radar workflow is `io.sentinel1.parcel_series`, which fetches, clips, reduces in linear power and removes the per-track offset.

`gfm4agri.data.parcel_explorer` supplies both for this thesis's data, keyed by the national identifier of the EuroCrops vector release. It reuses the vendored chart and animation rather than reimplementing them: they need five attributes from the parcel row and an inspection table that is allowed to be empty, so a EuroCrops parcel is presented to them under the names they expect, and the Austrian wording in the header is rewritten afterwards. The Sentinel-1 animation has no upstream counterpart and is written in the same visual language.

**Three corrections are applied that the vendored calls do not make on their own**, all in `gfm4agri.data.sentinel`:

| | what | why it matters |
|---|---|---|
| BOA offset | the reflectance offset is decided **per scene** from its processing baseline, and indices are then computed with `offset=0` | `compute_indices` hard-codes `-1000`, correct only from baseline 04.00 (January 2022). On 2021 data it inflates NDVI by roughly 0.45 |
| cloud screening | SCL classes 0, 1, 3, 8, 9, 10, 11 excluded per pixel, and a date is kept only when 60 % of the parcel survives | the vendored default blacklists nodata, saturated and snow only, so cloud enters the median |
| parcel mask | the reduction is taken over the declared polygon | the loader clips to a bounding box, which for a real parcel includes the neighbours |
""")

md("## 0. Setup")
code('''
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gfm4agri.data.parcel_explorer import (COUNTRY_ID, COUNTRY_NAME, explore, explore_both,
                                           explore_s1, find_parcel)

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
print("identifier column per country:", COUNTRY_ID)
print("outputs are written to results/explore/<country>_<parcel>_<year>/")
''')

md(f"""
## 1. Choosing a parcel

Any identifier from the EuroCrops layers works: `pollu_id` for Estonia, `PARCEL_ID` for Latvia, `OSA_ID` for Portugal. Lithuania is the exception and cannot be addressed this way, because it publishes only a field-block number, so an identifier there names a block rather than a parcel.

The country is inferred from the identifier when it is not given. The table below is the set of parcels the crop gallery of notebook 04 selected, which are convenient starting points: mid-sized, compact, and carrying the same HCAT code in both releases.
""")
code('''
parcels = pd.read_parquet(REPO / "results/eda/eurocrops/cache/gallery_parcels.parquet")
display(parcels.set_index("crop")[["label", "parcel_id", "hcat", "area_ha", "n_pixels", "compactness"]])

parcel, cc = find_parcel(parcels.loc[0, "parcel_id"])
print(f"lookup check: parcel {parcels.loc[0, 'parcel_id']} is in {COUNTRY_NAME[cc]}, "
      f"declared {parcel.iloc[0]['EC_trans_n']!r} -> HCAT {parcel.iloc[0]['EC_hcat_c']}")
''')

md(f"""
## 2. Sentinel-2

One call fetches the season, screens it, charts the five indices with Moran's I, animates NDVI with a cursor sweeping the series, and compares against EuroCropsML. The parcel below is winter wheat; change the identifier to follow another field.
""")
code(f'''
PARCEL = {SHOWCASE}          # winter wheat, Estonia. Change this to explore another parcel.
YEAR = 2021                  # EuroCrops carries one declaration year per country

s2 = explore(PARCEL, YEAR)
''')
md("""
The header names the declaration the parcel carries, the panels are the five indices reduced over the polygon, and the bottom panel is Moran's I of NDVI: how spatially organised the field is on each date, which the median cannot show. A hollow marker there means the arrangement is indistinguishable from random.

The animation underneath shows what produced the curve. The map is NDVI per date on a fixed colour scale, the black cursor marks the frame on screen, and the crosses on the series are dates the persistence filter rejected, which is the vendored convention: a drop that does not persist into the next acquisition is more likely cloud than a cut.
""")
code('''
print("dates kept:", len(s2.series["NDVI"]), "| dropped by screening:", s2.dates_dropped)
print("processing baselines:", s2.baseline.get("baselines"), "via", s2.baseline.get("baseline_source"))
display(s2.table.head(8))
''')

md("""
### 2.1 Against the EuroCropsML series

The same parcel, the same polygon, a different processing level: EuroCropsML reduces **L1C top-of-atmosphere** reflectance, this pull reduces **L2A surface reflectance**. The shapes agree; the levels do not, and the gap is the atmosphere.
""")
code('''
print({k: v for k, v in s2.comparison.items() if k != "matched"})
''')
md("""
Read the two numbers that matter. `pearson_r` says the trajectories are the same trajectory. `mean_difference` is positive because correcting for the atmosphere separates the red and near-infrared bands again and NDVI rises; L1C sits low. `roughness_here` against `roughness_eurocropsml` is the median absolute change between consecutive dates, and the EuroCropsML series is the rougher of the two, which is residual cloud rather than crop.
""")

md("""
## 3. Sentinel-1

Radar needs no cloud screening, so the observation record is regular where the optical one is ragged. `parcel_series` fetches the RTC cube, clips it, reduces each date in linear power, and removes the per-track offset; the chart shows both the acquisitions as they arrive and the corrected series, and the animation puts the VH map beside a cursor.

Two settings are not obvious and are handled inside `explore_s1`:

- `min_coverage` is measured over the **bounding box** of the area of interest. A parcel polygon fills only part of its own bounding box, so the value that is sensible for a rectangular area rejects every date. It stays at zero here and the polygon clip does the work.
- Several tracks see the same parcel from different geometries, each with a constant offset of around one decibel. Plotted raw the series is a sawtooth about the orbit rather than the crop, so `normalize_orbit` is applied and the `_adj` columns are what the chart draws.
""")
code('''
s1 = explore_s1(PARCEL, YEAR, ndvi=s2.series["NDVI"])
''')
code('''
print(f"{len(s1.features)} acquisitions on tracks {sorted(set(s1.features['relative_orbit']))}")
display(s1.features.head(6)[["date", "relative_orbit", "orbit_state", "vv_median_db",
                             "vh_median_db", "vh_median_db_adj", "n_pixels"]])
''')
md("""
VH tracks canopy development and falls at harvest, VV carries more of the soil and roughness signal, and both keep responding after NDVI has saturated. For this thesis the point is narrower than "radar is useful": TerraMind takes Sentinel-1 and Sentinel-2 together, so a modality that stays available through cloud is a live option for Phase 1 rather than a curiosity.
""")

md("""
## 4. What the call produced

Everything is written to `results/explore/<country>_<parcel>_<year>/`, which is git-ignored, so exploring a hundred parcels costs nothing in the repository. The animations are displayed above through downscaled copies; the full-resolution files are the ones listed here.
""")
code('''
out = s2.png.parent
for f in sorted(out.iterdir()):
    print(f"{f.stat().st_size / 1e6:6.2f} MB  {f.name}")
''')

md(f"""
## 5. Using it on other parcels

```python
explore(21084265)                       # potatoes, Estonia
explore(21652845, anim_band="NDMI")     # grassland; NDMI is the mowing-sensitive index
explore_s1(21534249, band="vv")         # winter rapeseed, VV instead of VH
explore_both(20583785)                  # spring barley, both sensors in one call

explore(PARCEL, season=("2021-04-01", "2021-09-30"))   # narrow the window
explore(PARCEL, cloud_cover_max=40, min_valid_fraction=0.8)  # stricter screening
explore(PARCEL, animation=False, compare=False)        # chart only, fastest
```

A full season over one parcel costs seconds for the search and a minute or two for the animation, so this is usable interactively while writing the protocol. Two caveats worth carrying: the scene classification layer misses hazy low-sun frames, so a date reported as clear is not always clear; and a parcel carries one declared label for the year, while the imagery may show two canopies when a catch crop or the next winter cereal follows the harvest.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11.15"}})
out = REPO / "notebooks" / "05_explore_parcel.ipynb"
nbf.write(nb, str(out))
print("wrote", out, "with", len(cells), "cells")
