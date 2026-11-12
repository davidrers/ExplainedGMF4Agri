"""The draw of the country and zone samples: balanced and spread, from the 2021 labels.

Reads `label_blocks.parquet` and `label_block_zones.parquet` (from `label_blocks.py`). A country unit is one
country's part of one block; a zone unit is one zone's chips inside one country's part of one block, since a block
usually holds two or three GAEZ classes.

1. **Roles.** Every block gets one role for both matrices, from a fixed random number of its position: a quarter
   go to the test pool, the rest to the training pool. A block is thus a test block wherever it is drawn, so no
   model is trained on a chip that any model is tested on.
2. **Domains.** Each of the 19 countries, and each GAEZ AEZ33 class with enough chips in both pools.
3. **The same labelled area for every model** (decided 5 November 2026): `TRAIN_HA` of labelled crop land to train
   on and `VAL_HA` to validate on; a test set holds at least `TEST_HA` and at least `TEST_MIN_UNITS` units (blocks),
   whichever is larger. Labelled area counts the land on the draft crop list only.
4. **Balanced and spread.** For each domain and role, `R` candidate samples are spread over the domain: the units are
   ordered along a space-filling curve and taken at equal steps of their labelled area, from a random start. The
   candidate whose crop-group areas come closest to the whole domain's (total variation distance) is kept. Test
   first, then training and validation from the training pool.

    poetry run python docs/research/transferability/draw_samples.py

Writes `samples.csv` (model, role, country, block, chips, hectares), `sample_models.csv` (per model: chips, blocks,
hectares, countries, how far each sample's crop mix is from its domain's), `export_cells.csv` (every chip cell to
export: country, column, row, block, pool, and whether a country sample needs it) and four figures in `figures/`.
"""
from __future__ import annotations

import hashlib
import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

from crop_groups import GROUPS
from gaez_domains import DOMAINS, GAEZ, HERE, NUTS, aez33_labels

TRAIN_HA, VAL_HA, TEST_HA = 250_000, 37_500, 75_000   # labelled hectares on the draft crop list
TEST_MIN_UNITS = 50
# Zones left out of the zone matrix by the rule of 6 November 2026 (fewer than 4 supported species with 1 % or more
# of the zone; `crop_classes_support.csv`): drawn, so that the other draws stay as they are, but not exported
DROPPED_ZONES = {25: "Dominantly very steep terrain: grassland is the only species above 1 %"}
TEST_SHARE = 0.25
R = 500
SEED = 0
BLOCK = 8960.0
EXTENT = (2.55e6, 1.45e6, 5.95e6, 4.95e6)
CROPS = [g for g, _ in GROUPS]                  # the draft groups; "other" is left out of the balance
FAMILIES = {                                    # display only: eight families of the draft groups
    "Cereals": ["wheat", "durum wheat", "barley", "rye", "oats", "triticale"],
    "Maize": ["maize"],
    "Oilseeds": ["rapeseed", "sunflower", "soy"],
    "Root crops": ["potatoes", "sugar beet"],
    "Legumes": ["dry pulses", "forage legumes"],
    "Grassland": ["grassland"],
    "Permanent crops": ["orchards", "olive", "vineyards"],
    "Fallow, vegetables": ["fallow", "vegetables"],
}
FAMILY_COLOURS = ["#d4a017", "#e8743b", "#8e5ea2", "#7b4b2a", "#2e8b57", "#a6d47a", "#c0392b", "#9aa5b1"]


def prn(bx: np.ndarray, by: np.ndarray, tag: str) -> np.ndarray:
    """A fixed random number in [0, 1) per block position."""
    return np.array([int(hashlib.sha256(f"{SEED}|{tag}|{x}|{y}".encode()).hexdigest()[:12], 16) / 16**12
                     for x, y in zip(bx, by)])


def morton(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    x, y = bx - bx.min(), by - by.min()
    z = np.zeros(len(x), dtype=np.int64)
    for i in range(14):
        z |= ((x >> i) & 1) << (2 * i) | ((y >> i) & 1) << (2 * i + 1)
    return z


def shares(df: pd.DataFrame) -> np.ndarray:
    a = df[[f"ha_{g}" for g in CROPS]].sum().to_numpy(float)
    return a / max(a.sum(), 1e-9)


def balanced_spread(pool: pd.DataFrame, population: pd.DataFrame, target_ha: float, rng,
                    min_units: int = 1, prefer: set | None = None) -> pd.DataFrame:
    """The spread candidate whose crop mix is closest to the population's; all of the pool if it is too small.

    With `prefer` (blocks as (country, bx, by)), each step along the curve takes one of its preferred units when it
    has any, so a zone's training sample reuses blocks already drawn for the countries while keeping the zone's own
    spread, and so its own mix of countries.
    """
    pool = pool[pool.ha > 0]
    if pool.ha.sum() <= target_ha or len(pool) <= min_units:
        return pool
    pool = pool.iloc[np.argsort(morton(pool.bx.to_numpy(), pool.by.to_numpy()), kind="stable")]
    # Every unit has the same chance: picking in proportion to area would favour the most cropped blocks and, summed,
    # over-weight their crops (grassland came out 7 to 10 points short in Sweden and Slovakia that way)
    n = len(pool)
    m = min(n, max(min_units, round(target_ha / pool.ha.mean())))
    step = n / m
    ha = pool.ha.to_numpy()
    pop = shares(population)
    best, best_d, fallback, fallback_d = None, np.inf, None, np.inf
    if prefer is not None:
        # within each step, the preferred units first; a pick then falls among them whenever the step has any
        stratum = np.minimum((np.arange(n) / step).astype(int), m - 1)
        liked = np.array([k in prefer for k in zip(pool.country, pool.bx, pool.by)])
        order = np.lexsort((~liked, stratum))
        pool, ha, stratum, liked = pool.iloc[order], ha[order], stratum[order], liked[order]
        start = np.searchsorted(stratum, np.arange(m))
        count = np.bincount(stratum, minlength=m)
        n_liked = np.bincount(stratum, weights=liked, minlength=m).astype(int)
        span = np.where(n_liked > 0, n_liked, count)
    for _ in range(R):
        if prefer is not None:
            picks = start + np.minimum(np.floor(rng.uniform(0, 1, m) * span).astype(int), span - 1)
        else:
            # one unit inside each of m equal steps along the curve, each at its own random position (stratified, spread)
            picks = np.minimum(np.floor(step * (np.arange(m) + rng.uniform(0, 1, m))).astype(int), n - 1)
        cand = pool.iloc[np.unique(picks)]
        d = 0.5 * np.abs(shares(cand) - pop).sum()
        total = ha[np.unique(picks)].sum()
        if abs(total - target_ha) <= 0.1 * target_ha or m == min_units:
            if d < best_d:
                best, best_d = cand, d
        elif d < fallback_d:
            fallback, fallback_d = cand, d
    return best if best is not None else fallback


def draw(units: pd.DataFrame, population: pd.DataFrame, test_ok, train_ok, rng,
         prefer: set | None = None) -> dict[str, pd.DataFrame]:
    """Test set drawn on its own; training and validation reuse the `prefer` blocks where they can."""
    test = balanced_spread(units[test_ok & (units.role == "test")], population, TEST_HA, rng, TEST_MIN_UNITS)
    pool = units[train_ok & (units.role == "train")]
    train = balanced_spread(pool, population, TRAIN_HA, rng, prefer=prefer)
    val = balanced_spread(pool.drop(train.index), population, VAL_HA, rng, prefer=prefer)
    return {"test": test, "train": train, "val": val}


def quality(sample: pd.DataFrame, population: pd.DataFrame) -> dict:
    s, p = shares(sample), shares(population)
    big = p >= 0.01
    supported = (sample[[f"n_{g}" for g in CROPS]].sum().to_numpy() >= 30)
    return {"tvd": 0.5 * np.abs(s - p).sum(), "worst_pp": 100 * np.abs(s - p).max(),
            "groups_1pct": int(big.sum()), "groups_1pct_30_parcels": int((big & supported).sum())}


def with_roles(units: pd.DataFrame) -> pd.DataFrame:
    """Each unit's pool from its block's position, and its labelled area on the draft crop list."""
    test = prn(units.bx.to_numpy(), units.by.to_numpy(), "role") < TEST_SHARE
    return units.assign(role=np.where(test, "test", "train"), ha=units[[f"ha_{g}" for g in CROPS]].sum(axis=1))


def main() -> None:
    rng = np.random.default_rng(SEED)
    blocks = with_roles(pd.read_parquet(HERE / "label_blocks.parquet"))
    zones = with_roles(pd.read_parquet(HERE / "label_block_zones.parquet"))
    labels = aez33_labels()

    models, rows = [], []
    for c in DOMAINS:
        dom = blocks[blocks.country == c]
        models.append((f"country {c}", "country", c, dom, draw(dom, dom, True, True, rng)))
    # Zone training and validation samples reuse the blocks of the country training and validation samples (5 November
    # 2026: test sets stay independent draws, training may share)
    reuse = {(r.country, r.bx, r.by) for _, _, _, _, parts in models for role in ("train", "val")
             for r in parts[role].itertuples()}
    zone_ok = []
    for z, dom in zones.groupby("aez"):
        if z not in labels:
            continue
        pools = dom.groupby("role").ha.sum()
        test_units = (dom.role == "test").sum()
        if pools.get("test", 0) >= TEST_HA and test_units >= TEST_MIN_UNITS and pools.get("train", 0) >= TRAIN_HA + VAL_HA:
            zone_ok.append(z)
            models.append((f"zone {z}", "zone", f"{z} {labels[z]}", dom, draw(dom, dom, True, True, rng, reuse)))

    summary = []
    for name, kind, domain, dom, parts in models:
        for role, part in parts.items():
            rows.append(part.assign(model=name, kind=kind, domain=domain, sample=role))
        tr = parts["train"]
        summary.append({"model": name, "kind": kind, "domain": domain,
                        **{f"{r}_chips": int(p.chips.sum()) for r, p in parts.items()},
                        **{f"{r}_blocks": len(p) for r, p in parts.items()},
                        **{f"{r}_ha": round(float(p.ha.sum())) for r, p in parts.items()},
                        "train_countries": ", ".join(f"{k} {v:.0%}" for k, v in
                                                     (tr.groupby("country").chips.sum() / tr.chips.sum())
                                                     .sort_values(ascending=False).items() if v >= 0.01),
                        "groups_1pct": ", ".join(g for g, share in zip(CROPS, shares(dom)) if share >= 0.01),
                        **{f"train_{k}": v for k, v in quality(tr, dom).items()},
                        **{f"test_{k}": v for k, v in quality(parts["test"], dom).items()}})
    samples = pd.concat(rows, ignore_index=True)
    samples[["model", "kind", "domain", "sample", "country", "bx", "by", "aez", "chips", "ha", "climate"]].round(1).to_csv(
        HERE / "samples.csv", index=False)
    export_cells(samples)
    summary = pd.DataFrame(summary)
    summary.round(3).to_csv(HERE / "sample_models.csv", index=False)

    # Chips to export: a block drawn for a country brings all its chips; one drawn only for zones, those zones' chips
    key = ["country", "bx", "by"]
    by_country = samples[samples.kind == "country"].drop_duplicates(key)
    zone_only = samples[samples.kind == "zone"].drop_duplicates(key + ["aez"])
    zone_only = zone_only[~zone_only.set_index(key).index.isin(by_country.set_index(key).index)]
    pd.set_option("display.width", 250)
    print(summary[["model", "train_chips", "val_chips", "test_chips", "train_tvd", "test_tvd", "test_worst_pp",
                   "test_groups_1pct", "test_groups_1pct_30_parcels", "train_countries"]].round(3).to_string(index=False))
    cells = pd.read_csv(HERE / "export_cells.csv")
    print(f"\nzones drawn: {len(zone_ok)}, {len(zone_ok) - len(DROPPED_ZONES)} in the zone matrix; chips to export: "
          f"country samples {cells.country_sample.sum():,}, plus {(~cells.country_sample).sum():,} for the zone "
          f"samples, {len(cells):,} in all")
    figures(blocks, zones, samples, summary)


def export_cells(samples: pd.DataFrame) -> None:
    """The chip cells to export: all of a country unit's cells, only the zone's cells of a zone unit."""
    from gaez_domains import ROOT
    cells = pd.read_parquet(ROOT / "data/eurocrops/label_parcels_2021.parquet", columns=["country", "gx", "gy", "aez"])
    cells = cells.drop_duplicates(["country", "gx", "gy"])
    cells["bx"], cells["by"] = cells.gx // 4, cells.gy // 4
    key = ["country", "bx", "by"]
    c = cells.merge(samples[samples.kind == "country"][key + ["role"]].drop_duplicates(key), on=key)
    zs = samples[(samples.kind == "zone") & ~samples.aez.isin(DROPPED_ZONES)]
    z = cells.merge(zs[key + ["aez", "role"]].drop_duplicates(key + ["aez"]), on=key + ["aez"])
    out = pd.concat([c.assign(country_sample=True), z.assign(country_sample=False)])
    out = (out.sort_values("country_sample", ascending=False).drop_duplicates(["country", "gx", "gy"])
           .rename(columns={"gx": "col", "gy": "row", "role": "pool"}))
    out[["country", "col", "row", "bx", "by", "pool", "country_sample"]].sort_values(["country", "col", "row"]).to_csv(
        HERE / "export_cells.csv", index=False)


# ----------------------------------------------------------------------------------------------- figures

def outlines():
    nuts = gpd.read_file(NUTS)
    codes = [c for cs in DOMAINS.values() for c in cs]
    return nuts[nuts.LEVL_CODE == 0].to_crs(3035), nuts[nuts.NUTS_ID.isin(codes)].to_crs(3035)


def squares(df: pd.DataFrame) -> list[Rectangle]:
    return [Rectangle((x * BLOCK, y * BLOCK), BLOCK, BLOCK) for x, y in zip(df.bx, df.by)]


def panel(ax, land, cand, background, extent, pad=20e3, tint="#d9d9d9"):
    land.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.3)
    ax.add_collection(PatchCollection(squares(background), facecolor=tint, edgecolor="none"))
    cand.boundary.plot(ax=ax, color="#888888", linewidth=0.3)
    x0, y0, x1, y1 = extent
    w = max(x1 - x0, y1 - y0) / 2 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.set_xlim(cx - w, cx + w)
    ax.set_ylim(cy - w, cy + w)
    ax.set_axis_off()


def extent_of(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return df.bx.min() * BLOCK, df.by.min() * BLOCK, (df.bx.max() + 1) * BLOCK, (df.by.max() + 1) * BLOCK


ROLE_COLOURS = {"train": "#1f5fa8", "val": "#7fb2e5", "test": "#d62728"}


def figures(blocks, zones, samples, summary) -> None:
    land, cand = outlines()
    colours = {int(i["quantity"]): i["color"] for i in
               json.loads((GAEZ / "GAEZ-V5.AEZ33.style.json").read_text())["colorMap"]["items"]}
    legend = [Patch(color=ROLE_COLOURS["train"], label="training"), Patch(color=ROLE_COLOURS["val"], label="validation"),
              Patch(color=ROLE_COLOURS["test"], label="test"), Patch(color="#d9d9d9", label="labelled, not drawn")]

    # One panel per country model, zoomed to the country
    countries = list(DOMAINS)
    fig, axes = plt.subplots(4, 5, figsize=(20, 16.5))
    for ax, c in zip(axes.flat, countries):
        dom = blocks[blocks.country == c]
        panel(ax, land, cand, dom, extent_of(dom))
        s = samples[samples.model == f"country {c}"]
        for role, colour in ROLE_COLOURS.items():
            ax.add_collection(PatchCollection(squares(s[s["sample"] == role]), facecolor=colour, edgecolor="none"))
        r = summary.set_index("model").loc[f"country {c}"]
        ax.set_title(f"{c}: {r.train_chips:,} / {r.val_chips:,} / {r.test_chips:,} chips", fontsize=11)
    for ax in axes.flat[len(countries):]:
        ax.set_axis_off()
    axes.flat[-1].legend(handles=legend, loc="center", frameon=False, fontsize=12, title="Blocks (training / validation / test chips)")
    fig.suptitle("Country models: where each one's training, validation and test blocks lie", fontsize=15, x=0.02, ha="left")
    fig.savefig(HERE / "figures/samples_countries.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    # One panel per zone model, zoomed to the zone; the zone itself in a tint of its GAEZ colour
    zone_models = summary[summary.kind == "zone"]
    n = len(zone_models)
    cols = 4
    fig, axes = plt.subplots(-(-n // cols) + (n % cols == 0), cols, figsize=(20, 5.3 * (-(-n // cols) + (n % cols == 0))))
    for ax, r in zip(axes.flat, zone_models.itertuples()):
        z = int(r.domain.split()[0])
        dom = zones[zones.aez == z]
        hexc = colours[z]
        tint = "#" + "".join(f"{int(255 - 0.45 * (255 - int(hexc[i:i + 2], 16))):02x}" for i in (1, 3, 5))
        panel(ax, land, cand, dom, extent_of(dom), pad=40e3, tint=tint)
        s = samples[samples.model == r.model]
        for role, colour in ROLE_COLOURS.items():
            ax.add_collection(PatchCollection(squares(s[s["sample"] == role]), facecolor=colour, edgecolor="none"))
        ax.set_title(f"{r.domain}\n{r.train_chips:,} / {r.val_chips:,} / {r.test_chips:,} chips · "
                     f"{r.train_countries.count(',') + 1} countries", fontsize=10.5)
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    axes.flat[-1].legend(handles=legend[:3] + [Patch(color="#e8e0b0", label="the zone, not drawn (tint of its GAEZ colour)")],
                         loc="center", frameon=False, fontsize=12, title="Blocks (training / validation / test chips)")
    fig.suptitle("Zone models: where each one's blocks lie (a block brings only its chips of that zone)", fontsize=15,
                 x=0.02, ha="left")
    fig.savefig(HERE / "figures/samples_zones.png", dpi=80, bbox_inches="tight")
    plt.close(fig)

    crops_figure(blocks, samples, summary, "country", lambda d: blocks[blocks.country == d])
    crops_figure(zones, samples, summary, "zone", lambda d: zones[zones.aez == int(d.split()[0])])


def family_shares(df: pd.DataFrame) -> np.ndarray:
    a = np.array([df[[f"ha_{g}" for g in gs]].to_numpy().sum() for gs in FAMILIES.values()], float)
    return a / max(a.sum(), 1e-9)


def crops_figure(units, samples, summary, kind, domain_of) -> None:
    """Per model, three bars: the whole domain, the training sample and the test sample, by crop family."""
    rows = summary[summary.kind == kind]
    fig, ax = plt.subplots(figsize=(13, 0.33 * 3.6 * len(rows) + 1.5))
    y, ticks, names = 0, [], []
    for r in rows.itertuples():
        part = samples[samples.model == r.model]
        for label, df in (("whole " + kind, domain_of(r.domain)), ("training", part[part["sample"] == "train"]),
                          ("test", part[part["sample"] == "test"])):
            left = 0.0
            for share, colour in zip(family_shares(df), FAMILY_COLOURS):
                ax.barh(y, share, left=left, color=colour, height=0.85, edgecolor="white", linewidth=0.6)
                left += share
            ticks.append(y)
            names.append(f"{r.domain}  ·  {label}" if label.startswith("whole") else label)
            y += 1
        y += 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the labelled area on the draft crop list, by family")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=c, label=f) for f, c in zip(FAMILIES, FAMILY_COLOURS)], loc="lower center",
              bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, fontsize=10)
    fig.savefig(HERE / f"figures/samples_crops_{kind}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
