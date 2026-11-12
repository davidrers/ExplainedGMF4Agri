"""The crop classes of the transfer experiment: every HCAT name of the 2021 labels mapped to a species and a group.

Decided on 6 November 2026 (`transfer_plan.md`, section 3): 21 species in 9 groups, two levels in one model. Each
HCAT name, from EuroCrops v11 (HCAT3) and v2 (HCAT4) alike, gets one of three levels:

* **species**: one of the 21 species, and its group;
* **group only**: the name gives the group but not the species (Lithuania's "unspecified cereals", a bare
  "legumes"); the model learns such land at the group level only;
* **ignore**: off the list, like unlabelled land (forest, unknown, crops that miss the 0.5 % rule, names
  beginning "other").

The species list follows the rule fixed before any result: a crop enters if it covers at least 0.5 % of the
labelled area in at least two of the 19 countries. Names are assigned by the rules below, in order, then by the
hand-checked overrides, which take precedence.

    poetry run python docs/research/transferability/crop_classes.py

Writes `crop_classes.csv` (name, level, species, group, largest share in a country, the countries) and
`crop_classes_support.csv` (per model: classes with support in its training and test samples) next to this file,
and prints the share of every country's labelled land that each level keeps.
"""
from __future__ import annotations

import re

import pandas as pd

from gaez_domains import HERE

GROUPS = {
    "Cereals": ["wheat", "durum wheat", "barley", "rye", "oats", "triticale"],
    "Maize": ["maize"],
    "Oilseeds": ["rapeseed", "sunflower", "soy"],
    "Root crops": ["potatoes", "sugar beet"],
    "Legumes": ["dry pulses", "forage legumes"],
    "Grassland": ["grassland"],
    "Fallow": ["fallow"],
    "Vegetables and other arable crops": ["vegetables", "buckwheat"],
    "Permanent crops": ["orchards", "olive", "vineyards"],
}
SPECIES = [s for ss in GROUPS.values() for s in ss]
GROUP_OF = {s: g for g, ss in GROUPS.items() for s in ss}

# Species rules, first match wins (HCAT3 and HCAT4 names)
SPECIES_RULES = [
    ("durum wheat", r"durum"),
    ("wheat", r"common_soft_wheat|(^|_)spelt$"),
    ("barley", r"(^|_)barley$"),
    ("rye", r"(^|_)rye$"),
    ("oats", r"(^|_)oats$"),
    ("triticale", r"(^|_)triticale$"),
    ("maize", r"maize"),
    ("rapeseed", r"rapeseed"),
    ("sunflower", r"^sunflower$"),
    ("soy", r"^soy"),
    ("potatoes", r"^potatoes$"),
    ("sugar beet", r"^sugar_beet$"),
    ("dry pulses", r"^(beans|peas|lupins|lentils|chickpeas|other_dry_pulses|legumes_dried_pulses_protein_crops)$"),
    ("forage legumes", r"alfalfa|clover|vetches|sainfoin|serradella|galega|melilot|birdsfoot|legumes_harvested_green"),
    ("grassland", r"grassland|pasture|meadow|poaceae|lolium|festuca|festulolium|timothy|cocksfoot|elymus|"
                  r"^temporary_grass$|^sod_turf$"),
    ("fallow", r"fallow"),
    ("buckwheat", r"buckwheat"),
    ("vegetables", r"vegetables|strawberr|cucurbit|cucumber|pumpkin|zucchini|melon|brassica|cabbage|cauliflower|"
                   r"broccoli|brussels|kale|kohlrabi|onion|garlic|leek|shallot|alliums|carrot|parsnip|radish|turnip|"
                   r"celer|lettuce|spinach|chicor|endive|tomato|aubergine|capsicum|pepper|asparagus|artichoke|"
                   r"root_vegetables|salads|beans_green|peas_green|rhubarb|fennel"),
    ("orchards", r"orchard|apple|pears$|^pears|plum|cherr|peach|nectarine|apricot|quince|fig$|kiwi|^nuts$|almond|"
                 r"hazel|walnut|chestnut|pistachio|pecan|citrus|orange|clementine|mandarin|lemon|carob|avocado|"
                 r"pomegranate|berries_berry|blackcurrant|redcurrant|gooseberr|raspberr|blackberr|blueberr|"
                 r"cranberr|aronia|hippophae|ribes|rubus|vaccinium"),
    ("olive", r"olive"),
    ("vineyards", r"vineyard|grape"),
]
# Names that give the group but not the species
GROUP_ONLY_RULES = [
    ("Cereals", r"unspecified_cereals|^cereal$|^cereals$|mixed_grains"),
    ("Legumes", r"^legumes$|unspecified_legumes$"),
    ("Oilseeds", r"^oilseed_crops$"),
    ("Permanent crops", r"^permanent_crops_perennial$|^permanent_crops$"),
]
# Hand-checked overrides, which take precedence: name -> (level, species or group, reason)
OVERRIDES = {
    "shrubberries_shrubs": ("ignore", None, "scrub, not orchards (the draft's 'berr' matched it)"),
    "other_shrubs": ("ignore", None, "scrub"),
    "unspecified_legumes_harvested_green": ("species", "forage legumes", "forage legumes, kind unnamed"),
    "plants_harvested_green": ("ignore", None, "green fodder of any crop: grass, legume or cereal"),
    "kitchen_gardens": ("ignore", None, "mixed small plots"),
    "sweet_potatoes": ("ignore", None, "not potatoes; off the list"),
    "beta_vulgaris": ("ignore", None, "generic beets: sugar, fodder or chard"),
    "other_beta_vulgaris": ("ignore", None, "other beets"),
    "mangelwurzel_beets": ("ignore", None, "fodder beet, off the list"),
    "green_beans": ("species", "vegetables", "a vegetable"),
}
IGNORE_FIRST = r"^other_(?!dry_pulses)"   # "other ..." names a crop outside the listed ones


def classify(name) -> tuple[str, str | None, str | None]:
    """(level, species, group) of one HCAT name."""
    if not isinstance(name, str):
        return "ignore", None, None
    if name in OVERRIDES:
        level, target, _ = OVERRIDES[name]
        if level == "species":
            return level, target, GROUP_OF[target]
        return (level, None, target) if level == "group only" else ("ignore", None, None)
    if re.search(IGNORE_FIRST, name) and not re.search(r"other_poaceae_grasses|other_orchards_fruits", name):
        return "ignore", None, None
    for group, pattern in GROUP_ONLY_RULES:
        if re.search(pattern, name):
            return "group only", None, group
    for species, pattern in SPECIES_RULES:
        if re.search(pattern, name):
            return "species", species, GROUP_OF[species]
    return "ignore", None, None


def support(samples: pd.DataFrame, hcat: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per model: species with support, that is at least 30 parcels in its training sample and at least 30 parcels
    in at least 5 blocks of its test sample (the scoring rule of a country pair). A zone model enters the zone matrix
    if at least 4 species are supported and each covers at least 1 % of the zone's labelled area on the list."""
    sp = table.set_index("hcat").species
    h = hcat.assign(species=hcat.hcat.map(sp)).dropna(subset=["species"])
    key = ["country", "bx", "by"]
    rows = []
    for model, s in samples.groupby("model", sort=False):
        kind = s.kind.iloc[0]
        cols = key + (["aez"] if kind == "zone" else [])
        part = h.merge(s[cols + ["sample"]], on=cols)
        tr = part[part["sample"] == "train"].groupby("species").parcels.sum()
        te = part[part["sample"] == "test"]
        te_parcels = te.groupby("species").parcels.sum()
        te_blocks = te[te.parcels > 0].drop_duplicates(key + ["species"]).groupby("species").size()
        train_ok = set(tr[tr >= 30].index)
        test_ok = set(te_parcels[te_parcels >= 30].index) & set(te_blocks[te_blocks >= 5].index)
        both = [x for x in SPECIES if x in train_ok and x in test_ok]
        dom = h[h.country == s.domain.iloc[0]] if kind == "country" else \
            h[h.aez == int(str(s.domain.iloc[0]).split()[0])]
        area = dom.groupby("species").area_ha.sum()
        big = set(area[area >= 0.01 * area.sum()].index)
        main = [x for x in both if x in big]
        rows.append({"model": model, "kind": kind, "species_train": len(train_ok), "species_test": len(test_ok),
                     "species_both": len(both), "species_both_1pct": len(main),
                     # the rule was agreed for zones that are almost one class; countries all stay (Ireland, three
                     # species above 1 %, would fail it)
                     "enters_matrix": kind == "country" or len(main) >= 4,
                     "species": ", ".join(both), "species_1pct": ", ".join(main)})
    return pd.DataFrame(rows)


def main() -> None:
    hcat = pd.read_parquet(HERE / "label_blocks_hcat.parquet")
    c = hcat.groupby(["country", "hcat"]).agg(ha=("area_ha", "sum")).reset_index()
    tot = c.groupby("country").ha.sum()
    c["pct"] = 100 * c.ha / c.country.map(tot)
    names = c.groupby("hcat").agg(ha=("ha", "sum"), max_pct=("pct", "max")).reset_index()
    names["countries"] = names.hcat.map(lambda n: ", ".join(
        f"{r.country} {r.pct:.1f}" for r in c[c.hcat == n].sort_values("pct", ascending=False).head(5).itertuples()))
    names[["level", "species", "group"]] = names.hcat.apply(lambda n: pd.Series(classify(n)))
    names["reason"] = names.hcat.map(lambda n: OVERRIDES.get(n, (None, None, ""))[2])
    names = names.sort_values(["level", "group", "species", "ha"], ascending=[False, True, True, False])
    names[["hcat", "level", "species", "group", "max_pct", "countries", "reason"]].round(2).to_csv(
        HERE / "crop_classes.csv", index=False)

    c = c.merge(names[["hcat", "level"]], on="hcat")
    keep = c.pivot_table(index="country", columns="level", values="pct", aggfunc="sum", fill_value=0)
    keep["species level"] = keep.get("species", 0)
    keep["group level"] = keep.get("species", 0) + keep.get("group only", 0)
    print(keep[["species level", "group level"]].round(1).T.to_string())
    sp_rule = c[c.level == "species"].merge(names[["hcat", "species"]], on="hcat").groupby(
        ["country", "species"]).pct.sum().unstack(fill_value=0)
    print("\ncountries with >= 0.5 % per species:", (sp_rule >= 0.5).sum().sort_values().to_dict())

    samples = pd.read_csv(HERE / "samples.csv")
    s = support(samples, hcat, names)
    s.to_csv(HERE / "crop_classes_support.csv", index=False)
    pd.set_option("display.width", 250)
    print(s[["model", "species_train", "species_test", "species_both", "species_both_1pct", "enters_matrix"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
