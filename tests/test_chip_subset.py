"""A pilot chip set is whole blocks of its parent and reads exactly like a chip set."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from conftest import SYN_SPLIT
from gfm4agri.data.chip_subset import choose_blocks, write_subset


def test_whole_blocks_are_drawn_per_partition_and_seeded(split_chipset):
    chips = pd.read_csv(split_chipset / "splits" / SYN_SPLIT / "chips.csv", dtype={"block_id": str})
    a = choose_blocks(chips, {"pool": 1, "val": 1, "test": 1}, seed=0)
    assert a == {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}
    assert a == choose_blocks(chips, {"pool": 1, "val": 1, "test": 1}, seed=0)
    with pytest.raises(ValueError, match="2 blocks wanted"):
        choose_blocks(chips, {"pool": 2}, seed=0)


def test_the_subset_is_a_chip_set_of_links_with_a_restricted_split(split_chipset):
    split = split_chipset / "splits" / SYN_SPLIT
    dest = split_chipset.parent / "SYN_2021_mini"
    write_subset(split_chipset, dest, split, {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}, 0)
    links = sorted(os.listdir(dest / "chips"))
    assert len(links) == 6 * 4 and all((dest / "chips" / f).is_symlink() for f in links)
    assert os.readlink(dest / "chips" / links[0]).startswith("../../SYN_2021/chips/")
    assert (dest / "manifest.json").read_bytes() == (split_chipset / "manifest.json").read_bytes()
    sub = dest / "splits" / SYN_SPLIT
    assert (sub / "training_data.txt").read_text().split() == ["EE_00000_00000", "EE_00001_00000"]
    rec = json.loads((sub / "split.json").read_text())
    assert rec["lists"] == {"training": 2, "validation": 2, "test": 2}
    assert rec["subset"]["parent"].endswith("SYN_2021")
    assert json.loads((dest / "subset.json").read_text())["blocks"]["val"] == ["1_0"]
    with pytest.raises(FileExistsError):
        write_subset(split_chipset, dest, split, {"pool": ["0_0"]}, 0)


def test_the_datamodule_reads_the_subset(split_chipset):
    pytest.importorskip("terratorch")
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

    split = split_chipset / "splits" / SYN_SPLIT
    dest = split_chipset.parent / "SYN_2021_mini"
    write_subset(split_chipset, dest, split, {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}, 0)
    dm = EuroCropsSegDataModule(dest, "tessera_v1", normalisation="chips", batch_size=1,
                                num_workers=0, split_dir=dest / "splits" / SYN_SPLIT)
    dm.setup("fit")
    dm.setup("test")
    assert (len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)) == (2, 2, 2)
