"""Data loading, class scheme handling and the Phase 1 splitting protocol."""

from gfm4agri.data.catalogue import CatalogueError, load_catalogue
from gfm4agri.data.class_scheme import ClassScheme, ClassSchemeError, load_class_scheme
from gfm4agri.data.representations import (
    BRIDGES,
    REPRESENTATIONS,
    REPRESENTATION_GROUPS,
    FeatureCacheKey,
    feature_cache_dir,
    group_of,
    required_buffer_m,
)
from gfm4agri.data.splits import (
    ALL_BUDGET,
    K_GRID,
    CountrySplit,
    SplitBundle,
    SplitConfig,
    build_split,
    build_support_order,
    intersect_label_space,
    read_split_bundle,
    support_at,
    write_split_bundle,
)

__all__ = [
    "ALL_BUDGET",
    "BRIDGES",
    "REPRESENTATIONS",
    "REPRESENTATION_GROUPS",
    "FeatureCacheKey",
    "feature_cache_dir",
    "group_of",
    "required_buffer_m",
    "K_GRID",
    "CatalogueError",
    "ClassScheme",
    "ClassSchemeError",
    "CountrySplit",
    "SplitBundle",
    "SplitConfig",
    "build_split",
    "build_support_order",
    "intersect_label_space",
    "load_catalogue",
    "load_class_scheme",
    "read_split_bundle",
    "support_at",
    "write_split_bundle",
]
