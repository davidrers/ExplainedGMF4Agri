#!/usr/bin/env python3
"""Compute per-band mean and standard deviation from a directory of GeoTIFF files.

Usage:
    python compute_band_stats.py <data_dir> [--pattern "*.tif"] [--scale 1.0]

Output:
    Prints means and stds lists ready to paste into a TerraTorch YAML config.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError:
    print("Error: rasterio is required. Install with: pip install rasterio", file=sys.stderr)
    sys.exit(1)


def compute_stats(data_dir: str, pattern: str = "*.tif", scale: float = 1.0):
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        print(f"No files matching '{pattern}' found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} files in {data_dir}")

    # First pass: means
    pixel_sum = None
    pixel_count = None
    n_bands = None

    for i, f in enumerate(files):
        with rasterio.open(f) as src:
            data = src.read().astype(np.float64) * scale
            if pixel_sum is None:
                n_bands = data.shape[0]
                pixel_sum = np.zeros(n_bands)
                pixel_count = np.zeros(n_bands)
            mask = np.isfinite(data)
            for b in range(n_bands):
                valid = data[b][mask[b]]
                pixel_sum[b] += valid.sum()
                pixel_count[b] += valid.size
        if (i + 1) % 100 == 0:
            print(f"  Pass 1: processed {i + 1}/{len(files)} files...")

    means = pixel_sum / pixel_count

    # Second pass: stds
    pixel_sq_diff = np.zeros(n_bands)
    for i, f in enumerate(files):
        with rasterio.open(f) as src:
            data = src.read().astype(np.float64) * scale
            mask = np.isfinite(data)
            for b in range(n_bands):
                valid = data[b][mask[b]]
                pixel_sq_diff[b] += np.sum((valid - means[b]) ** 2)
        if (i + 1) % 100 == 0:
            print(f"  Pass 2: processed {i + 1}/{len(files)} files...")

    stds = np.sqrt(pixel_sq_diff / pixel_count)

    return means, stds, n_bands


def main():
    parser = argparse.ArgumentParser(description="Compute band statistics for TerraTorch configs")
    parser.add_argument("data_dir", help="Directory containing GeoTIFF files")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for files (default: *.tif)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale factor applied to pixel values (e.g., 0.0001)")
    args = parser.parse_args()

    means, stds, n_bands = compute_stats(args.data_dir, args.pattern, args.scale)

    print(f"\nBands: {n_bands}")
    print(f"\n# Paste into your TerraTorch YAML config:")
    print(f"means: [{', '.join(f'{m:.6f}' for m in means)}]")
    print(f"stds:  [{', '.join(f'{s:.6f}' for s in stds)}]")


if __name__ == "__main__":
    main()
