#!/usr/bin/env python3
"""Generate synthetic segmentation volumes for testing.

Creates an HDF5 file with labeled tubular structures that simulate
neural segments for pipeline testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def generate_tubes(
    shape: tuple[int, int, int] = (64, 128, 128),
    num_tubes: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic tubular structures.

    Args:
        shape: Volume dimensions (z, y, x).
        num_tubes: Number of tubes to generate.
        seed: Random seed.

    Returns:
        Labeled volume with each tube having a unique label.
    """
    rng = np.random.RandomState(seed)
    volume = np.zeros(shape, dtype=np.uint32)

    for label_id in range(1, num_tubes + 1):
        # Random start and end points
        start = rng.rand(3) * np.array(shape) * 0.8 + np.array(shape) * 0.1
        end = rng.rand(3) * np.array(shape) * 0.8 + np.array(shape) * 0.1

        # Draw tube along path
        num_points = int(np.linalg.norm(end - start))
        for t in np.linspace(0, 1, max(num_points, 10)):
            point = start + t * (end - start)
            # Add slight curvature
            point += rng.randn(3) * 2
            point = np.clip(point, 0, np.array(shape) - 1).astype(int)

            # Draw sphere at each point
            radius = rng.randint(2, 5)
            z, y, x = point
            zr = slice(max(0, z - radius), min(shape[0], z + radius + 1))
            yr = slice(max(0, y - radius), min(shape[1], y + radius + 1))
            xr = slice(max(0, x - radius), min(shape[2], x + radius + 1))
            volume[zr, yr, xr] = label_id

    return volume


def introduce_gaps(volume: np.ndarray, num_gaps: int = 3, seed: int = 42) -> np.ndarray:
    """Introduce gaps in tubes to simulate segmentation breaks.

    Args:
        volume: Labeled volume.
        num_gaps: Number of gaps to introduce.
        seed: Random seed.

    Returns:
        Volume with gaps (label changes at gap points).
    """
    rng = np.random.RandomState(seed + 100)
    result = volume.copy()
    max_label = int(volume.max())

    for _ in range(num_gaps):
        # Pick a random location in a labeled region
        labeled_coords = np.argwhere(volume > 0)
        if len(labeled_coords) == 0:
            break
        idx = rng.randint(len(labeled_coords))
        center = labeled_coords[idx]

        # Create gap by relabeling a small region
        gap_size = rng.randint(3, 8)
        z, y, x = center
        zr = slice(max(0, z - gap_size), min(volume.shape[0], z + gap_size + 1))
        yr = slice(max(0, y - gap_size), min(volume.shape[1], y + gap_size + 1))
        xr = slice(max(0, x - gap_size), min(volume.shape[2], x + gap_size + 1))

        # Set gap region to 0 (background)
        result[zr, yr, xr] = 0

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--output", type=str, default="test_data/synthetic.h5")
    parser.add_argument("--shape", type=int, nargs=3, default=[64, 128, 128])
    parser.add_argument("--num-tubes", type=int, default=5)
    parser.add_argument("--num-gaps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Generating synthetic volume: shape={args.shape}, tubes={args.num_tubes}")
    volume = generate_tubes(tuple(args.shape), args.num_tubes, args.seed)
    volume = introduce_gaps(volume, args.num_gaps, args.seed)

    unique_labels = np.unique(volume)
    print(f"Labels: {len(unique_labels)} (including background)")
    print(f"Non-zero voxels: {np.count_nonzero(volume)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("labels", data=volume, compression="gzip")
        f.attrs["resolution"] = [30, 8, 8]
        f.attrs["description"] = "Synthetic segmentation volume for testing"

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
