#!/usr/bin/env python3
"""Diagnostic tool for examining accepted/rejected connections.

Reads pipeline output CSV files and prints summary statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Inspect pipeline connection decisions")
    parser.add_argument(
        "output_dir",
        type=str,
        help="Pipeline output directory containing CSV files",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["accepted", "rejected", "ambiguous"],
        default=None,
        help="Filter by connection status",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top connections to show",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load connections
    conn_path = output_dir / "connections.csv"
    if not conn_path.exists():
        print(f"No connections file found at {conn_path}")
        return

    df = pd.read_csv(conn_path)
    print(f"\n{'='*60}")
    print(f"Connection Summary ({len(df)} total)")
    print(f"{'='*60}")

    # Status counts
    status_counts = df["status"].value_counts()
    for status, count in status_counts.items():
        pct = count / len(df) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Score statistics
    print(f"\nComposite Score Statistics:")
    print(f"  Mean: {df['composite_score'].mean():.3f}")
    print(f"  Median: {df['composite_score'].median():.3f}")
    print(f"  Std: {df['composite_score'].std():.3f}")

    print(f"\nGap Distance Statistics:")
    print(f"  Mean: {df['gap_distance'].mean():.1f} nm")
    print(f"  Median: {df['gap_distance'].median():.1f} nm")
    print(f"  Max: {df['gap_distance'].max():.1f} nm")

    # Filter and show top connections
    if args.status:
        df = df[df["status"] == args.status]
        print(f"\nTop {args.top} {args.status} connections:")
    else:
        print(f"\nTop {args.top} connections by composite score:")

    top = df.nlargest(args.top, "composite_score")
    cols = ["candidate_id", "fragment_a", "fragment_b", "composite_score", "gap_distance", "status"]
    print(top[cols].to_string(index=False))

    # Load structures if available
    struct_path = output_dir / "structures.csv"
    if struct_path.exists():
        sdf = pd.read_csv(struct_path)
        print(f"\n{'='*60}")
        print(f"Structure Summary ({len(sdf)} structures)")
        print(f"{'='*60}")
        print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
