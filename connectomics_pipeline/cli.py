"""Command-line interface for the connectomics pipeline."""

from __future__ import annotations

import argparse
import sys

from connectomics_pipeline.pipeline import Pipeline
from connectomics_pipeline.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the connectomics pipeline."""
    parser = argparse.ArgumentParser(
        prog="connectomics-pipeline",
        description="Post-segmentation pipeline for large-scale connectomics data",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args(argv)

    config = load_config(args.config)

    if args.output_dir:
        config.export.output_dir = args.output_dir

    if args.verbose:
        config.logging.level = "DEBUG"

    pipeline = Pipeline(config)
    structures = pipeline.run()

    print(f"\nPipeline complete. {len(structures)} structures assembled.")
    print(f"Results saved to: {config.export.output_dir}")


if __name__ == "__main__":
    main()
