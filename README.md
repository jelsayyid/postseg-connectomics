# Connectomics Post-Segmentation Pipeline

**CPSC 4900 Senior Project**

A modular, conservative post-segmentation systems pipeline for large-scale connectomics data. Takes the output of automated segmentation workflows (e.g., PyTorch Connectomics) and produces graph-based connectivity representations suitable for downstream analysis, proofreading, and visualization.

## Pipeline Overview

```
Segmentation Volume → Fragment Extraction → Graph Construction → Candidate Generation
    → Conservative Validation → Assembly → Export (graphs, SWC, metadata)
```

**Design philosophy:** Conservative correctness over aggressive merging. Uncertainty is explicitly preserved, not discarded.

## Installation

```bash
# Core installation
pip install -e .

# With all optional dependencies
pip install -e ".[all]"
```

## Quick Start

```bash
# Run the pipeline with a config file
python scripts/run_pipeline.py --config configs/default.yaml

# Or use the CLI
connectomics-pipeline --config configs/default.yaml
```

## Project Structure

```
connectomics_pipeline/
├── io/              # Input/output: volume readers for HDF5, Zarr, precomputed
├── fragments/       # Fragment extraction, skeletonization, meshing
├── graph/           # Fragment graph construction
├── candidates/      # Candidate connection generation
├── validation/      # Conservative validation rules
├── assembly/        # Structure assembly from validated connections
├── export/          # Output in GraphML, SWC, CSV, Neuroglancer formats
├── visualization/   # Diagnostics and visualization tools
└── utils/           # Configuration, types, spatial math
```

## Configuration

All pipeline parameters are controlled via YAML config files. See `configs/default.yaml` for the full set of options.

## Documentation

- [Architecture Document](docs/ARCHITECTURE.md) — Full system design, data flow, and module interfaces

## References

- Lin et al., 2021 — PyTorch Connectomics (arXiv:2112.05754)
- Dorkenwald et al., 2025 — CAVE (Nature Methods)
- MICrONS Consortium, 2025 — Functional Connectomics (Nature)
- Plaza et al., 2018 — Analyzing Image Segmentation for Connectomics (BMC Medical Imaging)
