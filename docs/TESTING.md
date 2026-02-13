# Testing Documentation

## Overview

The connectomics pipeline uses **pytest** as its testing framework. Tests are organized by module and cover unit tests, integration tests, and end-to-end pipeline tests. All tests use synthetic data — no real connectomics volumes are required.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run a specific module
pytest tests/test_validation.py

# Run a specific test
pytest tests/test_candidates.py::TestProximityScore::test_zero_distance

# Run with coverage report
pytest tests/ --cov=connectomics_pipeline --cov-report=term-missing

# Run only fast unit tests (exclude slow integration)
pytest tests/ -m "not slow"
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures (volumes, fragments, configs)
├── test_io.py               # Volume reader tests (9 tests)
├── test_fragments.py        # Fragment extraction + store tests (16 tests)
├── test_graph.py            # Graph construction + spatial index tests (9 tests)
├── test_candidates.py       # Candidate generation + scoring tests (16 tests)
├── test_validation.py       # Validation rule tests (12 tests)
├── test_assembly.py         # Assembly + topology tests (8 tests)
└── test_pipeline.py         # End-to-end integration tests (2 tests)
```

**Total: 72 tests**

## Test Categories

### Unit Tests (modules tested in isolation)

| Module | File | Tests | What's Covered |
|--------|------|-------|----------------|
| I/O | `test_io.py` | 9 | NumpyReader shape/dtype/resolution, chunk reading, chunk iteration, HDF5 roundtrip |
| Fragments | `test_fragments.py` | 16 | Extraction (single/multi-label), min voxel filtering, boundary detection, connected components, store CRUD, spatial queries, metadata (endpoints, centroid), skeletonization |
| Graph | `test_graph.py` | 9 | Graph add/query, edge operations, neighbors, subgraph extraction, proximity/endpoint graph building, spatial index queries |
| Candidates | `test_candidates.py` | 16 | Proximity scoring (zero/max/beyond/mid/monotonic), alignment scoring, continuity scoring, size matching, composite score (equal/zero/weighted), full candidate generation |
| Validation | `test_validation.py` | 12 | MaxDistance (accept/reject), Curvature, DirectionReversal (aligned/misaligned), SizeDiscrepancy, BranchingLimit, CompositeScore (high/low), rule factory, full validation pipeline |
| Assembly | `test_assembly.py` | 8 | Topology (no cycles, cycle detection, branch points, branch order), confidence (single/weakest link/empty), basic assembly |

### Integration Tests (end-to-end)

| File | Tests | What's Covered |
|------|-------|----------------|
| `test_pipeline.py` | 2 | Full pipeline execution with synthetic data, export verification (config saved, CSV output) |

## Fixtures (conftest.py)

Shared fixtures provide consistent test data across all modules:

- **`resolution`** — Standard anisotropic resolution: `(30.0, 8.0, 8.0)` nm (z, y, x)
- **`synthetic_volume`** — 32x64x64 array with 3 distinct tubular objects (labels 1, 2, 3)
- **`numpy_reader`** — NumpyReader wrapping the synthetic volume
- **`sample_fragments`** — 3 pre-built Fragment objects (A, B, C) with skeletons, endpoints, bounding boxes
- **`fragment_store`** — FragmentStore populated with the 3 sample fragments
- **`default_config`** / **`fragment_config`** / **`graph_config`** / **`candidate_config`** / **`validation_config`** — Config fixtures with test-appropriate parameters

## Synthetic Test Data

Test data is generated programmatically (no external data files needed for unit tests):

- **In-memory volumes**: Created directly in fixtures using NumPy arrays
- **`scripts/generate_test_data.py`**: Generates larger synthetic HDF5 volumes with configurable tube count, gaps, and shape
- **`test_data/synthetic.h5`**: Pre-generated 64x128x128 volume with 5 tubes and 3 gaps (41 KB, .gitignored)

## Adding New Tests

1. Place test files in `tests/` with the `test_` prefix
2. Use existing fixtures from `conftest.py` where possible
3. Follow the existing `TestClassName::test_method` pattern
4. For new fixtures, add them to `conftest.py` if shared across modules
5. Use `tmp_path` (pytest built-in) for any tests that write to disk
