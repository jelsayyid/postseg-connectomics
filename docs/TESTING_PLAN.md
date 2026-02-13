# Testing Plan

## Goal

Validate the post-segmentation connectomics pipeline for correctness, robustness, and performance before use on real lab data. Testing follows a phased approach — each phase builds on the previous one.

## Phase 1: Unit Test Baseline (CURRENT — COMPLETE)

**Status:** Done (72/72 passing)

All individual modules have unit tests covering their core logic:
- I/O readers, fragment extraction, graph construction, candidate scoring, validation rules, assembly, topology detection

**Deliverable:** Green test suite, documented in Experiment Log entry #1.

## Phase 2: Coverage Analysis & Gap Filling

**Status:** Next up

**Tasks:**
1. Install `pytest-cov` and run coverage analysis
2. Identify modules/branches with < 80% coverage
3. Write targeted tests for uncovered paths, focusing on:
   - Error handling paths (malformed input, empty volumes, missing config fields)
   - Edge cases in scoring functions (boundary values, NaN/inf handling)
   - Export formats beyond CSV (GraphML, SWC, JSON, Neuroglancer)
   - Visualization module (currently untested)
4. Target: **>85% line coverage** across all modules

**Why this matters:** Coverage gaps often hide bugs. The current tests verify happy paths; we need to verify the pipeline degrades gracefully on unexpected input.

## Phase 3: Edge Case & Robustness Testing

**Tasks:**
1. **Empty/degenerate volumes**: Volume with all zeros, single-voxel fragments, one giant label
2. **Boundary conditions**: Fragments touching volume edges, fragments spanning full volume
3. **Extreme parameters**: Zero thresholds, very large distances, all rules enabled at once
4. **Label pathologies**: Non-contiguous same-label regions, extremely high label counts (1000+)
5. **Skeleton edge cases**: Linear (1D) fragments, single-point fragments, loops

**Why this matters:** Real connectomics data is messy. Automated segmentation produces all of these cases. The pipeline must handle them without crashing.

## Phase 4: Parameter Sensitivity Analysis

**Tasks:**
1. Create a parameter sweep script that varies key thresholds:
   - `accept_threshold` (0.3 to 0.95 in steps)
   - `reject_threshold` (0.1 to 0.5 in steps)
   - Scoring weights (proximity, alignment, continuity, size)
   - `max_distance_nm` in graph construction
2. Run pipeline on a fixed synthetic dataset for each parameter set
3. Record accept/reject/ambiguous counts and topology warnings
4. Visualize sensitivity curves (threshold vs acceptance rate)

**Why this matters:** The lab needs to understand how parameter choices affect reconstruction aggressiveness. This builds confidence in the conservative validation approach.

## Phase 5: Scalability & Performance Testing

**Tasks:**
1. Generate synthetic volumes at increasing sizes: 64^3, 128^3, 256^3, 512^3
2. Measure per-phase execution time and peak memory usage
3. Identify bottlenecks (likely: spatial indexing, graph construction, or skeleton extraction)
4. Establish baseline performance numbers for the lab's expected data scale

**Why this matters:** Real connectomics volumes are large (multi-GB). We need to know where the pipeline will slow down.

## Phase 6: Integration Testing with Realistic Data

**Tasks:**
1. Use `scripts/generate_test_data.py` to create volumes with controlled properties:
   - Varying tube density (sparse vs dense)
   - Different gap sizes (simulating segmentation quality)
   - Overlapping structures
2. Run full pipeline end-to-end with all validation rules enabled
3. Manually inspect outputs (exported graphs, CSVs) for correctness
4. Verify exported files can be loaded by downstream tools

**Why this matters:** This is the closest we can get to real data validation without actual lab volumes.

## Phase 7: Regression Test Suite

**Tasks:**
1. Identify key pipeline behaviors that must remain stable across code changes
2. Create golden-output tests: run pipeline on fixed synthetic data, save outputs, compare on future runs
3. Add CI configuration (GitHub Actions) to run the test suite on every push
4. Integrate `black` and `mypy` checks into CI

**Why this matters:** As the pipeline evolves, we need automated guards against regressions.

## Summary Timeline

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Unit test baseline | Complete |
| 2 | Coverage analysis & gap filling | **Next** |
| 3 | Edge cases & robustness | Planned |
| 4 | Parameter sensitivity | Planned |
| 5 | Scalability & performance | Planned |
| 6 | Realistic integration testing | Planned |
| 7 | Regression suite & CI | Planned |
