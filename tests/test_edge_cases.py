"""Phase 3: Edge case and robustness tests.

Tests pipeline behavior under degenerate, extreme, and boundary-condition inputs.
These scenarios mimic pathologies found in real automated segmentation data.
"""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.candidates.generator import CandidateGenerator, _find_best_endpoint_pair
from connectomics_pipeline.candidates.proximity import compute_proximity_score
from connectomics_pipeline.candidates.scoring import compute_composite_score, compute_size_score
from connectomics_pipeline.candidates.alignment import compute_alignment_score
from connectomics_pipeline.candidates.continuity import compute_continuity_score
from connectomics_pipeline.export.swc_export import export_swc
from connectomics_pipeline.fragments.extraction import FragmentExtractor
from connectomics_pipeline.fragments.mesh import MeshExtractor
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.io.numpy_reader import NumpyReader
from connectomics_pipeline.pipeline import Pipeline
from connectomics_pipeline.utils.config import (
    AssemblyConfig,
    CandidateConfig,
    ExportConfig,
    FragmentConfig,
    GraphConfig,
    InputConfig,
    LoggingConfig,
    PipelineConfig,
    RuleConfig,
    ValidationConfig,
)
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
    Skeleton,
    ValidationReport,
    ValidationResult,
)
from connectomics_pipeline.validation.report import build_report
from connectomics_pipeline.validation.rules import (
    BranchingLimitRule,
    CompositeScoreRule,
    CurvatureRule,
    DirectionReversalRule,
    MaxDistanceRule,
    SizeDiscrepancyRule,
)
from connectomics_pipeline.assembly.assembler import Assembler


# ===========================================================================
# Helpers
# ===========================================================================


def _quick_pipeline_config(tmp_path, **overrides):
    """Build a minimal pipeline config for edge-case tests."""
    defaults = dict(
        input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(64, 64, 64), chunk_overlap=(0, 0, 0),
        ),
        fragments=FragmentConfig(min_voxel_count=5, extract_skeletons=False, extract_meshes=False),
        graph=GraphConfig(max_distance_nm=5000.0),
        candidates=CandidateConfig(
            max_endpoint_distance_nm=5000.0, min_alignment_score=0.0, min_composite_score=0.0,
        ),
        validation=ValidationConfig(
            accept_threshold=0.3, reject_threshold=0.1,
            rules=[RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 5000.0})],
        ),
        assembly=AssemblyConfig(min_structure_fragments=2),
        export=ExportConfig(formats=["csv"], output_dir=str(tmp_path / "out")),
        logging=LoggingConfig(level="WARNING", file="", console=False),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_frag(fid, label, voxels, centroid, endpoints=None, skeleton=None):
    c = np.array(centroid, dtype=float)
    return Fragment(
        fragment_id=fid, label_id=label, voxel_count=voxels,
        bounding_box=BoundingBox(c - 50, c + 50),
        centroid=c,
        endpoints=endpoints or [c],
        skeleton=skeleton,
    )


# ===========================================================================
# 1. EMPTY / DEGENERATE VOLUMES
# ===========================================================================


class TestEmptyVolume:
    """Pipeline must handle a volume with no labeled voxels."""

    def test_all_zeros(self, tmp_path):
        vol = np.zeros((32, 32, 32), dtype=np.uint32)
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(32, 32, 32), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert structures == []
        assert len(pipeline.store) == 0
        assert len(pipeline.candidates) == 0

    def test_single_voxel_below_min(self, tmp_path):
        """One labeled voxel, below min_voxel_count threshold."""
        vol = np.zeros((16, 16, 16), dtype=np.uint32)
        vol[8, 8, 8] = 1
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(16, 16, 16), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert len(pipeline.store) == 0

    def test_single_fragment_no_structures(self, tmp_path):
        """Only one fragment — can't form a structure (min_structure_fragments=2)."""
        vol = np.zeros((16, 16, 16), dtype=np.uint32)
        vol[3:13, 3:13, 3:13] = 1
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(16, 16, 16), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert structures == []
        assert len(pipeline.store) == 1


# ===========================================================================
# 2. EXTREME FRAGMENT COUNTS
# ===========================================================================


class TestManyLabels:
    """Volume with many distinct small labels."""

    def test_high_label_count(self, tmp_path):
        vol = np.zeros((32, 32, 32), dtype=np.uint32)
        # Place 50 small cubes (2x2x2 = 8 voxels each)
        label = 1
        for z in range(0, 32, 4):
            for y in range(0, 32, 4):
                for x in range(0, 32, 4):
                    vol[z:z+2, y:y+2, x:x+2] = label
                    label += 1
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(8.0, 8.0, 8.0),
            chunk_size=(32, 32, 32), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(8.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        # Should extract many fragments (exact count depends on min_voxel)
        assert len(pipeline.store) > 10

    def test_one_giant_label(self, tmp_path):
        """Single label fills the entire volume."""
        vol = np.ones((16, 16, 16), dtype=np.uint32)
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(16, 16, 16), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert len(pipeline.store) == 1
        assert structures == []  # Only 1 fragment, no structures


# ===========================================================================
# 3. FRAGMENT EXTRACTION EDGE CASES
# ===========================================================================


class TestFragmentExtractionEdgeCases:
    def test_non_contiguous_same_label(self):
        """Same label, two disconnected blobs → two fragments."""
        vol = np.zeros((20, 20, 20), dtype=np.uint32)
        vol[2:6, 2:6, 2:6] = 1
        vol[14:18, 14:18, 14:18] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=5), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 2
        assert all(f.label_id == 1 for f in frags)

    def test_fragment_touching_all_boundaries(self):
        """Fragment spans entire chunk — is_boundary should be True."""
        vol = np.zeros((8, 8, 8), dtype=np.uint32)
        vol[:, :, :] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=5), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 1
        assert frags[0].is_boundary is True

    def test_fragment_in_corner(self):
        """Fragment in corner touches 3 boundaries."""
        vol = np.zeros((16, 16, 16), dtype=np.uint32)
        vol[0:3, 0:3, 0:3] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=5), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 1
        assert frags[0].is_boundary is True

    def test_fragment_not_touching_boundary(self):
        """Fragment fully interior."""
        vol = np.zeros((20, 20, 20), dtype=np.uint32)
        vol[5:15, 5:15, 5:15] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=5), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 1
        assert frags[0].is_boundary is False

    def test_min_voxel_exactly_at_threshold(self):
        """Fragment with exactly min_voxel_count voxels should be included."""
        vol = np.zeros((10, 10, 10), dtype=np.uint32)
        # Place exactly 10 voxels
        vol[0, 0, 0:10] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=10), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 1

    def test_min_voxel_one_below_threshold(self):
        """Fragment with min_voxel_count-1 voxels should be excluded."""
        vol = np.zeros((10, 10, 10), dtype=np.uint32)
        vol[0, 0, 0:9] = 1
        extractor = FragmentExtractor(FragmentConfig(min_voxel_count=10), resolution=(8.0, 8.0, 8.0))
        frags = extractor.extract_from_chunk(vol, (0, 0, 0))
        assert len(frags) == 0


# ===========================================================================
# 4. SCORING EDGE CASES
# ===========================================================================


class TestScoringEdgeCases:
    def test_proximity_negative_distance(self):
        """Negative distance should be treated as zero (perfect score)."""
        score = compute_proximity_score(-5.0, 1000.0)
        assert score >= 0.0

    def test_proximity_zero_max_distance(self):
        """Zero max distance — should not crash."""
        score = compute_proximity_score(10.0, 0.0)
        assert score == 0.0 or np.isfinite(score)

    def test_size_score_zero_radius(self):
        """One radius = 0 should not crash."""
        score = compute_size_score(0.0, 5.0)
        assert np.isfinite(score)

    def test_size_score_both_zero(self):
        """Both radii = 0 should return reasonable score."""
        score = compute_size_score(0.0, 0.0)
        assert np.isfinite(score)

    def test_composite_zero_weights(self):
        """All zero weights should not crash."""
        w = {"proximity": 0.0, "alignment": 0.0, "continuity": 0.0, "size": 0.0}
        score = compute_composite_score(0.5, 0.5, 0.5, 0.5, w)
        assert np.isfinite(score)

    def test_alignment_identical_fragments(self):
        """Two fragments at the same position."""
        frag = _make_frag(0, 1, 100, [50, 50, 50], endpoints=[np.array([50, 50, 50.0])])
        score = compute_alignment_score(frag, frag, frag.centroid, frag.centroid)
        assert np.isfinite(score)

    def test_continuity_coincident_endpoints(self):
        """Endpoints at the same position."""
        frag_a = _make_frag(0, 1, 100, [50, 50, 50])
        frag_b = _make_frag(1, 1, 100, [50, 50, 60])
        ep = np.array([50, 50, 55.0])
        score = compute_continuity_score(frag_a, frag_b, ep, ep)
        assert np.isfinite(score)


# ===========================================================================
# 5. VALIDATION RULE EDGE CASES
# ===========================================================================


def _make_candidate(cid=0, ep_a=(0, 0, 0), ep_b=(0, 0, 100), score=0.5, align=0.5):
    return CandidateConnection(
        candidate_id=cid, fragment_a=0, fragment_b=1,
        endpoint_a=np.array(ep_a, dtype=float),
        endpoint_b=np.array(ep_b, dtype=float),
        composite_score=score, alignment_score=align,
    )


class TestValidationRuleEdgeCases:
    def test_max_distance_zero_gap(self):
        """Endpoints at same position."""
        rule = MaxDistanceRule(max_distance_nm=100.0)
        c = _make_candidate(ep_a=(0, 0, 0), ep_b=(0, 0, 0))
        store = FragmentStore()
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.ACCEPTED
        assert result.confidence == 1.0

    def test_max_distance_exactly_at_threshold(self):
        """Gap distance exactly equals max_distance_nm."""
        rule = MaxDistanceRule(max_distance_nm=100.0)
        c = _make_candidate(ep_a=(0, 0, 0), ep_b=(0, 0, 100))
        store = FragmentStore()
        result = rule.evaluate(c, store)
        # At exactly the limit: confidence = 0.0, still accepted
        assert result.decision == ConnectionStatus.ACCEPTED
        assert abs(result.confidence) < 0.01

    def test_curvature_missing_fragment(self):
        """Fragment not in store → AMBIGUOUS."""
        rule = CurvatureRule()
        c = _make_candidate()
        store = FragmentStore()  # Empty store
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.AMBIGUOUS

    def test_size_discrepancy_missing_fragment(self):
        rule = SizeDiscrepancyRule()
        c = _make_candidate()
        store = FragmentStore()
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.AMBIGUOUS

    def test_size_discrepancy_equal_fragments(self):
        """Equal-sized fragments → high confidence accept."""
        rule = SizeDiscrepancyRule(max_radius_ratio=3.0)
        c = _make_candidate()
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [0, 0, 0]))
        store.add(_make_frag(1, 1, 100, [0, 0, 100]))
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.ACCEPTED
        assert result.confidence > 0.9

    def test_size_discrepancy_huge_mismatch(self):
        """1000x size difference → reject."""
        rule = SizeDiscrepancyRule(max_radius_ratio=3.0)
        c = _make_candidate()
        store = FragmentStore()
        store.add(_make_frag(0, 1, 10, [0, 0, 0]))
        store.add(_make_frag(1, 1, 10000, [0, 0, 100]))
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.REJECTED

    def test_branching_with_high_degree(self):
        """Node with many neighbors should be rejected."""
        rule = BranchingLimitRule(max_branches=3)
        graph = FragmentGraph()
        for i in range(6):
            graph.add_fragment(_make_frag(i, 1, 100, [0, 0, i * 100]))
        # Connect fragment 0 to 1,2,3,4,5
        for i in range(1, 6):
            graph.add_edge(0, i, distance=100.0)
        c = _make_candidate(ep_a=(0, 0, 0), ep_b=(0, 0, 100))
        store = FragmentStore()
        result = rule.evaluate(c, store, graph)
        assert result.decision == ConnectionStatus.REJECTED

    def test_composite_score_at_threshold(self):
        """Score exactly at reject threshold."""
        rule = CompositeScoreRule(reject_threshold=0.5)
        c = _make_candidate(score=0.5)
        store = FragmentStore()
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.ACCEPTED

    def test_composite_score_just_below(self):
        rule = CompositeScoreRule(reject_threshold=0.5)
        c = _make_candidate(score=0.499)
        store = FragmentStore()
        result = rule.evaluate(c, store)
        assert result.decision == ConnectionStatus.REJECTED


# ===========================================================================
# 6. ASSEMBLY EDGE CASES
# ===========================================================================


class TestAssemblyEdgeCases:
    def test_no_accepted_connections(self):
        """No accepted connections → no structures."""
        assembler = Assembler(AssemblyConfig(min_structure_fragments=2))
        graph = FragmentGraph()
        report = ValidationReport(results={}, accepted=[], rejected=[0], ambiguous=[])
        candidates = [_make_candidate(0)]
        structures = assembler.assemble(graph, candidates, report, FragmentStore())
        assert structures == []

    def test_all_ambiguous(self):
        """All connections ambiguous → no structures."""
        assembler = Assembler(AssemblyConfig(min_structure_fragments=2))
        graph = FragmentGraph()
        report = ValidationReport(results={}, accepted=[], rejected=[], ambiguous=[0, 1])
        candidates = [_make_candidate(0), _make_candidate(1)]
        structures = assembler.assemble(graph, candidates, report, FragmentStore())
        assert structures == []

    def test_ambiguous_flagged_on_structure(self):
        """Structure with nearby ambiguous connection should flag it."""
        assembler = Assembler(AssemblyConfig(min_structure_fragments=2, flag_ambiguous=True))
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [0, 0, 0]))
        store.add(_make_frag(1, 1, 100, [0, 0, 200]))
        store.add(_make_frag(2, 1, 100, [0, 0, 400]))

        graph = FragmentGraph()
        graph.add_fragment(store.get(0))
        graph.add_fragment(store.get(1))

        # Connection 0 is accepted (links 0 and 1)
        c_accepted = CandidateConnection(
            candidate_id=0, fragment_a=0, fragment_b=1,
            endpoint_a=np.array([0, 0, 100.0]), endpoint_b=np.array([0, 0, 100.0]),
            composite_score=0.9, status=ConnectionStatus.ACCEPTED,
        )
        # Connection 1 is ambiguous (involves fragment 1)
        c_ambiguous = CandidateConnection(
            candidate_id=1, fragment_a=1, fragment_b=2,
            endpoint_a=np.array([0, 0, 300.0]), endpoint_b=np.array([0, 0, 300.0]),
            composite_score=0.5, status=ConnectionStatus.AMBIGUOUS,
        )
        report = ValidationReport(
            results={0: [], 1: []},
            accepted=[0], rejected=[], ambiguous=[1],
        )
        structures = assembler.assemble(graph, [c_accepted, c_ambiguous], report, store)
        assert len(structures) == 1
        assert structures[0].has_ambiguous_regions is True
        assert 1 in structures[0].ambiguous_connections


# ===========================================================================
# 7. GRAPH BUILDER EDGE CASES
# ===========================================================================


class TestGraphBuilderEdgeCases:
    def test_empty_store(self):
        builder = GraphBuilder(GraphConfig(max_distance_nm=1000.0))
        store = FragmentStore()
        graph = builder.build(store)
        assert graph.num_nodes == 0
        assert graph.num_edges == 0

    def test_single_fragment(self):
        builder = GraphBuilder(GraphConfig(max_distance_nm=1000.0))
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [50, 50, 50]))
        graph = builder.build(store)
        assert graph.num_nodes == 1
        assert graph.num_edges == 0

    def test_fragments_beyond_max_distance(self):
        """Two fragments far apart should have no edge."""
        builder = GraphBuilder(GraphConfig(max_distance_nm=100.0))
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [0, 0, 0]))
        store.add(_make_frag(1, 1, 100, [0, 0, 50000]))
        graph = builder.build(store)
        assert graph.num_edges == 0

    def test_unknown_method_raises(self):
        builder = GraphBuilder(GraphConfig(construction_method="nonexistent"))
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [0, 0, 0]))
        with pytest.raises(ValueError, match="Unknown graph construction method"):
            builder.build(store)


# ===========================================================================
# 8. CANDIDATE GENERATOR EDGE CASES
# ===========================================================================


class TestCandidateGeneratorEdgeCases:
    def test_no_edges_no_candidates(self):
        store = FragmentStore()
        store.add(_make_frag(0, 1, 100, [0, 0, 0]))
        gen = CandidateGenerator(CandidateConfig(), store)
        graph = FragmentGraph()
        graph.add_fragment(store.get(0))
        candidates = gen.generate(graph)
        assert candidates == []

    def test_find_best_endpoint_pair_no_endpoints(self):
        """Fragments with no endpoints fall back to centroids."""
        frag_a = _make_frag(0, 1, 100, [0, 0, 0], endpoints=[])
        frag_b = _make_frag(1, 1, 100, [0, 0, 100], endpoints=[])
        ep_a, ep_b = _find_best_endpoint_pair(frag_a, frag_b)
        np.testing.assert_array_equal(ep_a, frag_a.centroid)
        np.testing.assert_array_equal(ep_b, frag_b.centroid)


# ===========================================================================
# 9. VALIDATION REPORT EDGE CASES
# ===========================================================================


class TestValidationReportEdgeCases:
    def test_report_summary_no_candidates(self):
        report = ValidationReport()
        s = report.summary()
        assert s["total"] == 0
        assert s["accept_rate"] == 0.0

    def test_all_hard_rejected(self):
        """Every candidate has at least one hard REJECT."""
        candidates = [_make_candidate(i) for i in range(5)]
        results = {
            i: [ValidationResult("r", ConnectionStatus.REJECTED, 0.1, "fail")]
            for i in range(5)
        }
        report = build_report(results, candidates)
        assert len(report.rejected) == 5
        assert len(report.accepted) == 0


# ===========================================================================
# 10. TYPES EDGE CASES
# ===========================================================================


class TestTypesEdgeCases:
    def test_bounding_box_zero_volume(self):
        """Degenerate bounding box with zero volume."""
        bb = BoundingBox(np.array([5, 5, 5.0]), np.array([5, 5, 5.0]))
        assert bb.volume == 0.0
        assert np.array_equal(bb.center, np.array([5, 5, 5.0]))

    def test_bounding_box_overlaps_touching(self):
        """Boxes sharing an edge should overlap."""
        a = BoundingBox(np.array([0, 0, 0.0]), np.array([10, 10, 10.0]))
        b = BoundingBox(np.array([10, 0, 0.0]), np.array([20, 10, 10.0]))
        assert a.overlaps(b) is True

    def test_bounding_box_no_overlap(self):
        a = BoundingBox(np.array([0, 0, 0.0]), np.array([10, 10, 10.0]))
        b = BoundingBox(np.array([20, 20, 20.0]), np.array([30, 30, 30.0]))
        assert a.overlaps(b) is False

    def test_bounding_box_expand(self):
        bb = BoundingBox(np.array([10, 10, 10.0]), np.array([20, 20, 20.0]))
        expanded = bb.expand(5.0)
        np.testing.assert_array_equal(expanded.min_corner, np.array([5, 5, 5.0]))
        np.testing.assert_array_equal(expanded.max_corner, np.array([25, 25, 25.0]))

    def test_skeleton_path_length_no_edges(self):
        skel = Skeleton(
            nodes=np.array([[0, 0, 0], [10, 0, 0.0]]),
            edges=np.zeros((0, 2), dtype=int),
            radii=np.array([1.0, 1.0]),
        )
        assert skel.path_length() == 0.0

    def test_skeleton_path_length(self):
        skel = Skeleton(
            nodes=np.array([[0, 0, 0], [10, 0, 0.0]]),
            edges=np.array([[0, 1]]),
            radii=np.array([1.0, 1.0]),
        )
        assert abs(skel.path_length() - 10.0) < 1e-10

    def test_candidate_gap_distance(self):
        c = _make_candidate(ep_a=(0, 0, 0), ep_b=(3, 4, 0))
        assert abs(c.gap_distance - 5.0) < 1e-10

    def test_fragment_physical_volume(self):
        frag = _make_frag(0, 1, 100, [50, 50, 50])
        assert frag.physical_volume > 0


# ===========================================================================
# 11. FULL PIPELINE EDGE CASES
# ===========================================================================


class TestPipelineEdgeCases:
    def test_volume_with_only_background(self, tmp_path):
        vol = np.zeros((16, 16, 16), dtype=np.uint32)
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(16, 16, 16), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert structures == []

    def test_two_distant_fragments(self, tmp_path):
        """Two fragments too far apart — no connections."""
        vol = np.zeros((16, 16, 64), dtype=np.uint32)
        vol[6:10, 6:10, 0:4] = 1
        vol[6:10, 6:10, 60:64] = 2
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(30.0, 8.0, 8.0),
            chunk_size=(16, 16, 64), chunk_overlap=(0, 0, 0),
        ), graph=GraphConfig(max_distance_nm=100.0))
        reader = NumpyReader(vol, resolution=(30.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert structures == []
        assert len(pipeline.store) == 2

    def test_two_touching_fragments(self, tmp_path):
        """Two adjacent fragments — should connect."""
        vol = np.zeros((16, 16, 32), dtype=np.uint32)
        vol[4:12, 4:12, 2:14] = 1
        vol[4:12, 4:12, 16:28] = 2
        config = _quick_pipeline_config(tmp_path, input=InputConfig(
            format="numpy", resolution=(8.0, 8.0, 8.0),
            chunk_size=(16, 16, 32), chunk_overlap=(0, 0, 0),
        ))
        reader = NumpyReader(vol, resolution=(8.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        assert len(pipeline.store) == 2
        assert len(pipeline.candidates) >= 1

    def test_strict_validation_rejects_all(self, tmp_path):
        """Very strict thresholds should reject everything."""
        vol = np.zeros((16, 16, 32), dtype=np.uint32)
        vol[4:12, 4:12, 2:12] = 1
        vol[4:12, 4:12, 18:28] = 2
        config = _quick_pipeline_config(
            tmp_path,
            input=InputConfig(
                format="numpy", resolution=(8.0, 8.0, 8.0),
                chunk_size=(16, 16, 32), chunk_overlap=(0, 0, 0),
            ),
            validation=ValidationConfig(
                accept_threshold=0.99, reject_threshold=0.98,
                rules=[RuleConfig(name="MaxDistanceRule", params={"max_distance_nm": 10.0})],
            ),
        )
        reader = NumpyReader(vol, resolution=(8.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        # With max_distance 10nm and fragments ~48nm apart, everything rejected
        assert structures == []

    def test_no_validation_rules(self, tmp_path):
        """Pipeline with empty rules list — candidates go to ambiguous."""
        vol = np.zeros((16, 16, 32), dtype=np.uint32)
        vol[4:12, 4:12, 2:14] = 1
        vol[4:12, 4:12, 16:28] = 2
        config = _quick_pipeline_config(
            tmp_path,
            input=InputConfig(
                format="numpy", resolution=(8.0, 8.0, 8.0),
                chunk_size=(16, 16, 32), chunk_overlap=(0, 0, 0),
            ),
            validation=ValidationConfig(
                accept_threshold=0.5, reject_threshold=0.2,
                rules=[],  # No rules
            ),
        )
        reader = NumpyReader(vol, resolution=(8.0, 8.0, 8.0))
        pipeline = Pipeline(config)
        structures = pipeline.run(reader=reader)
        # With no rules, all candidates should be ambiguous
        if pipeline.report and pipeline.candidates:
            assert len(pipeline.report.ambiguous) == len(pipeline.candidates)
