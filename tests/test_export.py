"""Tests for export modules: GraphML, JSON, SWC, and Neuroglancer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from connectomics_pipeline.export.graph_export import (
    export_graphml,
    export_json,
    _convert_numpy,
    _json_default,
)
from connectomics_pipeline.export.swc_export import export_swc, SWC_AXON, SWC_DENDRITE
from connectomics_pipeline.export.neuroglancer_export import export_annotations, _status_color
from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
    Skeleton,
)
from connectomics_pipeline.fragments.store import FragmentStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_graph(sample_fragments, fragment_store):
    """FragmentGraph with nodes and edges for export testing."""
    graph = FragmentGraph()
    for f in sample_fragments:
        graph.add_fragment(f)
    graph.add_edge(0, 1, distance=50.0)
    return graph


@pytest.fixture
def assembled_structure():
    return AssembledStructure(
        structure_id=0,
        fragment_ids=[0, 1],
        accepted_connections=[0],
        confidence=0.9,
        total_path_length=300.0,
        num_branches=0,
    )


@pytest.fixture
def candidate_connections():
    return [
        CandidateConnection(
            candidate_id=0,
            fragment_a=0,
            fragment_b=1,
            endpoint_a=np.array([50.0, 50.0, 200.0]),
            endpoint_b=np.array([50.0, 50.0, 250.0]),
            composite_score=0.85,
            status=ConnectionStatus.ACCEPTED,
        ),
        CandidateConnection(
            candidate_id=1,
            fragment_a=0,
            fragment_b=2,
            endpoint_a=np.array([50.0, 50.0, 0.0]),
            endpoint_b=np.array([50.0, 200.0, 50.0]),
            composite_score=0.2,
            status=ConnectionStatus.REJECTED,
        ),
        CandidateConnection(
            candidate_id=2,
            fragment_a=1,
            fragment_b=2,
            endpoint_a=np.array([50.0, 50.0, 450.0]),
            endpoint_b=np.array([50.0, 400.0, 50.0]),
            composite_score=0.5,
            status=ConnectionStatus.AMBIGUOUS,
        ),
    ]


# ---------------------------------------------------------------------------
# GraphML export
# ---------------------------------------------------------------------------


class TestGraphMLExport:
    def test_export_creates_file(self, simple_graph, tmp_path):
        out = tmp_path / "graph.graphml"
        export_graphml(simple_graph, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_creates_parent_dirs(self, simple_graph, tmp_path):
        out = tmp_path / "sub" / "dir" / "graph.graphml"
        export_graphml(simple_graph, out)
        assert out.exists()

    def test_export_contains_nodes(self, simple_graph, tmp_path):
        out = tmp_path / "graph.graphml"
        export_graphml(simple_graph, out)
        content = out.read_text()
        # GraphML should have node elements
        assert "<node" in content

    def test_export_contains_edges(self, simple_graph, tmp_path):
        out = tmp_path / "graph.graphml"
        export_graphml(simple_graph, out)
        content = out.read_text()
        assert "<edge" in content


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


class TestJSONExport:
    def test_export_creates_file(self, simple_graph, tmp_path):
        out = tmp_path / "graph.json"
        export_json(simple_graph, out)
        assert out.exists()

    def test_export_is_valid_json(self, simple_graph, tmp_path):
        out = tmp_path / "graph.json"
        export_json(simple_graph, out)
        data = json.loads(out.read_text())
        assert "nodes" in data or "links" in data

    def test_export_creates_parent_dirs(self, simple_graph, tmp_path):
        out = tmp_path / "nested" / "graph.json"
        export_json(simple_graph, out)
        assert out.exists()


class TestJsonDefault:
    def test_ndarray(self):
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_numpy_int(self):
        result = _json_default(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float(self):
        result = _json_default(np.float64(3.14))
        assert abs(result - 3.14) < 1e-10
        assert isinstance(result, float)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _json_default(object())


class TestConvertNumpy:
    def test_dict_with_ndarray(self):
        d = {"a": np.array([1, 2]), "b": "hello"}
        _convert_numpy(d)
        assert d["a"] == [1, 2]
        assert d["b"] == "hello"

    def test_nested_dict(self):
        d = {"outer": {"inner": np.array([3, 4])}}
        _convert_numpy(d)
        assert d["outer"]["inner"] == [3, 4]

    def test_list_with_ndarray(self):
        lst = [np.array([1, 2]), "text", {"key": np.array([5])}]
        _convert_numpy(lst)
        assert lst[0] == [1, 2]
        assert lst[2]["key"] == [5]


# ---------------------------------------------------------------------------
# SWC export
# ---------------------------------------------------------------------------


class TestSWCExport:
    def test_export_creates_file(self, assembled_structure, sample_fragments, tmp_path):
        store = FragmentStore()
        store.add_many(sample_fragments)
        out = tmp_path / "structure.swc"
        export_swc(assembled_structure, store, out)
        assert out.exists()

    def test_export_header(self, assembled_structure, sample_fragments, tmp_path):
        store = FragmentStore()
        store.add_many(sample_fragments)
        out = tmp_path / "structure.swc"
        export_swc(assembled_structure, store, out)
        content = out.read_text()
        assert "# SWC export" in content
        assert "# Fragments: 2" in content
        assert "# Confidence: 0.900" in content

    def test_export_has_nodes(self, assembled_structure, sample_fragments, tmp_path):
        store = FragmentStore()
        store.add_many(sample_fragments)
        out = tmp_path / "structure.swc"
        export_swc(assembled_structure, store, out)
        content = out.read_text()
        # Non-comment lines are SWC data
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        assert len(data_lines) > 0
        # Each data line has 7 fields: id type x y z radius parent
        fields = data_lines[0].split()
        assert len(fields) == 7

    def test_export_custom_type(self, assembled_structure, sample_fragments, tmp_path):
        store = FragmentStore()
        store.add_many(sample_fragments)
        out = tmp_path / "structure.swc"
        export_swc(assembled_structure, store, out, structure_type=SWC_DENDRITE)
        content = out.read_text()
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        # Second field should be the type code
        assert data_lines[0].split()[1] == str(SWC_DENDRITE)

    def test_export_missing_fragment(self, tmp_path):
        """Structure references a fragment not in store."""
        store = FragmentStore()
        structure = AssembledStructure(
            structure_id=0,
            fragment_ids=[999],
            confidence=0.5,
            total_path_length=0.0,
        )
        out = tmp_path / "missing.swc"
        export_swc(structure, store, out)
        assert out.exists()
        # Should have only header lines
        content = out.read_text()
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        assert len(data_lines) == 0

    def test_export_fragment_without_skeleton(self, tmp_path):
        """Fragment exists but has no skeleton."""
        store = FragmentStore()
        frag = Fragment(
            fragment_id=0,
            label_id=1,
            voxel_count=100,
            bounding_box=BoundingBox(
                min_corner=np.array([0.0, 0.0, 0.0]),
                max_corner=np.array([100.0, 100.0, 100.0]),
            ),
            centroid=np.array([50.0, 50.0, 50.0]),
            skeleton=None,
        )
        store.add(frag)
        structure = AssembledStructure(
            structure_id=0,
            fragment_ids=[0],
            confidence=0.5,
            total_path_length=0.0,
        )
        out = tmp_path / "no_skel.swc"
        export_swc(structure, store, out)
        content = out.read_text()
        data_lines = [l for l in content.strip().split("\n") if not l.startswith("#")]
        assert len(data_lines) == 0


# ---------------------------------------------------------------------------
# Neuroglancer export
# ---------------------------------------------------------------------------


class TestNeuroglancerExport:
    def test_export_creates_file(self, candidate_connections, tmp_path):
        structures = []
        export_annotations(candidate_connections, structures, tmp_path / "ng")
        assert (tmp_path / "ng" / "connections.json").exists()

    def test_export_valid_json(self, candidate_connections, tmp_path):
        export_annotations(candidate_connections, [], tmp_path / "ng")
        data = json.loads((tmp_path / "ng" / "connections.json").read_text())
        assert data["type"] == "annotation"
        assert len(data["annotations"]) == 3

    def test_annotation_fields(self, candidate_connections, tmp_path):
        export_annotations(candidate_connections, [], tmp_path / "ng")
        data = json.loads((tmp_path / "ng" / "connections.json").read_text())
        ann = data["annotations"][0]
        assert ann["type"] == "line"
        assert "pointA" in ann
        assert "pointB" in ann
        assert "color" in ann
        assert ann["props"]["status"] == "accepted"

    def test_status_colors(self, candidate_connections, tmp_path):
        export_annotations(candidate_connections, [], tmp_path / "ng")
        data = json.loads((tmp_path / "ng" / "connections.json").read_text())
        colors = [a["color"] for a in data["annotations"]]
        assert colors[0] == "#00ff00"  # accepted
        assert colors[1] == "#ff0000"  # rejected
        assert colors[2] == "#ffff00"  # ambiguous

    def test_empty_candidates(self, tmp_path):
        export_annotations([], [], tmp_path / "ng")
        data = json.loads((tmp_path / "ng" / "connections.json").read_text())
        assert len(data["annotations"]) == 0


class TestStatusColor:
    def test_known_statuses(self):
        assert _status_color("accepted") == "#00ff00"
        assert _status_color("rejected") == "#ff0000"
        assert _status_color("ambiguous") == "#ffff00"
        assert _status_color("proposed") == "#808080"

    def test_unknown_status(self):
        assert _status_color("unknown") == "#ffffff"
