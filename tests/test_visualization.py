"""Tests for the visualization module (neuroglancer annotations + 3D plot)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    BoundingBox,
    CandidateConnection,
    ConnectionStatus,
    Fragment,
)
from connectomics_pipeline.visualization.neuroglancer_annotations import (
    _generate_colors,
    _hsv_to_rgb,
    generate_structure_annotations,
)
from connectomics_pipeline.visualization.plot_connections import (
    STATUS_COLORS,
    plot_connections_3d,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frag(fid: int, centroid=(50.0, 50.0, 50.0)) -> Fragment:
    c = np.array(centroid, dtype=float)
    return Fragment(
        fragment_id=fid,
        label_id=1,
        voxel_count=100,
        bounding_box=BoundingBox(min_corner=np.zeros(3), max_corner=np.ones(3) * 100),
        centroid=c,
        endpoints=[c.copy()],
    )


def _structure(sid: int, frag_ids: list[int]) -> AssembledStructure:
    return AssembledStructure(
        structure_id=sid,
        fragment_ids=frag_ids,
        accepted_connections=[],
        ambiguous_connections=[],
        confidence=0.9,
        total_path_length=100.0,
        num_branches=0,
        has_ambiguous_regions=False,
        topology_warnings=[],
    )


def _candidate(cid: int, status: ConnectionStatus) -> CandidateConnection:
    return CandidateConnection(
        candidate_id=cid,
        fragment_a=0,
        fragment_b=1,
        endpoint_a=np.zeros(3),
        endpoint_b=np.ones(3) * 10.0,
        composite_score=0.5,
        status=status,
    )


# ---------------------------------------------------------------------------
# neuroglancer_annotations
# ---------------------------------------------------------------------------


class TestGenerateColors:
    def test_zero_returns_white(self):
        colors = _generate_colors(0)
        assert colors == ["#ffffff"]

    def test_single_color(self):
        colors = _generate_colors(1)
        assert len(colors) == 1
        assert colors[0].startswith("#")
        assert len(colors[0]) == 7

    def test_multiple_colors(self):
        colors = _generate_colors(5)
        assert len(colors) == 5
        for c in colors:
            assert c.startswith("#")
            assert len(c) == 7


class TestHsvToRgb:
    def test_red_hue(self):
        r, g, b = _hsv_to_rgb(0.0, 1.0, 1.0)
        assert abs(r - 1.0) < 1e-6

    def test_output_in_unit_range(self):
        r, g, b = _hsv_to_rgb(0.33, 0.8, 0.9)
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0


class TestGenerateStructureAnnotations:
    def test_creates_json_file(self, tmp_path):
        store = FragmentStore()
        store.add(_frag(0))
        structure = _structure(0, [0])

        generate_structure_annotations([structure], store, tmp_path)

        out = tmp_path / "structures.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["type"] == "annotation"
        assert len(data["annotations"]) == 1

    def test_missing_fragment_skipped(self, tmp_path):
        """Fragment IDs in structure but not in store are silently skipped."""
        store = FragmentStore()  # empty — frag 99 not present
        structure = _structure(0, [99])

        generate_structure_annotations([structure], store, tmp_path)

        data = json.loads((tmp_path / "structures.json").read_text())
        assert data["annotations"] == []

    def test_empty_structures_list(self, tmp_path):
        store = FragmentStore()
        generate_structure_annotations([], store, tmp_path)
        assert (tmp_path / "structures.json").exists()

    def test_multiple_structures(self, tmp_path):
        store = FragmentStore()
        for i in range(3):
            store.add(_frag(i, (float(i * 100), 0.0, 0.0)))
        structures = [_structure(0, [0, 1]), _structure(1, [2])]

        generate_structure_annotations(structures, store, tmp_path)

        data = json.loads((tmp_path / "structures.json").read_text())
        assert len(data["annotations"]) == 3

    def test_annotation_contents(self, tmp_path):
        store = FragmentStore()
        store.add(_frag(0, (10.0, 20.0, 30.0)))
        generate_structure_annotations([_structure(0, [0])], store, tmp_path)
        data = json.loads((tmp_path / "structures.json").read_text())
        ann = data["annotations"][0]
        assert ann["type"] == "point"
        assert len(ann["point"]) == 3


# ---------------------------------------------------------------------------
# plot_connections
# ---------------------------------------------------------------------------


class TestStatusColors:
    def test_all_connection_statuses_present(self):
        assert ConnectionStatus.ACCEPTED in STATUS_COLORS
        assert ConnectionStatus.REJECTED in STATUS_COLORS
        assert ConnectionStatus.AMBIGUOUS in STATUS_COLORS
        assert ConnectionStatus.PROPOSED in STATUS_COLORS


class TestPlotConnections3d:
    def test_skips_gracefully_without_matplotlib(self):
        """Returns silently when _HAS_MPL is False."""
        store = FragmentStore()
        with patch("connectomics_pipeline.visualization.plot_connections._HAS_MPL", False):
            plot_connections_3d([], store)  # must not raise

    def test_saves_figure_to_file(self, tmp_path):
        store = FragmentStore()
        store.add(_frag(0, (0.0, 0.0, 0.0)))
        store.add(_frag(1, (10.0, 0.0, 0.0)))
        candidates = [
            _candidate(0, ConnectionStatus.ACCEPTED),
            _candidate(1, ConnectionStatus.REJECTED),
            _candidate(2, ConnectionStatus.AMBIGUOUS),
        ]
        out = tmp_path / "plot.png"

        with patch("connectomics_pipeline.visualization.plot_connections._HAS_MPL", True):
            import matplotlib

            matplotlib.use("Agg")
            plot_connections_3d(candidates, store, output_path=out)

        assert out.exists()

    def test_empty_candidates_with_output(self, tmp_path):
        store = FragmentStore()
        out = tmp_path / "empty.png"

        with patch("connectomics_pipeline.visualization.plot_connections._HAS_MPL", True):
            import matplotlib

            matplotlib.use("Agg")
            plot_connections_3d([], store, output_path=out)

        assert out.exists()

    def test_show_called_without_output_path(self):
        """plt.show() is called when no output_path is provided."""
        store = FragmentStore()

        with patch("connectomics_pipeline.visualization.plot_connections._HAS_MPL", True):
            with patch("connectomics_pipeline.visualization.plot_connections.plt") as mock_plt:
                mock_fig = MagicMock()
                mock_plt.figure.return_value = mock_fig
                mock_fig.add_subplot.return_value = MagicMock()
                plot_connections_3d([], store)
                mock_plt.show.assert_called_once()

    def test_proposed_status_not_plotted(self, tmp_path):
        """PROPOSED status candidates are absent from the plotted statuses loop."""
        store = FragmentStore()
        store.add(_frag(0))
        store.add(_frag(1))
        out = tmp_path / "proposed.png"

        with patch("connectomics_pipeline.visualization.plot_connections._HAS_MPL", True):
            import matplotlib

            matplotlib.use("Agg")
            # PROPOSED is not in the loop's status list — just verify no error
            plot_connections_3d([_candidate(0, ConnectionStatus.PROPOSED)], store, output_path=out)

        assert out.exists()
