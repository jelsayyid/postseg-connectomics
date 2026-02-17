"""Tests for chunk boundary stitching and union-find."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.fragments.stitching import ChunkStitcher, UnionFind
from connectomics_pipeline.utils.types import BoundingBox, Fragment, Skeleton

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fragment(frag_id, label_id, min_corner, max_corner, is_boundary=False, chunk_origin=None):
    mc = np.array(min_corner, dtype=float)
    xc = np.array(max_corner, dtype=float)
    return Fragment(
        fragment_id=frag_id,
        label_id=label_id,
        voxel_count=100,
        bounding_box=BoundingBox(min_corner=mc, max_corner=xc),
        centroid=(mc + xc) / 2,
        endpoints=[(mc + xc) / 2],
        is_boundary=is_boundary,
        chunk_origin=chunk_origin,
    )


# ---------------------------------------------------------------------------
# UnionFind
# ---------------------------------------------------------------------------


class TestUnionFind:
    def test_find_new_element(self):
        uf = UnionFind()
        assert uf.find(0) == 0

    def test_union_same_root(self):
        uf = UnionFind()
        uf.union(0, 0)
        assert uf.find(0) == 0

    def test_union_two_elements(self):
        uf = UnionFind()
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_union_chain(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_groups_single(self):
        uf = UnionFind()
        uf.find(5)
        groups = uf.groups()
        assert 5 in groups
        assert groups[5] == [5]

    def test_groups_merged(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 2
        # Each group has 2 members
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [2, 2]

    def test_groups_all_merged(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)
        groups = uf.groups()
        assert len(groups) == 1
        assert len(list(groups.values())[0]) == 4

    def test_rank_balancing(self):
        """Union by rank should keep tree shallow."""
        uf = UnionFind()
        for i in range(10):
            uf.union(0, i)
        # All should resolve to same root
        root = uf.find(0)
        for i in range(10):
            assert uf.find(i) == root


# ---------------------------------------------------------------------------
# ChunkStitcher
# ---------------------------------------------------------------------------


class TestChunkStitcher:
    def test_empty_fragments(self):
        stitcher = ChunkStitcher()
        result = stitcher.stitch([], [])
        assert result == []

    def test_no_boundary_fragments(self):
        """Non-boundary fragments are not merged."""
        stitcher = ChunkStitcher()
        frags = [
            make_fragment(
                0, 1, [0, 0, 0], [100, 100, 100], is_boundary=False, chunk_origin=(0, 0, 0)
            ),
            make_fragment(
                1, 1, [0, 0, 100], [100, 100, 200], is_boundary=False, chunk_origin=(0, 0, 128)
            ),
        ]
        result = stitcher.stitch(frags, [((0, 0, 0), (0, 0, 128))])
        assert len(result) == 2

    def test_matching_boundary_fragments(self):
        """Boundary fragments with same label and overlapping bbox get merged."""
        stitcher = ChunkStitcher(overlap=(8, 16, 16))
        frags = [
            make_fragment(
                0, 1, [0, 0, 90], [100, 100, 128], is_boundary=True, chunk_origin=(0, 0, 0)
            ),
            make_fragment(
                1, 1, [0, 0, 112], [100, 100, 200], is_boundary=True, chunk_origin=(0, 0, 128)
            ),
        ]
        result = stitcher.stitch(frags, [((0, 0, 0), (0, 0, 128))])
        # Should merge into 1
        assert len(result) == 1

    def test_different_labels_not_merged(self):
        """Boundary fragments with different labels should not merge."""
        stitcher = ChunkStitcher()
        frags = [
            make_fragment(
                0, 1, [0, 0, 90], [100, 100, 128], is_boundary=True, chunk_origin=(0, 0, 0)
            ),
            make_fragment(
                1, 2, [0, 0, 112], [100, 100, 200], is_boundary=True, chunk_origin=(0, 0, 128)
            ),
        ]
        result = stitcher.stitch(frags, [((0, 0, 0), (0, 0, 128))])
        assert len(result) == 2

    def test_non_overlapping_bbox_not_merged(self):
        """Same label but non-overlapping bboxes should not merge."""
        stitcher = ChunkStitcher()
        frags = [
            make_fragment(0, 1, [0, 0, 0], [50, 50, 50], is_boundary=True, chunk_origin=(0, 0, 0)),
            make_fragment(
                1, 1, [60, 60, 200], [100, 100, 300], is_boundary=True, chunk_origin=(0, 0, 128)
            ),
        ]
        result = stitcher.stitch(frags, [((0, 0, 0), (0, 0, 128))])
        assert len(result) == 2

    def test_merged_fragment_properties(self):
        """Verify merged fragment has correct combined properties."""
        stitcher = ChunkStitcher()
        frags = [
            make_fragment(
                0, 1, [0, 0, 90], [100, 100, 130], is_boundary=True, chunk_origin=(0, 0, 0)
            ),
            make_fragment(
                1, 1, [0, 0, 110], [100, 100, 200], is_boundary=True, chunk_origin=(0, 0, 128)
            ),
        ]
        # Give different voxel counts
        frags[0] = Fragment(
            fragment_id=0,
            label_id=1,
            voxel_count=200,
            bounding_box=BoundingBox(np.array([0, 0, 90.0]), np.array([100, 100, 130.0])),
            centroid=np.array([50, 50, 110.0]),
            endpoints=[np.array([50, 50, 90.0])],
            is_boundary=True,
            chunk_origin=(0, 0, 0),
        )
        frags[1] = Fragment(
            fragment_id=1,
            label_id=1,
            voxel_count=300,
            bounding_box=BoundingBox(np.array([0, 0, 110.0]), np.array([100, 100, 200.0])),
            centroid=np.array([50, 50, 155.0]),
            endpoints=[np.array([50, 50, 200.0])],
            is_boundary=True,
            chunk_origin=(0, 0, 128),
        )
        result = stitcher.stitch(frags, [((0, 0, 0), (0, 0, 128))])
        assert len(result) == 1
        merged = result[0]
        assert merged.voxel_count == 500
        assert merged.label_id == 1
        # Bbox should span both
        np.testing.assert_array_less(merged.bounding_box.min_corner, np.array([1, 1, 91]))
        np.testing.assert_array_less(np.array([99, 99, 199]), merged.bounding_box.max_corner)
        # Endpoints collected from both
        assert len(merged.endpoints) == 2
        # Not boundary after merge
        assert merged.is_boundary is False

    def test_single_fragment_passthrough(self):
        """Single fragment with no pairs passes through unchanged."""
        stitcher = ChunkStitcher()
        frags = [make_fragment(0, 1, [0, 0, 0], [100, 100, 100])]
        result = stitcher.stitch(frags, [])
        assert len(result) == 1
        assert result[0].fragment_id == 0


class TestUnionFindRankSwap:
    def test_rank_swap_line_35(self):
        """Line 35: union swaps rx/ry when rank[rx] < rank[ry].

        Build two trees with rank=1 each, merge them (rank now 2 for root),
        then union a fresh node (rank=0) with that root to force the swap.
        """
        uf = UnionFind()
        # Build rank-1 tree rooted at 0: union(0,1) → rank[0]=1
        uf.union(0, 1)
        # Build rank-1 tree rooted at 2: union(2,3) → rank[2]=1
        uf.union(2, 3)
        # Merge them: union(0,2) → rank[0]=2, rank[2]=1 → equal rank path (no swap)
        uf.union(0, 2)
        # Now root of all is 0 with rank=2.
        # Union fresh node 4 with the high-rank root:
        # find(4)=4 (rank 0), find(0)=0 (rank 2) → rank[rx=4] < rank[ry=0] → swap!
        uf.union(4, 0)
        # All five nodes should share the same root
        root = uf.find(0)
        for node in [0, 1, 2, 3, 4]:
            assert uf.find(node) == root
