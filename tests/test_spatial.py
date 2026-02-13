"""Tests for spatial math utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from connectomics_pipeline.utils.spatial import (
    SpatialIndex,
    angle_between,
    estimate_curvature,
    estimate_tangent,
    euclidean_distance,
)

# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------


class TestEuclideanDistance:
    def test_same_point(self):
        assert euclidean_distance(np.array([0, 0, 0]), np.array([0, 0, 0])) == 0.0

    def test_unit_distance(self):
        d = euclidean_distance(np.array([0, 0, 0]), np.array([1, 0, 0]))
        assert abs(d - 1.0) < 1e-10

    def test_3d_distance(self):
        d = euclidean_distance(np.array([1, 2, 3]), np.array([4, 6, 3]))
        assert abs(d - 5.0) < 1e-10  # 3-4-5 triangle

    def test_accepts_lists(self):
        d = euclidean_distance([0, 0, 0], [3, 4, 0])
        assert abs(d - 5.0) < 1e-10


# ---------------------------------------------------------------------------
# estimate_tangent
# ---------------------------------------------------------------------------


class TestEstimateTangent:
    def test_straight_line_x(self):
        nodes = np.array([[0, 0, i] for i in range(10)], dtype=float)
        t = estimate_tangent(nodes, 5)
        # Should point along z-axis
        assert abs(abs(t[2]) - 1.0) < 1e-10

    def test_single_node(self):
        nodes = np.array([[0, 0, 0]], dtype=float)
        t = estimate_tangent(nodes, 0)
        # Fallback: [1, 0, 0]
        np.testing.assert_array_equal(t, [1.0, 0.0, 0.0])

    def test_two_nodes(self):
        nodes = np.array([[0, 0, 0], [0, 10, 0]], dtype=float)
        t = estimate_tangent(nodes, 0)
        assert abs(abs(t[1]) - 1.0) < 1e-10

    def test_unit_vector(self):
        nodes = np.array([[0, 0, i * 5] for i in range(10)], dtype=float)
        t = estimate_tangent(nodes, 5)
        assert abs(np.linalg.norm(t) - 1.0) < 1e-10

    def test_zero_length_segment(self):
        nodes = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=float)
        t = estimate_tangent(nodes, 1)
        # Should return default when norm ~0
        np.testing.assert_array_equal(t, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# estimate_curvature
# ---------------------------------------------------------------------------


class TestEstimateCurvature:
    def test_straight_line(self):
        nodes = np.array([[0, 0, i] for i in range(20)], dtype=float)
        c = estimate_curvature(nodes, 10)
        assert c < 0.1  # Nearly zero curvature

    def test_boundary_index(self):
        nodes = np.array([[0, 0, i] for i in range(10)], dtype=float)
        assert estimate_curvature(nodes, 0) == 0.0
        assert estimate_curvature(nodes, 9) == 0.0

    def test_sharp_turn(self):
        # Straight in z, then sharp turn in y
        nodes = np.array(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, 3],
                [0, 2, 3],
                [0, 3, 3],
                [0, 4, 3],
            ],
            dtype=float,
        )
        c = estimate_curvature(nodes, 4, window=2)
        assert c > 0.3  # Noticeable curvature


# ---------------------------------------------------------------------------
# angle_between
# ---------------------------------------------------------------------------


class TestAngleBetween:
    def test_same_direction(self):
        a = angle_between(np.array([1, 0, 0]), np.array([1, 0, 0]))
        assert abs(a) < 1e-10

    def test_opposite_direction(self):
        a = angle_between(np.array([1, 0, 0]), np.array([-1, 0, 0]))
        assert abs(a - math.pi) < 1e-10

    def test_perpendicular(self):
        a = angle_between(np.array([1, 0, 0]), np.array([0, 1, 0]))
        assert abs(a - math.pi / 2) < 1e-10

    def test_zero_vector(self):
        a = angle_between(np.array([0, 0, 0]), np.array([1, 0, 0]))
        assert a == 0.0

    def test_non_unit_vectors(self):
        a = angle_between(np.array([5, 0, 0]), np.array([0, 3, 0]))
        assert abs(a - math.pi / 2) < 1e-10


# ---------------------------------------------------------------------------
# SpatialIndex
# ---------------------------------------------------------------------------


class TestSpatialIndex:
    def test_empty_index(self):
        idx = SpatialIndex(np.zeros((0, 3)))
        assert idx.query_radius(np.array([0, 0, 0]), 10.0) == []

    def test_empty_nearest(self):
        idx = SpatialIndex(np.zeros((0, 3)))
        dists, indices = idx.query_nearest(np.array([0, 0, 0]))
        assert len(dists) == 0
        assert len(indices) == 0

    def test_query_radius(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=float)
        idx = SpatialIndex(pts)
        result = idx.query_radius(np.array([0, 0, 0]), 2.0)
        assert 0 in result
        assert 1 in result
        assert 2 not in result

    def test_query_nearest_single(self):
        pts = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
        idx = SpatialIndex(pts)
        dists, indices = idx.query_nearest(np.array([4, 0, 0]), k=1)
        assert indices[0] == 1
        assert abs(dists[0] - 1.0) < 1e-10

    def test_query_nearest_k(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0]], dtype=float)
        idx = SpatialIndex(pts)
        dists, indices = idx.query_nearest(np.array([0.5, 0, 0]), k=2)
        assert len(indices) == 2
        assert set(indices) == {0, 1}

    def test_points_property(self):
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        idx = SpatialIndex(pts)
        np.testing.assert_array_equal(idx.points, pts)
