"""Tests for mesh extraction."""

from __future__ import annotations

import numpy as np
import pytest

from connectomics_pipeline.fragments.mesh import MeshExtractor


class TestMeshExtractor:
    def test_extract_cube(self):
        """Solid cube should produce a valid mesh."""
        extractor = MeshExtractor(resolution=(1.0, 1.0, 1.0))
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mesh = extractor.extract(mask)
        assert mesh.num_vertices > 0
        assert mesh.num_faces > 0

    def test_extract_sphere(self):
        """Spherical mask should produce a mesh."""
        extractor = MeshExtractor(resolution=(1.0, 1.0, 1.0))
        mask = np.zeros((30, 30, 30), dtype=np.uint8)
        center = np.array([15, 15, 15])
        for z in range(30):
            for y in range(30):
                for x in range(30):
                    if np.linalg.norm(np.array([z, y, x]) - center) < 10:
                        mask[z, y, x] = 1
        mesh = extractor.extract(mask)
        assert mesh.num_vertices > 0
        assert mesh.num_faces > 0

    def test_extract_too_small(self):
        """Mask with < 4 voxels should return empty mesh."""
        extractor = MeshExtractor()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[5, 5, 5] = 1
        mask[5, 5, 6] = 1
        mesh = extractor.extract(mask)
        assert mesh.num_vertices == 0
        assert mesh.num_faces == 0

    def test_extract_empty(self):
        """All-zero mask should return empty mesh."""
        extractor = MeshExtractor()
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mesh = extractor.extract(mask)
        assert mesh.num_vertices == 0
        assert mesh.num_faces == 0

    def test_resolution_applied(self):
        """Vertex coordinates should be scaled by resolution."""
        res = (30.0, 8.0, 8.0)
        extractor = MeshExtractor(resolution=res)
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[5:15, 5:15, 5:15] = 1
        mesh = extractor.extract(mask)
        # Vertices should be in physical coordinates, not voxel
        # The max extent in z should be ~15*30 = 450 area, not ~15
        assert mesh.vertices[:, 0].max() > 20  # z scaled by 30

    def test_labeled_mask(self):
        """Mask with label > 1 treated as binary."""
        extractor = MeshExtractor(resolution=(1.0, 1.0, 1.0))
        mask = np.zeros((20, 20, 20), dtype=np.uint32)
        mask[5:15, 5:15, 5:15] = 5  # Label 5
        mesh = extractor.extract(mask)
        assert mesh.num_vertices > 0
