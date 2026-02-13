"""Chunk boundary stitching using union-find."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import BoundingBox, Fragment

logger = get_logger("fragments.stitching")


class UnionFind:
    """Disjoint-set / union-find data structure."""

    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        """Return mapping from root -> list of members."""
        result: Dict[int, List[int]] = {}
        for x in self._parent:
            root = self.find(x)
            result.setdefault(root, []).append(x)
        return result


class ChunkStitcher:
    """Identifies and merges fragments across chunk boundaries."""

    def __init__(self, overlap: Tuple[int, ...] = (8, 16, 16)):
        self.overlap = overlap

    def stitch(
        self,
        fragments: List[Fragment],
        chunk_pairs: List[Tuple[Tuple[int, ...], Tuple[int, ...]]],
    ) -> List[Fragment]:
        """Merge fragments that match across chunk boundaries.

        Args:
            fragments: All fragments from all chunks.
            chunk_pairs: Pairs of (chunk_origin_a, chunk_origin_b) for adjacent chunks.

        Returns:
            Merged list of fragments with updated IDs.
        """
        if not fragments:
            return []

        uf = UnionFind()

        # Group fragments by chunk origin
        by_chunk: Dict[Tuple[int, ...], List[Fragment]] = {}
        for f in fragments:
            origin = tuple(f.chunk_origin) if f.chunk_origin else (0, 0, 0)
            by_chunk.setdefault(origin, []).append(f)

        # For each pair of adjacent chunks, find matching boundary fragments
        for origin_a, origin_b in chunk_pairs:
            frags_a = [f for f in by_chunk.get(origin_a, []) if f.is_boundary]
            frags_b = [f for f in by_chunk.get(origin_b, []) if f.is_boundary]

            for fa in frags_a:
                for fb in frags_b:
                    if fa.label_id == fb.label_id and fa.bounding_box.overlaps(fb.bounding_box):
                        uf.union(fa.fragment_id, fb.fragment_id)

        # Merge groups
        groups = uf.groups()
        merged_ids: dict[int, int] = {}
        for root, members in groups.items():
            for m in members:
                merged_ids[m] = root

        # Build merged fragment list
        frag_dict = {f.fragment_id: f for f in fragments}
        merged_fragments: List[Fragment] = []
        seen_roots: set[int] = set()

        for f in fragments:
            root = merged_ids.get(f.fragment_id, f.fragment_id)
            if root in seen_roots:
                continue
            seen_roots.add(root)

            members = groups.get(root, [root])
            if len(members) == 1:
                merged_fragments.append(f)
            else:
                merged = self._merge_fragments(
                    root, [frag_dict[m] for m in members if m in frag_dict]
                )
                merged_fragments.append(merged)

        logger.info(
            "Stitching: %d fragments -> %d after merging",
            len(fragments),
            len(merged_fragments),
        )
        return merged_fragments

    def _merge_fragments(self, root_id: int, members: List[Fragment]) -> Fragment:
        """Merge multiple fragments into one."""
        total_voxels = sum(f.voxel_count for f in members)

        all_min = np.stack([f.bounding_box.min_corner for f in members])
        all_max = np.stack([f.bounding_box.max_corner for f in members])
        merged_bbox = BoundingBox(
            min_corner=all_min.min(axis=0),
            max_corner=all_max.max(axis=0),
        )

        # Weighted centroid
        weights = np.array([f.voxel_count for f in members], dtype=float)
        centroids = np.stack([f.centroid for f in members])
        merged_centroid = np.average(centroids, axis=0, weights=weights)

        # Collect all endpoints
        all_endpoints = []
        for f in members:
            all_endpoints.extend(f.endpoints)

        return Fragment(
            fragment_id=root_id,
            label_id=members[0].label_id,
            voxel_count=total_voxels,
            bounding_box=merged_bbox,
            centroid=merged_centroid,
            endpoints=all_endpoints,
            skeleton=members[0].skeleton,
            mesh=members[0].mesh,
            is_boundary=False,
        )
