"""Fragment adjacency graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from connectomics_pipeline.utils.types import Fragment


class FragmentGraph:
    """Wrapper around networkx.Graph for fragment adjacency."""

    def __init__(self) -> None:
        self._graph = nx.Graph()

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def add_fragment(self, fragment: Fragment) -> None:
        """Add a fragment as a node."""
        self._graph.add_node(
            fragment.fragment_id,
            centroid=fragment.centroid,
            endpoints=fragment.endpoints,
            voxel_count=fragment.voxel_count,
            label_id=fragment.label_id,
        )

    def add_edge(
        self,
        frag_a: int,
        frag_b: int,
        distance: float,
        endpoint_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs: Any,
    ) -> None:
        """Add an edge between two fragments."""
        attrs: Dict[str, Any] = {"distance": distance}
        if endpoint_pair is not None:
            attrs["endpoint_a"] = endpoint_pair[0]
            attrs["endpoint_b"] = endpoint_pair[1]
        attrs.update(kwargs)
        self._graph.add_edge(frag_a, frag_b, **attrs)

    def has_edge(self, frag_a: int, frag_b: int) -> bool:
        return self._graph.has_edge(frag_a, frag_b)

    def get_neighbors(self, fragment_id: int) -> List[int]:
        """Get neighbor fragment IDs."""
        if fragment_id not in self._graph:
            return []
        return list(self._graph.neighbors(fragment_id))

    def get_edge_data(self, frag_a: int, frag_b: int) -> Optional[Dict[str, Any]]:
        """Get edge attributes."""
        return self._graph.get_edge_data(frag_a, frag_b)

    def edges(self) -> List[Tuple[int, int]]:
        """Return all edges as (frag_a, frag_b) tuples."""
        return list(self._graph.edges())

    def nodes(self) -> Set[int]:
        """Return all node IDs."""
        return set(self._graph.nodes())

    def node_data(self, fragment_id: int) -> Dict[str, Any]:
        """Get node attributes."""
        return dict(self._graph.nodes[fragment_id])

    def subgraph(self, fragment_ids: List[int]) -> FragmentGraph:
        """Return a subgraph containing only the specified fragments."""
        new = FragmentGraph()
        new._graph = self._graph.subgraph(fragment_ids).copy()
        return new
