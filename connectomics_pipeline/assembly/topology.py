"""Topology checks for assembled structures."""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx


def check_topology(graph: nx.Graph) -> Tuple[List[str], int]:
    """Check topology of an assembled structure.

    Args:
        graph: Subgraph representing one assembled structure.

    Returns:
        (topology_warnings, num_branch_points)
    """
    warnings: List[str] = []

    # Cycle detection
    cycles = list(nx.cycle_basis(graph))
    if cycles:
        warnings.append(f"Contains {len(cycles)} cycle(s)")

    # Branch point counting (nodes with degree > 2)
    branch_points = [n for n in graph.nodes() if graph.degree(n) > 2]
    num_branches = len(branch_points)
    if num_branches > 0:
        warnings.append(f"Contains {num_branches} branch point(s)")

    # Check for isolated nodes (degree 0)
    isolated = list(nx.isolates(graph))
    if isolated:
        warnings.append(f"Contains {len(isolated)} isolated node(s)")

    return warnings, num_branches


def count_branch_order(graph: nx.Graph, root: int | None = None) -> int:
    """Count the maximum branch order in a tree-like structure.

    Args:
        graph: Structure graph.
        root: Optional root node. If None, uses the node with highest degree.

    Returns:
        Maximum branch order.
    """
    if graph.number_of_nodes() == 0:
        return 0

    if root is None:
        root = max(graph.nodes(), key=lambda n: graph.degree(n))

    # BFS to compute branch order
    max_order = 0
    visited = {root}
    queue = [(root, 0)]

    while queue:
        node, order = queue.pop(0)
        neighbors = [n for n in graph.neighbors(node) if n not in visited]
        if len(neighbors) > 1:
            order += 1
        max_order = max(max_order, order)
        for n in neighbors:
            visited.add(n)
            queue.append((n, order))

    return max_order
