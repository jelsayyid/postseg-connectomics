"""Export fragment graph to GraphML and JSON formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np

from connectomics_pipeline.graph.fragment_graph import FragmentGraph
from connectomics_pipeline.utils.logging import get_logger

logger = get_logger("export.graph")


def export_graphml(graph: FragmentGraph, path: str | Path) -> None:
    """Export fragment graph to GraphML format.

    Args:
        graph: Fragment adjacency graph.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to serializable types
    g = graph.graph.copy()
    for node in g.nodes():
        data = g.nodes[node]
        for key, val in list(data.items()):
            if isinstance(val, np.ndarray):
                data[key] = val.tolist().__repr__()
            elif isinstance(val, list) and val and isinstance(val[0], np.ndarray):
                data[key] = [v.tolist() for v in val].__repr__()

    for u, v in g.edges():
        data = g.edges[u, v]
        for key, val in list(data.items()):
            if isinstance(val, np.ndarray):
                data[key] = val.tolist().__repr__()

    nx.write_graphml(g, str(path))
    logger.info("Exported graph to %s", path)


def export_json(graph: FragmentGraph, path: str | Path) -> None:
    """Export fragment graph to JSON format.

    Args:
        graph: Fragment adjacency graph.
        path: Output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = nx.node_link_data(graph.graph)
    # Convert numpy arrays
    _convert_numpy(data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    logger.info("Exported graph JSON to %s", path)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _convert_numpy(obj):
    """Recursively convert numpy types in a dict/list structure."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, np.ndarray):
                obj[k] = v.tolist()
            elif isinstance(v, (dict, list)):
                _convert_numpy(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, np.ndarray):
                obj[i] = v.tolist()
            elif isinstance(v, (dict, list)):
                _convert_numpy(v)
