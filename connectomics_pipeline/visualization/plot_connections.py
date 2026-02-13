"""3D visualization of fragments and connections."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.utils.logging import get_logger
from connectomics_pipeline.utils.types import CandidateConnection, ConnectionStatus

logger = get_logger("visualization.plot_connections")

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


STATUS_COLORS = {
    ConnectionStatus.ACCEPTED: "green",
    ConnectionStatus.REJECTED: "red",
    ConnectionStatus.AMBIGUOUS: "orange",
    ConnectionStatus.PROPOSED: "gray",
}


def plot_connections_3d(
    candidates: List[CandidateConnection],
    store: FragmentStore,
    output_path: Optional[str | Path] = None,
    title: str = "Fragment Connections",
) -> None:
    """Plot fragments and connections in 3D, color-coded by status.

    Args:
        candidates: Candidate connections to plot.
        store: Fragment store for centroid data.
        output_path: If provided, save figure to this path.
        title: Plot title.
    """
    if not _HAS_MPL:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot fragment centroids
    fragments = store.all_fragments()
    if fragments:
        centroids = np.array([f.centroid for f in fragments])
        ax.scatter(
            centroids[:, 2],  # x
            centroids[:, 1],  # y
            centroids[:, 0],  # z
            c="blue",
            alpha=0.3,
            s=10,
            label="Fragments",
        )

    # Plot connections
    for status in [
        ConnectionStatus.REJECTED,
        ConnectionStatus.AMBIGUOUS,
        ConnectionStatus.ACCEPTED,
    ]:
        status_candidates = [c for c in candidates if c.status == status]
        if not status_candidates:
            continue

        for c in status_candidates:
            ax.plot(
                [c.endpoint_a[2], c.endpoint_b[2]],
                [c.endpoint_a[1], c.endpoint_b[1]],
                [c.endpoint_a[0], c.endpoint_b[0]],
                color=STATUS_COLORS[status],
                alpha=0.5,
                linewidth=1,
            )

        # Add to legend
        ax.plot(
            [], [], color=STATUS_COLORS[status], label=f"{status.value} ({len(status_candidates)})"
        )

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title(title)
    ax.legend()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        logger.info("Saved connection plot to %s", path)
    else:
        plt.show()

    plt.close(fig)
