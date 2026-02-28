"""Main pipeline orchestrator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from connectomics_pipeline.assembly.assembler import Assembler
from connectomics_pipeline.candidates.generator import CandidateGenerator
from connectomics_pipeline.evaluation.ground_truth import evaluate_decisions
from connectomics_pipeline.export.graph_export import export_graphml, export_json
from connectomics_pipeline.export.metadata_export import (
    export_connection_decisions,
    export_fragment_metadata,
    export_structure_summaries,
)
from connectomics_pipeline.export.neuroglancer_export import export_annotations
from connectomics_pipeline.export.precomputed_segmentation import (
    build_corrected_volume,
    write_precomputed,
)
from connectomics_pipeline.export.swc_export import export_swc
from connectomics_pipeline.fragments.extraction import FragmentExtractor
from connectomics_pipeline.fragments.mesh import MeshExtractor
from connectomics_pipeline.fragments.metadata import compute_endpoints
from connectomics_pipeline.fragments.skeleton import Skeletonizer
from connectomics_pipeline.fragments.store import FragmentStore
from connectomics_pipeline.fragments.stitching import ChunkStitcher
from connectomics_pipeline.graph.builder import GraphBuilder
from connectomics_pipeline.io.volume_reader import BaseVolumeReader
from connectomics_pipeline.utils.config import PipelineConfig, save_config
from connectomics_pipeline.utils.logging import get_logger, setup_logging
from connectomics_pipeline.utils.types import (
    AssembledStructure,
    CandidateConnection,
    ValidationReport,
)
from connectomics_pipeline.validation.pipeline import ValidationPipeline
from connectomics_pipeline.visualization.diagnostics import (
    connection_statistics,
    fragment_statistics,
    structure_statistics,
)

logger = get_logger("pipeline")


class Pipeline:
    """Orchestrates the full post-segmentation connectomics pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.store = FragmentStore()
        self.candidates: List[CandidateConnection] = []
        self.report: Optional[ValidationReport] = None
        self.structures: List[AssembledStructure] = []
        self._volume: Optional[np.ndarray] = None
        self._resolution: Optional[tuple] = None

    def run(self, reader: Optional[BaseVolumeReader] = None) -> List[AssembledStructure]:
        """Run the full pipeline.

        Args:
            reader: Volume reader. If None, creates one from config.

        Returns:
            List of assembled structures.
        """
        setup_logging(self.config.logging)
        logger.info("Starting pipeline: %s v%s", self.config.name, self.config.version)
        start_time = time.time()

        np.random.seed(self.config.seed)

        # Create output directory
        output_dir = Path(self.config.export.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config for reproducibility
        save_config(self.config, output_dir / "pipeline_config.yaml")

        # 1. Load volume
        if reader is None:
            reader = self._create_reader()
        resolution = reader.resolution
        self._resolution = resolution
        logger.info("Volume shape: %s, resolution: %s", reader.shape, resolution)

        # 2. Extract fragments (retain volume for corrected-segmentation export)
        self._volume = reader.read_all()
        self._extract_fragments(reader, resolution)

        # 3. Skeletonize and compute metadata
        self._compute_geometry(resolution)

        # 4. Build graph
        graph_builder = GraphBuilder(self.config.graph)
        graph = graph_builder.build(self.store)

        # 5. Generate candidates
        candidate_gen = CandidateGenerator(self.config.candidates, self.store)
        self.candidates = candidate_gen.generate(graph)

        # 6. Validate
        validator = ValidationPipeline(self.config.validation)
        self.report = validator.validate(self.candidates, self.store, graph)

        # 7. Assemble
        assembler = Assembler(self.config.assembly)
        self.structures = assembler.assemble(graph, self.candidates, self.report, self.store)

        # 8. Export
        self._export(output_dir, graph)

        # 9. Summary
        elapsed = time.time() - start_time
        self._log_summary(elapsed)

        return self.structures

    def _create_reader(self) -> BaseVolumeReader:
        """Create a volume reader from config."""
        fmt = self.config.input.format
        path = self.config.input.path
        resolution = tuple(self.config.input.resolution)
        dataset = self.config.input.dataset

        if fmt == "hdf5":
            from connectomics_pipeline.io.hdf5_reader import HDF5Reader

            return HDF5Reader(path, dataset=dataset, resolution=resolution)
        elif fmt == "zarr":
            from connectomics_pipeline.io.zarr_reader import ZarrReader

            return ZarrReader(path, dataset=dataset, resolution=resolution)
        elif fmt == "precomputed":
            from connectomics_pipeline.io.precomputed_reader import PrecomputedReader

            return PrecomputedReader(path, resolution=resolution)
        elif fmt == "numpy":
            from connectomics_pipeline.io.numpy_reader import NumpyReader

            data = np.load(path)
            return NumpyReader(data, resolution=resolution)
        else:
            raise ValueError(f"Unknown input format: {fmt}")

    def _extract_fragments(self, reader: BaseVolumeReader, resolution) -> None:
        """Extract fragments from all chunks."""
        logger.info("Extracting fragments...")
        extractor = FragmentExtractor(self.config.fragments, resolution=resolution)
        chunk_size = tuple(self.config.input.chunk_size)
        overlap = tuple(self.config.input.chunk_overlap)

        all_fragments = []
        chunk_origins = []
        for offset, chunk in reader.chunk_iterator(chunk_size, overlap):
            frags = extractor.extract_from_chunk(chunk, offset)
            all_fragments.extend(frags)
            chunk_origins.append(offset)

        # Stitch across chunk boundaries
        if len(chunk_origins) > 1:
            stitcher = ChunkStitcher(overlap=overlap)
            chunk_pairs = _compute_chunk_pairs(chunk_origins, chunk_size, overlap)
            all_fragments = stitcher.stitch(all_fragments, chunk_pairs)

        self.store.add_many(all_fragments)
        logger.info("Extracted %d fragments", len(self.store))

    def _compute_geometry(self, resolution) -> None:
        """Compute skeletons, meshes, and endpoints for all fragments."""
        if self.config.fragments.extract_skeletons and self._volume is not None:
            logger.info("Skeletonizing fragments...")
            skeletonizer = Skeletonizer(
                method=self.config.fragments.skeleton_method,
                resolution=resolution,
                params=self.config.fragments.skeleton_params,
            )
            res = np.array(resolution, dtype=float)
            vol_shape = np.array(self._volume.shape)
            skeletonized = failed = 0

            for frag in self.store.all_fragments():
                # Convert bounding box (nm) to voxel indices
                min_vox = np.round(frag.bounding_box.min_corner / res).astype(int)
                max_vox = np.round(frag.bounding_box.max_corner / res).astype(int)
                min_vox = np.clip(min_vox, 0, vol_shape - 1)
                max_vox = np.clip(max_vox, min_vox + 1, vol_shape)

                slices = tuple(slice(int(mn), int(mx)) for mn, mx in zip(min_vox, max_vox))
                subvol = self._volume[slices]
                mask = (subvol == frag.label_id).astype(np.uint32)

                voxel_count = int(mask.sum())
                if voxel_count < self.config.fragments.min_voxel_count:
                    failed += 1
                    continue

                max_skel_vox = self.config.fragments.max_skeleton_voxels
                if voxel_count > max_skel_vox:
                    # For very large fragments, PCA gives axis-aligned endpoints
                    # much faster than TEASAR and captures the two axon tips.
                    frag.endpoints = _pca_endpoints(mask, min_vox, res)
                    skeletonized += 1
                    continue

                try:
                    # Use label_id=1 with binary mask to avoid uint32 overflow
                    # on large original label IDs (XPRESS labels can be >100M)
                    skel = skeletonizer.skeletonize(mask, label_id=1)
                    if skel is not None and skel.num_nodes > 0:
                        # kimimaro outputs local physical coords (anisotropy * voxel_idx)
                        # Translate to global physical space by adding bbox min corner
                        skel.nodes += min_vox * res
                        frag.skeleton = skel
                        skeletonized += 1
                except Exception:
                    failed += 1

            logger.info(
                "Skeletonization: %d/%d fragments skeletonized (%d failed/skipped)",
                skeletonized,
                len(self.store),
                failed,
            )

        # Compute endpoints from skeletons (terminal nodes) or centroid fallback
        for frag in self.store.all_fragments():
            if not frag.endpoints:
                frag.endpoints = compute_endpoints(frag)

        # Rebuild index after endpoint updates
        self.store._rebuild_index()

    def _export(self, output_dir: Path, graph) -> None:
        """Export all results."""
        logger.info("Exporting results to %s", output_dir)

        formats = self.config.export.formats

        if "graphml" in formats:
            export_graphml(graph, output_dir / "fragment_graph.graphml")

        if "json" in formats:
            export_json(graph, output_dir / "fragment_graph.json")

        if "csv" in formats:
            export_fragment_metadata(self.store, output_dir / "fragments.csv")
            if self.candidates and self.report:
                export_connection_decisions(
                    self.candidates, self.report, output_dir / "connections.csv"
                )
            if self.structures:
                export_structure_summaries(self.structures, output_dir / "structures.csv")

        if "swc" in formats:
            for structure in self.structures:
                export_swc(
                    structure,
                    self.store,
                    output_dir / f"structure_{structure.structure_id}.swc",
                )

        if "neuroglancer" in formats:
            if self.candidates:
                export_annotations(
                    self.candidates,
                    self.structures,
                    output_dir / "neuroglancer",
                )

        if "precomputed_seg" in formats:
            if self._volume is not None and self._resolution is not None:
                corrected = build_corrected_volume(
                    self._volume,
                    self.candidates,
                    self.store,
                    self._resolution,
                )
                write_precomputed(
                    corrected,
                    output_dir / "corrected_segmentation",
                    self._resolution,
                )

    def _log_summary(self, elapsed: float) -> None:
        """Log pipeline summary statistics."""
        logger.info("=" * 60)
        logger.info("Pipeline complete in %.1f seconds", elapsed)

        frag_stats = fragment_statistics(self.store)
        logger.info(
            "Fragments: %d (mean size: %.0f voxels)",
            frag_stats["count"],
            frag_stats.get("mean_size", 0),
        )

        if self.report:
            conn_stats = connection_statistics(self.candidates, self.report)
            logger.info(
                "Connections: %d total (%d accepted, %d rejected, %d ambiguous)",
                conn_stats["total"],
                conn_stats["accepted"],
                conn_stats["rejected"],
                conn_stats["ambiguous"],
            )

        struct_stats = structure_statistics(self.structures)
        logger.info("Structures: %d", struct_stats["count"])

        if self.config.export.evaluate_ground_truth and self.candidates:
            gt = evaluate_decisions(self.candidates, self.store)
            logger.info(
                "Ground truth eval (label-ID oracle): "
                "precision=%.3f recall=%.3f F1=%.3f "
                "(TP=%d FP=%d TN=%d FN=%d | ambiguous: %d same-label, %d diff-label)",
                gt["precision"],
                gt["recall"],
                gt["f1"],
                gt["true_positives"],
                gt["false_positives"],
                gt["true_negatives"],
                gt["false_negatives"],
                gt["ambiguous_same_label"],
                gt["ambiguous_diff_label"],
            )

        logger.info("=" * 60)


def _pca_endpoints(mask: np.ndarray, min_vox: np.ndarray, res: np.ndarray) -> list:
    """Fast PCA-based endpoint estimation for large fragments.

    Returns the two voxels at the extremes of the principal axis, translated
    to global physical coordinates.  O(N) in the number of foreground voxels.
    Used as a TEASAR substitute when voxel_count > max_skeleton_voxels.
    """
    coords = np.argwhere(mask)  # (N, 3) local voxel coords
    if len(coords) < 2:
        return [(coords[0] + min_vox) * res] if len(coords) == 1 else []
    centered = coords - coords.mean(axis=0)
    # SVD: Vt[0] is the first right singular vector (principal axis)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    principal = Vt[0]
    proj = centered @ principal
    pt1 = (coords[np.argmin(proj)] + min_vox) * res
    pt2 = (coords[np.argmax(proj)] + min_vox) * res
    return [pt1, pt2]


def _compute_chunk_pairs(chunk_origins, chunk_size, overlap):
    """Compute pairs of adjacent chunk origins."""
    pairs = []
    origins_set = set(tuple(o) for o in chunk_origins)
    step = tuple(cs - ov for cs, ov in zip(chunk_size, overlap))

    for origin in chunk_origins:
        for dim in range(3):
            neighbor = list(origin)
            neighbor[dim] += step[dim]
            neighbor = tuple(neighbor)
            if neighbor in origins_set:
                pairs.append((tuple(origin), neighbor))

    return pairs
