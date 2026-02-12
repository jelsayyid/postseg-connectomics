"""Pipeline configuration loading and validation.

Loads YAML config files and provides typed access to pipeline parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class InputConfig:
    format: str = "hdf5"
    path: str = ""
    dataset: str = "labels"
    resolution: tuple = (30.0, 8.0, 8.0)
    chunk_size: tuple = (128, 256, 256)
    chunk_overlap: tuple = (8, 16, 16)


@dataclass
class FragmentConfig:
    min_voxel_count: int = 100
    extract_skeletons: bool = True
    extract_meshes: bool = False
    skeleton_method: str = "teasar"
    skeleton_params: Dict[str, Any] = field(default_factory=lambda: {
        "invalidation_d0": 10,
        "dust_threshold": 100,
    })


@dataclass
class GraphConfig:
    construction_method: str = "proximity"
    max_distance_nm: float = 2000.0


@dataclass
class CandidateConfig:
    max_endpoint_distance_nm: float = 1500.0
    min_alignment_score: float = 0.3
    min_composite_score: float = 0.2
    weights: Dict[str, float] = field(default_factory=lambda: {
        "proximity": 0.35,
        "alignment": 0.30,
        "continuity": 0.25,
        "size": 0.10,
    })


@dataclass
class RuleConfig:
    name: str = ""
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    accept_threshold: float = 0.8
    reject_threshold: float = 0.3
    rules: List[RuleConfig] = field(default_factory=list)


@dataclass
class AssemblyConfig:
    min_structure_fragments: int = 2
    flag_ambiguous: bool = True
    detect_cycles: bool = True
    max_branch_order: int = 10


@dataclass
class ExportConfig:
    formats: List[str] = field(default_factory=lambda: ["graphml", "csv"])
    output_dir: str = "./output"
    include_rejected: bool = False
    include_ambiguous: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "pipeline.log"
    console: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    name: str = "post-segmentation-pipeline"
    version: str = "0.1.0"
    seed: int = 42

    input: InputConfig = field(default_factory=InputConfig)
    fragments: FragmentConfig = field(default_factory=FragmentConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    assembly: AssemblyConfig = field(default_factory=AssemblyConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _build_dataclass(cls, data: dict):
    """Recursively build a dataclass from a dict, ignoring unknown keys."""
    if data is None:
        return cls()
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {}
    for k, v in data.items():
        if k not in field_names:
            continue
        f = next(f for f in dataclasses.fields(cls) if f.name == k)
        # If the field type is itself a dataclass, recurse
        if dataclasses.is_dataclass(f.type):
            filtered[k] = _build_dataclass(f.type, v)
        elif f.type == tuple or (hasattr(f.type, '__origin__') and f.type.__origin__ is tuple):
            filtered[k] = tuple(v) if isinstance(v, list) else v
        else:
            filtered[k] = v
    return cls(**filtered)


def load_config(path: str | Path) -> PipelineConfig:
    """Load pipeline configuration from a YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Fully populated PipelineConfig object.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        return PipelineConfig()

    pipeline_raw = raw.get("pipeline", {})
    config = PipelineConfig(
        name=pipeline_raw.get("name", "post-segmentation-pipeline"),
        version=pipeline_raw.get("version", "0.1.0"),
        seed=pipeline_raw.get("seed", 42),
    )

    if "input" in raw:
        config.input = _build_dataclass(InputConfig, raw["input"])
    if "fragments" in raw:
        config.fragments = _build_dataclass(FragmentConfig, raw["fragments"])
    if "graph" in raw:
        config.graph = _build_dataclass(GraphConfig, raw["graph"])
    if "candidates" in raw:
        config.candidates = _build_dataclass(CandidateConfig, raw["candidates"])
    if "validation" in raw:
        vdata = raw["validation"]
        config.validation = ValidationConfig(
            accept_threshold=vdata.get("accept_threshold", 0.8),
            reject_threshold=vdata.get("reject_threshold", 0.3),
            rules=[
                RuleConfig(name=r["name"], params=r.get("params", {}))
                for r in vdata.get("rules", [])
            ],
        )
    if "assembly" in raw:
        config.assembly = _build_dataclass(AssemblyConfig, raw["assembly"])
    if "export" in raw:
        config.export = _build_dataclass(ExportConfig, raw["export"])
    if "logging" in raw:
        config.logging = _build_dataclass(LoggingConfig, raw["logging"])

    return config


def save_config(config: PipelineConfig, path: str | Path) -> None:
    """Save pipeline configuration to a YAML file for reproducibility."""
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            result = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                result[f.name] = _to_dict(val)
            return result
        elif isinstance(obj, (list, tuple)):
            return [_to_dict(v) for v in obj]
        elif isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    from enum import Enum
    path = Path(path)
    data = _to_dict(config)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
