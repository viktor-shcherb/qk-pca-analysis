"""YAML configuration parsing and dataclasses."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    dataset_name: str = "viktoroo/sniffed-qk-smollm2-360m-tr512-pre-rope"
    split: Optional[str] = None  # None = auto-discover from repo structure
    kinds: List[str] = field(default_factory=lambda: ["q", "k"])
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    max_samples: Optional[int] = None
    sample_heads: Optional[int] = None  # randomly sample N heads from discovered set
    seed: Optional[int] = None  # random seed for reproducible sampling


@dataclass
class PCAConfig:
    n_components: int = 10
    backend: str = "sklearn"  # "sklearn" or "torch"


@dataclass
class ParallelConfig:
    enabled: bool = True
    max_workers: Optional[int] = None  # None = min(cpu_count, 4)
    backend: str = "process"


@dataclass
class AnalysisConfig:
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    correlation_threshold: float = 0.3  # |r| above this = "high correlation"


@dataclass
class OutputConfig:
    results_dir: str = "results"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"

    @property
    def results_path(self) -> Path:
        return Path(self.output.results_dir)


def _build_dataclass(cls, raw: dict | None):
    """Recursively instantiate a dataclass from a (possibly nested) dict."""
    if raw is None:
        return cls()
    kwargs = {}
    for f in cls.__dataclass_fields__:
        if f not in raw:
            continue
        field_type = cls.__dataclass_fields__[f].type
        # Handle nested dataclasses
        if isinstance(raw[f], dict) and hasattr(field_type, "__dataclass_fields__"):
            kwargs[f] = _build_dataclass(field_type, raw[f])
        else:
            kwargs[f] = raw[f]
    return cls(**kwargs)


def _resolve_nested_dataclass(cls, raw: dict | None):
    """Build a top-level Config, resolving nested sections."""
    if raw is None:
        return cls()
    kwargs = {}
    field_map = {
        "data": DataConfig,
        "pca": PCAConfig,
        "analysis": AnalysisConfig,
        "output": OutputConfig,
    }
    for key, value in raw.items():
        if key in field_map and isinstance(value, dict):
            kwargs[key] = _build_dataclass(field_map[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a ``Config`` instance.

    Environment variables in values are expanded (``$VAR`` or ``${VAR}``).
    """
    path = Path(path)
    text = path.read_text()
    text = os.path.expandvars(text)
    raw = yaml.safe_load(text) or {}

    # Handle the nested 'parallel' inside 'analysis'
    if "analysis" in raw and isinstance(raw["analysis"], dict):
        if "parallel" in raw["analysis"] and isinstance(raw["analysis"]["parallel"], dict):
            raw["analysis"]["parallel"] = _build_dataclass(
                ParallelConfig, raw["analysis"]["parallel"]
            )

    return _resolve_nested_dataclass(Config, raw)
