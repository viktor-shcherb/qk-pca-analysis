"""Tests for qk_pca.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from qk_pca.config import (
    AnalysisConfig,
    Config,
    DataConfig,
    OutputConfig,
    ParallelConfig,
    PCAConfig,
    _build_dataclass,
    load_config,
)


class TestDataclassDefaults:
    def test_data_config_defaults(self):
        dc = DataConfig()
        assert dc.split is None
        assert dc.kinds == ["q", "k"]
        assert dc.layers is None
        assert dc.heads is None
        assert dc.max_samples is None

    def test_pca_config_defaults(self):
        pc = PCAConfig()
        assert pc.n_components == 10
        assert pc.backend == "sklearn"

    def test_parallel_config_defaults(self):
        pc = ParallelConfig()
        assert pc.enabled is True
        assert pc.max_workers is None
        assert pc.backend == "process"

    def test_config_defaults(self):
        cfg = Config()
        assert cfg.device == "auto"
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.pca, PCAConfig)
        assert isinstance(cfg.analysis, AnalysisConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_results_path_property(self):
        cfg = Config(output=OutputConfig(results_dir="foo/bar"))
        assert cfg.results_path == Path("foo/bar")


class TestBuildDataclass:
    def test_none_returns_default(self):
        dc = _build_dataclass(PCAConfig, None)
        assert dc == PCAConfig()

    def test_partial_dict(self):
        dc = _build_dataclass(PCAConfig, {"n_components": 5})
        assert dc.n_components == 5
        assert dc.backend == "sklearn"

    def test_full_dict(self):
        dc = _build_dataclass(DataConfig, {
            "dataset_name": "my/repo",
            "split": "test",
            "kinds": ["q"],
            "layers": [0, 1],
            "heads": [3],
            "max_samples": 100,
        })
        assert dc.dataset_name == "my/repo"
        assert dc.split == "test"
        assert dc.kinds == ["q"]
        assert dc.layers == [0, 1]
        assert dc.heads == [3]
        assert dc.max_samples == 100

    def test_extra_keys_ignored(self):
        dc = _build_dataclass(PCAConfig, {"n_components": 3, "unknown_key": True})
        assert dc.n_components == 3


class TestLoadConfig:
    def test_full_config(self, tmp_path):
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("""\
data:
  dataset_name: my/dataset
  split: validation
  kinds: [k]
  layers: [0, 1, 2]
  heads: [0]
  max_samples: 500
pca:
  n_components: 5
  backend: torch
analysis:
  parallel:
    enabled: false
    max_workers: 2
    backend: process
output:
  results_dir: /tmp/test-results
device: cpu
""")
        cfg = load_config(cfg_file)
        assert cfg.data.dataset_name == "my/dataset"
        assert cfg.data.split == "validation"
        assert cfg.data.kinds == ["k"]
        assert cfg.data.layers == [0, 1, 2]
        assert cfg.data.heads == [0]
        assert cfg.data.max_samples == 500
        assert cfg.pca.n_components == 5
        assert cfg.pca.backend == "torch"
        assert cfg.analysis.parallel.enabled is False
        assert cfg.analysis.parallel.max_workers == 2
        assert cfg.output.results_dir == "/tmp/test-results"
        assert cfg.device == "cpu"

    def test_minimal_config(self, tmp_path):
        cfg_file = tmp_path / "minimal.yaml"
        cfg_file.write_text("device: cpu\n")
        cfg = load_config(cfg_file)
        assert cfg.device == "cpu"
        # Everything else is default
        assert cfg.data.split is None
        assert cfg.pca.n_components == 10

    def test_empty_config(self, tmp_path):
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        assert cfg == Config()

    def test_env_var_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_DATASET", "expanded/dataset")
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text("""\
data:
  dataset_name: $MY_DATASET
""")
        cfg = load_config(cfg_file)
        assert cfg.data.dataset_name == "expanded/dataset"

    def test_partial_sections(self, tmp_path):
        """Omitting sub-sections still yields defaults."""
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text("""\
pca:
  n_components: 3
""")
        cfg = load_config(cfg_file)
        assert cfg.pca.n_components == 3
        assert cfg.pca.backend == "sklearn"
        # data, analysis, output all defaults
        assert cfg.data.kinds == ["q", "k"]
        assert cfg.analysis.parallel.enabled is True

    def test_example_config_loads(self):
        """The shipped example.yaml must load without errors."""
        cfg = load_config("configs/example.yaml")
        assert cfg.data.dataset_name == "viktoroo/sniffed-qk-smollm2-360m-tr512-pre-rope"
        assert cfg.pca.n_components == 10
        assert cfg.analysis.parallel.enabled is True
