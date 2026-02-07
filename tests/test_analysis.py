"""Tests for qk_pca.analysis."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from qk_pca.analysis import _resolve_max_workers, analyze_single_head, run_analysis
from qk_pca.config import (
    AnalysisConfig,
    Config,
    DataConfig,
    OutputConfig,
    ParallelConfig,
    PCAConfig,
)
from qk_pca.correlation import CorrelationResult
from qk_pca.data import HeadTarget


class TestResolveMaxWorkers:
    def test_disabled(self):
        cfg = Config(analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)))
        assert _resolve_max_workers(cfg) is None

    def test_explicit_workers(self):
        cfg = Config(
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=True, max_workers=2))
        )
        assert _resolve_max_workers(cfg) == 2

    def test_auto_workers(self):
        cfg = Config(
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=True, max_workers=None))
        )
        result = _resolve_max_workers(cfg)
        import os
        assert result == min(os.cpu_count() or 1, 4)
        assert result >= 1


def _make_fake_vectors(n=50, dim=64):
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors[:, 0] = np.arange(n, dtype=np.float32) * 2  # strong PC0-position corr
    positions = np.arange(n, dtype=np.int32)
    return vectors, positions


class TestAnalyzeSingleHead:
    @patch("qk_pca.analysis.load_head_data")
    def test_returns_correct_structure(self, mock_load):
        mock_load.return_value = _make_fake_vectors(n=50, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        result_target, correlations, n_samples = analyze_single_head(
            dataset_name="test/repo",
            target=target,
            split="train",
            n_components=5,
            pca_backend="sklearn",
            device="cpu",
            max_samples=None,
        )
        assert result_target == target
        assert len(correlations) == 5
        assert n_samples == 50
        assert all(isinstance(c, CorrelationResult) for c in correlations)

    @patch("qk_pca.analysis.load_head_data")
    def test_first_pc_correlates_with_position(self, mock_load):
        mock_load.return_value = _make_fake_vectors(n=100, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        _, correlations, _ = analyze_single_head(
            dataset_name="test/repo",
            target=target,
            split="train",
            n_components=5,
            pca_backend="sklearn",
            device="cpu",
            max_samples=None,
        )
        # PC0 should correlate strongly with position since we injected it
        assert abs(correlations[0].correlation) > 0.9

    @patch("qk_pca.analysis.load_head_data")
    def test_passes_max_samples(self, mock_load):
        mock_load.return_value = _make_fake_vectors(n=30, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        analyze_single_head(
            dataset_name="test/repo",
            target=target,
            split="train",
            n_components=3,
            pca_backend="sklearn",
            device="cpu",
            max_samples=30,
        )
        mock_load.assert_called_once_with(
            "test/repo", target, split="train", max_samples=30
        )


class TestRunAnalysis:
    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_sequential_produces_csv_and_json(self, mock_load, mock_discover, tmp_path):
        targets = [
            HeadTarget(layer=0, head=0, kind="q"),
            HeadTarget(layer=0, head=0, kind="k"),
        ]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=50, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            pca=PCAConfig(n_components=3),
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        results_dir = run_analysis(cfg)

        csv_path = results_dir / "correlations.csv"
        json_path = results_dir / "summary.json"
        assert csv_path.exists()
        assert json_path.exists()

        df = pd.read_csv(csv_path)
        assert set(df.columns) == {
            "layer", "head", "kind", "pca_dim",
            "correlation", "p_value", "explained_variance_ratio",
        }
        # 2 heads * 3 components = 6 rows
        assert len(df) == 6
        assert set(df["kind"].unique()) == {"q", "k"}

        summary = json.loads(json_path.read_text())
        assert summary["dataset"] == "test/repo"
        assert summary["n_components"] == 3
        assert summary["heads_analyzed"] == 2
        assert summary["total_samples"] == 100  # 50 * 2
        assert "timestamp" in summary
        assert "top_correlations" in summary

        # high_correlation_summary should be present
        hcs = summary["high_correlation_summary"]
        assert hcs["threshold"] == 0.3
        assert hcs["total_components"] == 6
        assert hcs["heads_total"] == 2
        assert isinstance(hcs["high_abs_components"], int)
        assert isinstance(hcs["high_abs_hits"], list)

    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_parallel_produces_same_output(self, mock_load, mock_discover, tmp_path):
        """Test parallel path by replacing ProcessPoolExecutor with ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        targets = [HeadTarget(layer=0, head=0, kind="q")]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=50, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            pca=PCAConfig(n_components=3),
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=True, max_workers=1)),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        # Use ThreadPoolExecutor to avoid pickling issues with mocks
        with patch("qk_pca.analysis.ProcessPoolExecutor", ThreadPoolExecutor):
            results_dir = run_analysis(cfg)

        csv_path = results_dir / "correlations.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 3  # 1 head * 3 components

    @patch("qk_pca.analysis.discover_available_heads")
    def test_no_heads_found(self, mock_discover, tmp_path):
        mock_discover.return_value = (None, [])
        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
        )
        results_dir = run_analysis(cfg)
        assert results_dir == Path(tmp_path / "results")
        # No CSV should be written
        assert not (results_dir / "correlations.csv").exists()

    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_csv_sorted(self, mock_load, mock_discover, tmp_path):
        targets = [
            HeadTarget(layer=1, head=0, kind="q"),
            HeadTarget(layer=0, head=1, kind="k"),
            HeadTarget(layer=0, head=0, kind="q"),
        ]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=30, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            pca=PCAConfig(n_components=2),
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        run_analysis(cfg)
        df = pd.read_csv(tmp_path / "results" / "correlations.csv")

        # Verify sorted by (layer, head, kind, pca_dim)
        sort_keys = df[["layer", "head", "kind", "pca_dim"]].values.tolist()
        assert sort_keys == sorted(sort_keys)

    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_sample_heads(self, mock_load, mock_discover, tmp_path):
        """sample_heads should randomly subset the discovered heads."""
        targets = [HeadTarget(layer=i, head=0, kind="q") for i in range(20)]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=30, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo", sample_heads=3),
            pca=PCAConfig(n_components=2),
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        run_analysis(cfg)
        df = pd.read_csv(tmp_path / "results" / "correlations.csv")
        # 3 sampled heads * 2 components = 6 rows
        assert len(df) == 6
        assert df[["layer", "head", "kind"]].drop_duplicates().shape[0] == 3

    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_high_correlation_summary_counts(self, mock_load, mock_discover, tmp_path):
        """Verify high_correlation_summary counts with known strong signal."""
        targets = [HeadTarget(layer=0, head=0, kind="q")]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=100, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            pca=PCAConfig(n_components=3),
            analysis=AnalysisConfig(
                parallel=ParallelConfig(enabled=False),
                correlation_threshold=0.3,
            ),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        run_analysis(cfg)
        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        hcs = summary["high_correlation_summary"]
        # PC0 has strong injected correlation, should be >= threshold
        assert hcs["high_abs_components"] >= 1
        assert hcs["heads_with_high_abs"] >= 1

    @patch("qk_pca.analysis.discover_available_heads")
    @patch("qk_pca.analysis.load_head_data")
    def test_json_top_correlations_limited(self, mock_load, mock_discover, tmp_path):
        """top_correlations should have at most 20 entries."""
        # Create many heads so we get many rows
        targets = [HeadTarget(layer=i, head=0, kind="q") for i in range(10)]
        mock_discover.return_value = ("pfx", targets)
        mock_load.return_value = _make_fake_vectors(n=30, dim=64)

        cfg = Config(
            data=DataConfig(dataset_name="test/repo"),
            pca=PCAConfig(n_components=5),
            analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)),
            output=OutputConfig(results_dir=str(tmp_path / "results")),
            device="cpu",
        )
        run_analysis(cfg)
        summary = json.loads((tmp_path / "results" / "summary.json").read_text())
        assert len(summary["top_correlations"]) <= 20
