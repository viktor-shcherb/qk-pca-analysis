"""Tests for qk_pca.cli."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest


VENV_PYTHON = "/Users/Viktor/PycharmProjects/qk-pca-analysis/venv/bin/python"


class TestCLIHelp:
    """Verify CLI entry points parse arguments correctly."""

    def test_analyze_help(self):
        result = subprocess.run(
            [VENV_PYTHON, "-m", "qk_pca.cli", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_visualize_help(self):
        result = subprocess.run(
            [VENV_PYTHON, "-m", "qk_pca.cli", "visualize", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--layer" in result.stdout
        assert "--head" in result.stdout
        assert "--kind" in result.stdout
        assert "--example-id" in result.stdout
        assert "--output" in result.stdout


class TestCLIEntryPoints:
    def test_pca_analyze_entry_point_exists(self):
        result = subprocess.run(
            ["/Users/Viktor/PycharmProjects/qk-pca-analysis/venv/bin/pca-analyze", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_pca_visualize_entry_point_exists(self):
        result = subprocess.run(
            ["/Users/Viktor/PycharmProjects/qk-pca-analysis/venv/bin/pca-visualize", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "--layer" in result.stdout

    def test_analyze_missing_config(self):
        result = subprocess.run(
            ["/Users/Viktor/PycharmProjects/qk-pca-analysis/venv/bin/pca-analyze"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "config" in result.stderr.lower()

    def test_visualize_missing_args(self):
        result = subprocess.run(
            ["/Users/Viktor/PycharmProjects/qk-pca-analysis/venv/bin/pca-visualize"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestMainAnalyze:
    @patch("qk_pca.analysis.run_analysis")
    @patch("qk_pca.config.load_config")
    def test_calls_run_analysis(self, mock_load_config, mock_run, tmp_path, monkeypatch):
        from qk_pca.config import Config
        mock_load_config.return_value = Config()
        mock_run.return_value = tmp_path

        monkeypatch.setattr(sys, "argv", ["pca-analyze", "--config", "test.yaml"])
        from qk_pca.cli import main_analyze
        main_analyze()

        mock_load_config.assert_called_once_with("test.yaml")
        mock_run.assert_called_once()


class TestMainVisualize:
    @patch("qk_pca.visualization.scatter_with_position")
    @patch("qk_pca.config.load_config")
    def test_calls_scatter(self, mock_load_config, mock_scatter, monkeypatch):
        from qk_pca.config import Config
        mock_load_config.return_value = Config()

        monkeypatch.setattr(sys, "argv", [
            "pca-visualize",
            "--config", "test.yaml",
            "--layer", "2",
            "--head", "5",
            "--kind", "k",
            "--example-id", "42",
            "--output", "out.png",
            "--pc-x", "1",
            "--pc-y", "3",
        ])
        from qk_pca.cli import main_visualize
        main_visualize()

        mock_load_config.assert_called_once_with("test.yaml")
        mock_scatter.assert_called_once()
        call_kwargs = mock_scatter.call_args
        assert call_kwargs[1]["layer"] == 2
        assert call_kwargs[1]["head"] == 5
        assert call_kwargs[1]["kind"] == "k"
        assert call_kwargs[1]["example_id"] == 42
        assert call_kwargs[1]["output"] == "out.png"
        assert call_kwargs[1]["pc_x"] == 1
        assert call_kwargs[1]["pc_y"] == 3
