"""Tests for qk_pca.visualization."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np
import pandas as pd
import pytest

from qk_pca.config import Config, DataConfig, PCAConfig
from qk_pca.visualization import scatter_with_position


def _make_fake_dataframe(n=100, dim=64, n_examples=5):
    """Create a fake DataFrame mimicking a head's parquet data."""
    rng = np.random.default_rng(42)
    vectors = [rng.standard_normal(dim).astype(np.float32).tolist() for _ in range(n)]
    example_ids = [i % n_examples for i in range(n)]
    positions = [i // n_examples for i in range(n)]
    return pd.DataFrame({
        "vector": vectors,
        "position": positions,
        "example_id": example_ids,
    })


class TestScatterWithPosition:
    @patch("qk_pca.visualization.load_head_dataframe")
    def test_creates_png(self, mock_load_df, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "scatter.png"
        result = scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output
        )
        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("qk_pca.visualization.load_head_dataframe")
    def test_creates_svg(self, mock_load_df, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "scatter.svg"
        scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output
        )
        assert output.exists()
        assert output.stat().st_size > 0

    @patch("qk_pca.visualization.load_head_dataframe")
    def test_creates_parent_dirs(self, mock_load_df, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "nested" / "dir" / "scatter.png"
        scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output
        )
        assert output.exists()

    @patch("qk_pca.visualization.load_head_dataframe")
    def test_invalid_example_id_raises(self, mock_load_df, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe(n_examples=5)
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "scatter.png"
        with pytest.raises(ValueError, match="example_id=999 not found"):
            scatter_with_position(
                cfg, layer=0, head=0, kind="q", example_id=999, output=output
            )

    @patch("qk_pca.visualization.load_head_dataframe")
    def test_custom_pc_axes(self, mock_load_df, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "scatter_pc2_pc3.png"
        scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output,
            pc_x=2, pc_y=3,
        )
        assert output.exists()

    @patch("qk_pca.visualization.resolve_split", return_value="mypfx")
    @patch("qk_pca.visualization.load_head_dataframe")
    def test_calls_load_correctly(self, mock_load_df, mock_resolve, tmp_path):
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="my/repo", split="mypfx"),
            pca=PCAConfig(n_components=3),
            device="cpu",
        )
        output = tmp_path / "scatter.png"
        scatter_with_position(
            cfg, layer=2, head=5, kind="k", example_id=0, output=output
        )
        mock_resolve.assert_called_once_with("my/repo", "mypfx")
        mock_load_df.assert_called_once_with("my/repo", ANY, split="mypfx")
        # Verify the HeadTarget
        call_target = mock_load_df.call_args[0][1]
        assert call_target.layer == 2
        assert call_target.head == 5
        assert call_target.kind == "k"

    @patch("qk_pca.visualization.resolve_split", return_value="discovered_prefix")
    @patch("qk_pca.visualization.load_head_dataframe")
    def test_auto_discovers_prefix_when_split_none(self, mock_load_df, mock_resolve, tmp_path):
        """When config split is None, resolve_split discovers the prefix."""
        mock_load_df.return_value = _make_fake_dataframe()
        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split=None),
            pca=PCAConfig(n_components=5),
            device="cpu",
        )
        output = tmp_path / "scatter.png"
        scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output
        )
        mock_resolve.assert_called_once_with("test/repo", None)
        mock_load_df.assert_called_once_with("test/repo", ANY, split="discovered_prefix")

    @patch("qk_pca.visualization.load_head_dataframe")
    def test_single_position_sequence(self, mock_load_df, tmp_path):
        """A sequence with all same positions should not crash (pos_max == pos_min)."""
        rng = np.random.default_rng(0)
        n = 50
        vectors = [rng.standard_normal(16).astype(np.float32).tolist() for _ in range(n)]
        # All same position for example_id=0
        positions = [0] * 10 + list(range(10, n - 10 + 10))
        example_ids = [0] * 10 + [1] * (n - 10)

        mock_load_df.return_value = pd.DataFrame({
            "vector": vectors,
            "position": positions,
            "example_id": example_ids,
        })

        cfg = Config(
            data=DataConfig(dataset_name="test/repo", split="pfx"),
            pca=PCAConfig(n_components=3),
            device="cpu",
        )
        output = tmp_path / "scatter.png"
        scatter_with_position(
            cfg, layer=0, head=0, kind="q", example_id=0, output=output
        )
        assert output.exists()
