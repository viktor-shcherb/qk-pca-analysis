"""Tests for qk_pca.data."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from qk_pca.data import HeadTarget, _CONFIG_RE, discover_available_heads, load_head_data, resolve_split

MOCK_TARGET = "qk_pca.data.list_repo_files"


class TestHeadTarget:
    def test_config_name_zero_padded(self):
        t = HeadTarget(layer=0, head=0, kind="q")
        assert t.config_name == "l00h00q"

    def test_config_name_two_digits(self):
        t = HeadTarget(layer=12, head=7, kind="k")
        assert t.config_name == "l12h07k"

    def test_str(self):
        t = HeadTarget(layer=3, head=5, kind="q")
        assert str(t) == "l03h05q"

    def test_frozen(self):
        t = HeadTarget(layer=0, head=0, kind="q")
        with pytest.raises(AttributeError):
            t.layer = 1

    def test_equality(self):
        a = HeadTarget(layer=1, head=2, kind="q")
        b = HeadTarget(layer=1, head=2, kind="q")
        assert a == b

    def test_inequality(self):
        a = HeadTarget(layer=1, head=2, kind="q")
        b = HeadTarget(layer=1, head=2, kind="k")
        assert a != b

    def test_hashable(self):
        """Frozen dataclasses are hashable."""
        t = HeadTarget(layer=0, head=0, kind="q")
        assert hash(t) is not None
        s = {t}
        assert t in s


class TestConfigRegex:
    @pytest.mark.parametrize("name,expected", [
        ("l00h00q", (0, 0, "q")),
        ("l12h07k", (12, 7, "k")),
        ("l99h99q", (99, 99, "q")),
        ("l0h0q", (0, 0, "q")),
    ])
    def test_valid_names(self, name, expected):
        m = _CONFIG_RE.match(name)
        assert m is not None
        assert (int(m.group(1)), int(m.group(2)), m.group(3)) == expected

    @pytest.mark.parametrize("name", [
        "l00h00",       # missing kind
        "l00h00x",      # invalid kind
        "README.md",
        ".gitattributes",
        "",
    ])
    def test_invalid_names(self, name):
        assert _CONFIG_RE.match(name) is None


class TestDiscoverAvailableHeads:
    @patch(MOCK_TARGET)
    def test_discovers_configs_at_root(self, mock_files):
        mock_files.return_value = [
            "l00h00q/data.parquet",
            "l00h00k/data.parquet",
            "l01h00q/data.parquet",
            "README.md",
        ]
        prefix, heads = discover_available_heads("test/repo")
        assert prefix is None  # configs at repo root
        assert len(heads) == 3
        assert HeadTarget(0, 0, "q") in heads
        assert HeadTarget(0, 0, "k") in heads
        assert HeadTarget(1, 0, "q") in heads

    @patch(MOCK_TARGET)
    def test_discovers_configs_under_prefix(self, mock_files):
        """Realistic layout: PREFIX/l00h00q/data.parquet"""
        mock_files.return_value = [
            "HuggingFaceTB_SmolLM2_360M/l00h00q/data.parquet",
            "HuggingFaceTB_SmolLM2_360M/l00h00k/data.parquet",
            "HuggingFaceTB_SmolLM2_360M/l01h00q/data.parquet",
            "README.md",
            ".gitattributes",
        ]
        prefix, heads = discover_available_heads("test/repo")
        assert prefix == "HuggingFaceTB_SmolLM2_360M"
        assert len(heads) == 3
        assert HeadTarget(0, 0, "q") in heads

    @patch(MOCK_TARGET)
    def test_filter_by_kinds(self, mock_files):
        mock_files.return_value = [
            "pfx/l00h00q/data.parquet",
            "pfx/l00h00k/data.parquet",
        ]
        _, heads = discover_available_heads("test/repo", kinds=["q"])
        assert len(heads) == 1
        assert heads[0].kind == "q"

    @patch(MOCK_TARGET)
    def test_filter_by_layers(self, mock_files):
        mock_files.return_value = [
            "pfx/l00h00q/data.parquet",
            "pfx/l01h00q/data.parquet",
            "pfx/l02h00q/data.parquet",
        ]
        _, heads = discover_available_heads("test/repo", layers=[0, 2])
        assert len(heads) == 2
        assert heads[0].layer == 0
        assert heads[1].layer == 2

    @patch(MOCK_TARGET)
    def test_filter_by_heads(self, mock_files):
        mock_files.return_value = [
            "pfx/l00h00q/data.parquet",
            "pfx/l00h01q/data.parquet",
            "pfx/l00h02q/data.parquet",
        ]
        _, heads = discover_available_heads("test/repo", heads=[1])
        assert len(heads) == 1
        assert heads[0].head == 1

    @patch(MOCK_TARGET)
    def test_sorted_output(self, mock_files):
        mock_files.return_value = [
            "pfx/l02h01k/data.parquet",
            "pfx/l00h00q/data.parquet",
            "pfx/l01h00q/data.parquet",
            "pfx/l00h00k/data.parquet",
        ]
        _, heads = discover_available_heads("test/repo")
        layers_heads_kinds = [(h.layer, h.head, h.kind) for h in heads]
        assert layers_heads_kinds == sorted(layers_heads_kinds)

    @patch(MOCK_TARGET)
    def test_deduplication(self, mock_files):
        """Multiple files under same config dir should yield one target."""
        mock_files.return_value = [
            "pfx/l00h00q/data.parquet",
            "pfx/l00h00q/metadata.json",
            "pfx/l00h00q/other_file.txt",
        ]
        _, heads = discover_available_heads("test/repo")
        assert len(heads) == 1

    @patch(MOCK_TARGET)
    def test_empty_repo(self, mock_files):
        mock_files.return_value = []
        prefix, heads = discover_available_heads("test/repo")
        assert prefix is None
        assert heads == []

    @patch(MOCK_TARGET)
    def test_combined_filters(self, mock_files):
        mock_files.return_value = [
            "pfx/l00h00q/data.parquet",
            "pfx/l00h00k/data.parquet",
            "pfx/l00h01q/data.parquet",
            "pfx/l01h00q/data.parquet",
            "pfx/l01h01k/data.parquet",
        ]
        _, heads = discover_available_heads(
            "test/repo", kinds=["q"], layers=[0], heads=[0]
        )
        assert len(heads) == 1
        assert heads[0] == HeadTarget(0, 0, "q")


def _make_fake_parquet(tmp_path, n=50, dim=64):
    """Create a fake parquet file and return its path."""
    rng = np.random.default_rng(0)
    vectors = [rng.standard_normal(dim).astype(np.float32).tolist() for _ in range(n)]
    positions = list(range(n))
    df = pd.DataFrame({"vector": vectors, "position": positions})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    return str(path)


class TestLoadHeadData:
    @patch("qk_pca.data._download_parquet")
    def test_basic_load(self, mock_download, tmp_path):
        mock_download.return_value = _make_fake_parquet(tmp_path, n=50, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        vectors, positions = load_head_data("test/repo", target, split="pfx")
        mock_download.assert_called_once_with("test/repo", target, "pfx")
        assert vectors.shape == (50, 64)
        assert vectors.dtype == np.float32
        assert positions.shape == (50,)
        assert positions.dtype == np.int32

    @patch("qk_pca.data._download_parquet")
    def test_max_samples_truncation(self, mock_download, tmp_path):
        mock_download.return_value = _make_fake_parquet(tmp_path, n=100, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        vectors, positions = load_head_data("test/repo", target, max_samples=30)
        assert vectors.shape[0] == 30
        assert positions.shape[0] == 30

    @patch("qk_pca.data._download_parquet")
    def test_max_samples_no_truncation_when_smaller(self, mock_download, tmp_path):
        mock_download.return_value = _make_fake_parquet(tmp_path, n=20, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        vectors, positions = load_head_data("test/repo", target, max_samples=100)
        assert vectors.shape[0] == 20

    @patch("qk_pca.data._download_parquet")
    def test_max_samples_none(self, mock_download, tmp_path):
        mock_download.return_value = _make_fake_parquet(tmp_path, n=50, dim=64)
        target = HeadTarget(layer=0, head=0, kind="q")
        vectors, positions = load_head_data("test/repo", target, max_samples=None)
        assert vectors.shape[0] == 50

    @patch("qk_pca.data._download_parquet")
    def test_split_none(self, mock_download, tmp_path):
        mock_download.return_value = _make_fake_parquet(tmp_path, n=10, dim=8)
        target = HeadTarget(layer=0, head=0, kind="q")
        load_head_data("test/repo", target, split=None)
        mock_download.assert_called_once_with("test/repo", target, None)


class TestResolveSplit:
    def test_returns_split_when_set(self):
        """Should return split as-is without calling discover."""
        assert resolve_split("test/repo", "my_prefix") == "my_prefix"

    @patch(MOCK_TARGET)
    def test_discovers_prefix_when_split_is_none(self, mock_files):
        mock_files.return_value = [
            "HuggingFaceTB_SmolLM2_360M/l00h00q/data.parquet",
        ]
        result = resolve_split("test/repo", None)
        assert result == "HuggingFaceTB_SmolLM2_360M"

    @patch(MOCK_TARGET)
    def test_returns_none_when_no_prefix(self, mock_files):
        mock_files.return_value = [
            "l00h00q/data.parquet",
        ]
        result = resolve_split("test/repo", None)
        assert result is None
