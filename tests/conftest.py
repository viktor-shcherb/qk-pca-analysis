"""Shared fixtures for tests."""

from __future__ import annotations

import numpy as np
import pytest

from qk_pca.config import (
    AnalysisConfig,
    Config,
    DataConfig,
    OutputConfig,
    ParallelConfig,
    PCAConfig,
)
from qk_pca.data import HeadTarget


@pytest.fixture
def rng():
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_vectors(rng):
    """Synthetic vectors (100 samples, 64 dims) with known structure.

    PC0 correlates strongly with position (r ~ 0.99).
    """
    n, dim = 100, 64
    positions = np.arange(n, dtype=np.int32)
    # Build vectors where the first dimension correlates with position
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    vectors[:, 0] = positions * 2.0 + rng.standard_normal(n).astype(np.float32) * 0.1
    return vectors, positions


@pytest.fixture
def head_target():
    return HeadTarget(layer=0, head=0, kind="q")


@pytest.fixture
def config_for_test(tmp_path):
    """A Config instance pointing at a tmp results dir, sequential mode."""
    return Config(
        data=DataConfig(
            dataset_name="test/dataset",
            split=None,
            kinds=["q", "k"],
            max_samples=50,
        ),
        pca=PCAConfig(n_components=5, backend="sklearn"),
        analysis=AnalysisConfig(parallel=ParallelConfig(enabled=False)),
        output=OutputConfig(results_dir=str(tmp_path / "results")),
        device="cpu",
    )
