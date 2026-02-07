"""Tests for qk_pca.correlation."""

from __future__ import annotations

import numpy as np
import pytest

from qk_pca.correlation import CorrelationResult, compute_correlations


class TestComputeCorrelations:
    def test_perfect_positive_correlation(self):
        positions = np.arange(100, dtype=np.int32)
        projections = np.column_stack([
            positions.astype(np.float64),  # perfect correlation
            np.ones(100),                  # zero variance -> (0.0, 1.0)
        ])
        evr = np.array([0.5, 0.3])
        results = compute_correlations(projections, positions, evr)
        assert len(results) == 2
        assert results[0].pca_dim == 0
        assert results[0].correlation == pytest.approx(1.0)
        assert results[0].p_value < 1e-10
        assert results[0].explained_variance_ratio == 0.5

    def test_perfect_negative_correlation(self):
        positions = np.arange(100, dtype=np.int32)
        projections = (-positions.astype(np.float64)).reshape(-1, 1)
        evr = np.array([0.8])
        results = compute_correlations(projections, positions, evr)
        assert results[0].correlation == pytest.approx(-1.0)

    def test_no_correlation(self, rng):
        """Uncorrelated random data should yield |r| close to 0."""
        n = 10000
        positions = np.arange(n, dtype=np.int32)
        projections = rng.standard_normal((n, 1))
        evr = np.array([0.5])
        results = compute_correlations(projections, positions, evr)
        assert abs(results[0].correlation) < 0.05

    def test_zero_variance_projections(self):
        """Constant projections -> correlation=0, p_value=1."""
        positions = np.arange(50, dtype=np.int32)
        projections = np.ones((50, 1))
        evr = np.array([0.1])
        results = compute_correlations(projections, positions, evr)
        assert results[0].correlation == 0.0
        assert results[0].p_value == 1.0

    def test_zero_variance_positions(self):
        """Constant positions -> correlation=0, p_value=1."""
        positions = np.full(50, 5, dtype=np.int32)
        projections = np.arange(50, dtype=np.float64).reshape(-1, 1)
        evr = np.array([0.2])
        results = compute_correlations(projections, positions, evr)
        assert results[0].correlation == 0.0
        assert results[0].p_value == 1.0

    def test_multiple_dimensions(self, rng):
        n = 100
        positions = np.arange(n, dtype=np.int32)
        projections = np.column_stack([
            positions.astype(np.float64),
            -positions.astype(np.float64),
            rng.standard_normal(n),
        ])
        evr = np.array([0.4, 0.3, 0.1])
        results = compute_correlations(projections, positions, evr)
        assert len(results) == 3
        assert results[0].correlation == pytest.approx(1.0)
        assert results[1].correlation == pytest.approx(-1.0)
        assert abs(results[2].correlation) < 0.3

    def test_result_types(self):
        positions = np.arange(20, dtype=np.int32)
        projections = np.arange(20, dtype=np.float64).reshape(-1, 1)
        evr = np.array([0.5])
        results = compute_correlations(projections, positions, evr)
        r = results[0]
        assert isinstance(r, CorrelationResult)
        assert isinstance(r.pca_dim, int)
        assert isinstance(r.correlation, float)
        assert isinstance(r.p_value, float)
        assert isinstance(r.explained_variance_ratio, float)

    def test_evr_passed_through(self):
        """explained_variance_ratio should be copied from input array."""
        positions = np.arange(20, dtype=np.int32)
        projections = np.column_stack([
            np.arange(20, dtype=np.float64),
            np.arange(20, dtype=np.float64),
        ])
        evr = np.array([0.42, 0.13])
        results = compute_correlations(projections, positions, evr)
        assert results[0].explained_variance_ratio == pytest.approx(0.42)
        assert results[1].explained_variance_ratio == pytest.approx(0.13)
