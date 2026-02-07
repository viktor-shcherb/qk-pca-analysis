"""Tests for qk_pca.pca."""

from __future__ import annotations

import numpy as np
import pytest

from qk_pca.pca import PCAResult, fit_pca


class TestFitPCASklearn:
    def test_output_shapes(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        n, dim = vectors.shape
        n_components = 5
        result = fit_pca(vectors, n_components=n_components, backend="sklearn")
        assert isinstance(result, PCAResult)
        assert result.components.shape == (n_components, dim)
        assert result.explained_variance_ratio.shape == (n_components,)
        assert result.mean.shape == (dim,)
        assert result.projections.shape == (n, n_components)

    def test_variance_ratios_sum_leq_1(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="sklearn")
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-6

    def test_variance_ratios_descending(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="sklearn")
        ratios = result.explained_variance_ratio
        assert np.all(ratios[:-1] >= ratios[1:])

    def test_components_orthogonal(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="sklearn")
        gram = result.components @ result.components.T
        np.testing.assert_allclose(gram, np.eye(5), atol=1e-6)

    def test_mean_subtracted(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="sklearn")
        np.testing.assert_allclose(result.mean, vectors.mean(axis=0), atol=1e-5)

    def test_n_components_clamped_to_min_dim(self, rng):
        """When n_components > min(N, dim), it should be clamped."""
        vectors = rng.standard_normal((10, 5)).astype(np.float32)
        result = fit_pca(vectors, n_components=100, backend="sklearn")
        assert result.components.shape[0] == 5  # min(10, 5)
        assert result.projections.shape[1] == 5

    def test_n_components_clamped_to_n_samples(self, rng):
        vectors = rng.standard_normal((3, 64)).astype(np.float32)
        result = fit_pca(vectors, n_components=10, backend="sklearn")
        assert result.components.shape[0] == 3
        assert result.projections.shape[1] == 3

    def test_single_component(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=1, backend="sklearn")
        assert result.projections.shape[1] == 1
        assert result.components.shape[0] == 1

    def test_projections_reconstruct(self, synthetic_vectors):
        """projections @ components + mean should approximate original vectors."""
        vectors, _ = synthetic_vectors
        n_components = vectors.shape[1]  # full rank
        result = fit_pca(vectors, n_components=n_components, backend="sklearn")
        reconstructed = result.projections @ result.components + result.mean
        np.testing.assert_allclose(reconstructed, vectors, atol=1e-4)


class TestFitPCATorch:
    def test_output_shapes(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        n, dim = vectors.shape
        n_components = 5
        result = fit_pca(vectors, n_components=n_components, backend="torch", device="cpu")
        assert result.components.shape == (n_components, dim)
        assert result.explained_variance_ratio.shape == (n_components,)
        assert result.mean.shape == (dim,)
        assert result.projections.shape == (n, n_components)

    def test_variance_ratios_sum_leq_1(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="torch", device="cpu")
        assert result.explained_variance_ratio.sum() <= 1.0 + 1e-6

    def test_variance_ratios_descending(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=5, backend="torch", device="cpu")
        ratios = result.explained_variance_ratio
        assert np.all(ratios[:-1] >= ratios[1:])

    def test_n_components_clamped(self, rng):
        vectors = rng.standard_normal((10, 5)).astype(np.float32)
        result = fit_pca(vectors, n_components=100, backend="torch", device="cpu")
        assert result.components.shape[0] == 5

    def test_results_numpy(self, synthetic_vectors):
        """All outputs should be numpy arrays, not torch tensors."""
        vectors, _ = synthetic_vectors
        result = fit_pca(vectors, n_components=3, backend="torch", device="cpu")
        assert isinstance(result.components, np.ndarray)
        assert isinstance(result.explained_variance_ratio, np.ndarray)
        assert isinstance(result.mean, np.ndarray)
        assert isinstance(result.projections, np.ndarray)


class TestBackendAgreement:
    """sklearn and torch backends should produce equivalent results."""

    def test_explained_variance_ratios_close(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        sk = fit_pca(vectors, n_components=5, backend="sklearn")
        th = fit_pca(vectors, n_components=5, backend="torch", device="cpu")
        np.testing.assert_allclose(
            sk.explained_variance_ratio, th.explained_variance_ratio, atol=1e-5
        )

    def test_mean_identical(self, synthetic_vectors):
        vectors, _ = synthetic_vectors
        sk = fit_pca(vectors, n_components=5, backend="sklearn")
        th = fit_pca(vectors, n_components=5, backend="torch", device="cpu")
        np.testing.assert_allclose(sk.mean, th.mean, atol=1e-5)

    def test_projection_magnitudes_close(self, synthetic_vectors):
        """Projections may differ by sign, but magnitudes should match."""
        vectors, _ = synthetic_vectors
        sk = fit_pca(vectors, n_components=5, backend="sklearn")
        th = fit_pca(vectors, n_components=5, backend="torch", device="cpu")
        # Compare absolute values column-wise
        for i in range(5):
            np.testing.assert_allclose(
                np.abs(sk.projections[:, i]),
                np.abs(th.projections[:, i]),
                atol=1e-4,
            )
