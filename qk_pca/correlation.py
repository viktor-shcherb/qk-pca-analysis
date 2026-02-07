"""Pearson correlation between PCA projections and token positions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr


@dataclass
class CorrelationResult:
    """Correlation results for one PCA dimension."""

    pca_dim: int
    correlation: float
    p_value: float
    explained_variance_ratio: float


def compute_correlations(
    projections: np.ndarray,
    positions: np.ndarray,
    explained_variance_ratio: np.ndarray,
) -> list[CorrelationResult]:
    """Compute Pearson correlation between each PCA dimension and position.

    Parameters
    ----------
    projections : ndarray of shape (N, n_components)
    positions : ndarray of shape (N,)
    explained_variance_ratio : ndarray of shape (n_components,)

    Returns
    -------
    List of ``CorrelationResult``, one per PCA dimension.
    """
    positions_f = positions.astype(np.float64)
    results = []
    for dim in range(projections.shape[1]):
        proj = projections[:, dim].astype(np.float64)
        # pearsonr requires variance in both inputs
        if np.std(proj) == 0 or np.std(positions_f) == 0:
            corr, pval = 0.0, 1.0
        else:
            corr, pval = pearsonr(proj, positions_f)
        results.append(
            CorrelationResult(
                pca_dim=dim,
                correlation=float(corr),
                p_value=float(pval),
                explained_variance_ratio=float(explained_variance_ratio[dim]),
            )
        )
    return results
