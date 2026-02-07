"""PCA fitting with sklearn and torch backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    """Result of fitting PCA on a set of vectors."""

    components: np.ndarray  # (n_components, head_dim)
    explained_variance_ratio: np.ndarray  # (n_components,)
    mean: np.ndarray  # (head_dim,)
    projections: np.ndarray  # (N, n_components)


def fit_pca(
    vectors: np.ndarray,
    n_components: int = 10,
    backend: str = "sklearn",
    device: str = "cpu",
) -> PCAResult:
    """Fit PCA and project vectors.

    Parameters
    ----------
    vectors : ndarray of shape (N, dim)
    n_components : number of principal components
    backend : ``"sklearn"`` (default) or ``"torch"``
    device : torch device string, only used when backend is ``"torch"``
    """
    n_components = min(n_components, vectors.shape[0], vectors.shape[1])

    if backend == "torch":
        return _fit_torch(vectors, n_components, device)
    return _fit_sklearn(vectors, n_components)


def _fit_sklearn(vectors: np.ndarray, n_components: int) -> PCAResult:
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(vectors)
    return PCAResult(
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        mean=pca.mean_,
        projections=projections,
    )


def _fit_torch(vectors: np.ndarray, n_components: int, device: str) -> PCAResult:
    dev = torch.device(device)
    X = torch.from_numpy(vectors).to(dev)
    mean = X.mean(dim=0)
    X_centered = X - mean

    # Full SVD (exact, like sklearn)
    U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:n_components].cpu().numpy()
    # Explained variance ratio
    explained_var = (S[:n_components] ** 2) / (X_centered.shape[0] - 1)
    total_var = (S**2).sum() / (X_centered.shape[0] - 1)
    explained_variance_ratio = (explained_var / total_var).cpu().numpy()

    projections = (X_centered @ Vt[:n_components].T).cpu().numpy()
    mean_np = mean.cpu().numpy()

    return PCAResult(
        components=components,
        explained_variance_ratio=explained_variance_ratio,
        mean=mean_np,
        projections=projections,
    )
