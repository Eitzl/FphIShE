"""Dimensionality reduction of embeddings (PCA)."""

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X, n_components=128):
    return PCA(n_components=n_components).fit(X)


def transform(pca, X):
    return pca.transform(X).astype(np.float32)
