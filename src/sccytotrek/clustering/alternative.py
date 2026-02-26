"""
Alternative clustering methods (K-Means, Agglomerative, Spectral, GMM, DBSCAN).
"""

import scanpy as sc
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import scanpy as sc

def run_kmeans(adata: AnnData, n_clusters: int = 5, use_rep: str = 'X_pca', random_state: int = 42) -> AnnData:
    """Run K-Means clustering."""
    print(f"Running K-Means (k={n_clusters})...")
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    adata.obs['kmeans'] = pd.Categorical(model.fit_predict(X).astype(str))
    return adata

def run_agglomerative(adata: AnnData, n_clusters: int = 5, use_rep: str = 'X_pca') -> AnnData:
    """Run Agglomerative (Hierarchical) clustering."""
    print(f"Running Agglomerative Clustering (k={n_clusters})...")
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    model = AgglomerativeClustering(n_clusters=n_clusters)
    adata.obs['agglomerative'] = pd.Categorical(model.fit_predict(X).astype(str))
    return adata

def run_spectral(adata: AnnData, n_clusters: int = 5, use_rep: str = 'X_pca', random_state: int = 42) -> AnnData:
    """Run Spectral clustering."""
    print(f"Running Spectral Clustering (k={n_clusters})...")
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    model = SpectralClustering(n_clusters=n_clusters, assign_labels='discretize', random_state=random_state)
    adata.obs['spectral'] = pd.Categorical(model.fit_predict(X).astype(str))
    return adata

def run_gmm(adata: AnnData, n_components: int = 5, use_rep: str = 'X_pca', random_state: int = 42) -> AnnData:
    """Run Gaussian Mixture Model clustering."""
    print(f"Running Gaussian Mixture Model (k={n_components})...")
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    adata.obs['gmm'] = pd.Categorical(model.fit_predict(X).astype(str))
    return adata

def run_dbscan(adata: AnnData, eps: float = 0.5, min_samples: int = 5, use_rep: str = 'X_pca') -> AnnData:
    """Run DBSCAN density-based clustering."""
    print(f"Running DBSCAN (eps={eps}, min_samples={min_samples})...")
    X = adata.obsm[use_rep] if use_rep in adata.obsm else adata.X
    model = DBSCAN(eps=eps, min_samples=min_samples)
    adata.obs['dbscan'] = pd.Categorical(model.fit_predict(X).astype(str))
    return adata

def run_louvain(adata: AnnData, resolution: float = 1.0, random_state: int = 42, **kwargs) -> AnnData:
    """
    Run conventional Louvain clustering via Scanpy.
    """
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.louvain(adata, resolution=resolution, random_state=random_state, **kwargs)
    return adata

def run_leiden(adata: AnnData, resolution: float = 1.0, random_state: int = 42, **kwargs) -> AnnData:
    """
    Run conventional Leiden clustering via Scanpy.
    """
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state, **kwargs)
    return adata
