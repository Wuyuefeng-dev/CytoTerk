"""
Custom implementation for doublet identification reducing dependency on Scrublet.
"""

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import scipy.sparse as sp

def identify_doublets(adata: AnnData, expected_rate: float = 0.05, n_neighbors: int = 30) -> AnnData:
    """
    Identify potential doublets by simulating doublets from data and using a k-NN graph.
    This is an original, dependency-reduced implementation inspired by Scrublet.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Best run on raw, unnormalized counts.
    expected_rate : float
        The estimated doublet rate.
    n_neighbors : int
        Number of nearest neighbors to construct the graph.
        
    Returns
    -------
    AnnData
        Updated AnnData with 'doublet_score' and 'predicted_doublet' in obs.
    """
    print(f"Running custom original doublet detection (Expected Rate: {expected_rate})...")
    
    X = adata.X
    n_obs = X.shape[0]
    
    # 1. Simulate doublets
    n_sim = n_obs * 2
    idx1 = np.random.choice(n_obs, n_sim, replace=True)
    idx2 = np.random.choice(n_obs, n_sim, replace=True)
    
    if sp.issparse(X):
        # A simple additive model for doublets
        X_sim = X[idx1] + X[idx2]
        X_comb = sp.vstack([X, X_sim])
    else:
        X_sim = X[idx1] + X[idx2]
        X_comb = np.vstack([X, X_sim])
        
    # 2. Basic dimensionality reduction (PCA on log-counts)
    if sp.issparse(X_comb):
        X_comb_log = X_comb.copy()
        X_comb_log.data = np.log1p(X_comb_log.data)
    else:
        X_comb_log = np.log1p(X_comb)
        
    try:
        pca = PCA(n_components=30, random_state=42)
        if sp.issparse(X_comb_log):
             X_comb_pca = pca.fit_transform(X_comb_log.toarray())
        else:
             X_comb_pca = pca.fit_transform(X_comb_log)
    except MemoryError:
        print("MemoryError during custom doublet detection. Try downsampling or using highly variable genes first.")
        return adata
        
    X_obs_pca = X_comb_pca[:n_obs]
    
    # 3. k-NN graph and scoring
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_comb_pca)
    distances, indices = nn.kneighbors(X_obs_pca)
    
    # Score: proportion of simulated doublets in the neighborhood
    # simulated doublets have indices >= n_obs
    scores = np.mean(indices >= n_obs, axis=1)
    
    # Adjust scores relative to expected rate (simplified thresholding)
    threshold = np.quantile(scores, 1.0 - expected_rate)
    
    adata.obs['doublet_score'] = scores
    adata.obs['predicted_doublet'] = scores >= threshold
    
    n_doublets = adata.obs['predicted_doublet'].sum()
    print(f"Custom algorithm identified {n_doublets} doublets ({n_doublets/adata.n_obs*100:.1f}%).")
    
    return adata
