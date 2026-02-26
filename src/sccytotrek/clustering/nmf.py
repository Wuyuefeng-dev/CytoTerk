"""
NMF (Non-negative Matrix Factorization) clustering and program discovery.
"""

import scanpy as sc
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import scipy # Added for scipy.sparse.issparse

def run_nmf(adata: AnnData, n_components: int = 10, random_state: int = 42) -> AnnData:
    """
    Run Non-negative Matrix Factorization (NMF) on the data.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Ideally normalized, log-transformed, and subset to HVGs.
    n_components : int
        Number of NMF factors to compute.
    random_state : int
        Random seed for stability.
        
    Returns
    -------
    AnnData
        Updated AnnData object with 'X_nmf' in obsm and 'nmf_features' in varm.
    """
    print(f"Running NMF with {n_components} components...")
    
    # NMF requires non-negative data. If data is centered or scaled with negative values, 
    # we need to fallback to the raw positive counts/logcounts. Note that standard 
    # sc.pp.log1p produces non-negative values.
    # We will assume adata.X is non-negative.
    
    X = adata.X
    if scipy.sparse.issparse(X):
        # NMF in sklearn technically works on sparse matrices, but we need to ensure fast computation.
        pass
    
    model = NMF(n_components=n_components, init='nndsvda', random_state=random_state, max_iter=500)
    
    W = model.fit_transform(X) # Cell loadings
    H = model.components_      # Gene loadings
    
    adata.obsm['X_nmf'] = W
    adata.varm['nmf_features'] = H.T # Changed from nmf_loadings to nmf_features as per docstring
    
    # Assign soft-cluster based on max NMF loading
    adata.obs['nmf_cluster'] = [f"NMF_{i}" for i in W.argmax(axis=1)]
    
    return adata

def extract_nmf_markers(adata: AnnData, top_n: int = 50) -> pd.DataFrame:
    """
    Extract the top 'weighted' feature genes for each NMF cluster.
    This fulfills the goal of simultaneously identifying cell groups and 
    a weighted list of feature genes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing 'nmf_features' in varm.
    top_n : int
        Number of top genes to retrieve per NMF component.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of top genes and their weights for each NMF component.
    """
    if 'nmf_features' not in adata.varm:
        raise ValueError("NMF features not found in adata.varm. Run run_nmf first.")
        
    H_T = adata.varm['nmf_features'] # Genes x Components
    genes = adata.var_names
    n_components = H_T.shape[1]
    
    results = {}
    for i in range(n_components):
        weights = H_T[:, i]
        # Get indices of top N weights
        top_indices = np.argsort(weights)[::-1][:top_n]
        
        # Store gene names and weights
        results[f'NMF_{i}_gene'] = genes[top_indices].values
        results[f'NMF_{i}_weight'] = weights[top_indices]
        
    return pd.DataFrame(results)

def get_nmf_markers(adata: AnnData, top_n: int = 10) -> dict:
    """
    Get top driving genes for each NMF component.
    """
    H = adata.varm['nmf_loadings'].T # shape (n_components, n_genes)
    markers = {}
    for i in range(H.shape[0]):
        top_indices = H[i, :].argsort()[::-1][:top_n]
        markers[f"NMF_{i}"] = adata.var_names[top_indices].tolist()
        
    return markers
