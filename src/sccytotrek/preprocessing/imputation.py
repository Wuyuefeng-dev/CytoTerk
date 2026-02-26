"""
Custom feature (gene) imputation to address dropout in scRNA-seq.
"""

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

def impute_knn_smoothing(adata: AnnData, n_neighbors: int = 15, use_rep: str = 'X_pca') -> AnnData:
    """
    Impute gene expression by smoothing over a k-Nearest Neighbors graph.
    This provides an original implementation of expression recovery 
    that mitigates standard dropout effects.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Expected to have a lower-dimensional representation
        (e.g., PCA) available to reliably find neighbors.
    n_neighbors : int
        Number of neighbors to average over.
    use_rep : str
        Representation in `adata.obsm` to base distance calculations on.
        Usually 'X_pca' or 'X_scVI'.
        
    Returns
    -------
    AnnData
        A new AnnData object with imputed counts in `X` and original in `raw`.
    """
    print(f"Running custom kNN-smoothing feature imputation (k={n_neighbors})...")
    
    if use_rep not in adata.obsm:
        if 'X_pca' not in adata.obsm:
            print("No PCA found. Running PCA on highly variable genes...")
            # Simple PCA just for neighbor finding
            temp_adata = adata.copy()
            if 'highly_variable' not in temp_adata.var:
                sc.pp.highly_variable_genes(temp_adata)
            sc.tl.pca(temp_adata, svd_solver='arpack')
            adata.obsm['X_pca'] = temp_adata.obsm['X_pca']
        use_rep = 'X_pca'
            
    rep_data = adata.obsm[use_rep]
    
    # 1. Fit kNN
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    nn.fit(rep_data)
    
    # Get distance graph (adjacency matrix)
    print("Constructing neighbor affinity matrix...")
    adj = nn.kneighbors_graph(rep_data, mode='connectivity')
    
    # Add self-connectivity and normalize rows to sum to 1
    # This turns it into an averaging matrix
    adj = adj + sp.eye(adj.shape[0])
    row_sums = np.array(adj.sum(axis=1)).flatten()
    inv_row_sums = 1.0 / row_sums
    D_inv = sp.diags(inv_row_sums)
    
    transition_matrix = D_inv.dot(adj)
    
    # 2. Smooth data
    print("Smoothing expression matrix...")
    adata_imputed = adata.copy()
    
    # Cache raw if not exists to not lose un-imputed data
    if adata_imputed.raw is None:
        adata_imputed.raw = adata_imputed.copy()
        
    if sp.issparse(adata_imputed.X):
        # Sparse matrix multiplication
        adata_imputed.X = transition_matrix.dot(adata_imputed.X)
    else:
        # Dense
        adata_imputed.X = transition_matrix.dot(adata_imputed.X)
        
    print("Feature imputation complete.")
    return adata_imputed
