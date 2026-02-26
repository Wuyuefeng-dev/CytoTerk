"""
Pseudobulk and aggregation methods.
"""

from anndata import AnnData
import pandas as pd
import numpy as np
import scipy.sparse as sp

def make_pseudobulk(adata: AnnData, groupby: str, mode: str = "sum") -> AnnData:
    """
    Aggregate single-cell profiles into pseudobulk profiles based on a grouping variable.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in `adata.obs` to group cells by (e.g., 'leiden_0.5' or 'patient_id').
    mode : str
        Aggregation mode: 'sum' or 'mean'.
        
    Returns
    -------
    AnnData
        A new AnnData object where rows are the unique groups in `groupby`.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(f"Column '{groupby}' not found in adata.obs")
        
    groups = adata.obs[groupby].astype("category").cat.categories
    
    # For efficiency with sparse matrices
    X = adata.X
    grouped_X = []
    
    for group in groups:
        idx = (adata.obs[groupby] == group).values
        if not np.any(idx):
            continue
            
        group_data = X[idx, :]
        
        if mode == "sum":
            res = group_data.sum(axis=0)
        elif mode == "mean":
            res = group_data.mean(axis=0)
        else:
            raise ValueError("Mode must be 'sum' or 'mean'")
            
        # Handle sparse sum returns (matrix -> array)
        if hasattr(res, 'A1'):
            res = res.A1
        elif hasattr(res, 'flatten'):
            res = res.flatten()
            
        grouped_X.append(res)
        
    pb_X = np.vstack(grouped_X)
    
    # Create new AnnData
    pb_adata = AnnData(X=pb_X, var=adata.var.copy())
    pb_adata.obs_names = [str(g) for g in groups]
    pb_adata.obs[groupby] = groups
    
    print(f"Created pseudobulk AnnData with {pb_adata.n_obs} {groupby} groups across {pb_adata.n_vars} genes.")
    return pb_adata
