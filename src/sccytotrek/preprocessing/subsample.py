"""
Module for cell subsampling in large datasets.
"""

from anndata import AnnData
import scanpy as sc

def subsample_cells(adata: AnnData, target_cells: int = 10000, random_state: int = 42) -> AnnData:
    """
    Subsample cells from a large AnnData object to reduce computational burden.
    If the current number of cells is less than the target, returns the original object.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    target_cells : int
        Maximum number of cells to retain.
    random_state : int
        Random seed for reproducibility.
        
    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    if adata.n_obs <= target_cells:
        print(f"Data has {adata.n_obs} cells (<= target {target_cells}). No subsampling needed.")
        return adata
        
    print(f"Subsampling {adata.n_obs} cells down to {target_cells}...")
    
    # We copy to avoid modifying the original view destructively if it's passed around, 
    # though scanpy subsample usually operates in place or returns a view.
    adata_sub = sc.pp.subsample(adata, n_obs=target_cells, random_state=random_state, copy=True)
    return adata_sub

def subsample_by_group(adata: AnnData, groupby: str, target_cells_per_group: int = 500, random_state: int = 42) -> AnnData:
    """
    Subsample cells to have a maximum number of cells per group to balance representations.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Key in `adata.obs` delineating groups.
    target_cells_per_group : int
        Max cells per group.
    random_state : int
        Random seed.
        
    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    if groupby not in adata.obs:
        raise ValueError(f"'{groupby}' not in adata.obs")
        
    print(f"Subsampling to max {target_cells_per_group} cells per group in '{groupby}'...")
    
    # scanpy doesn't have a direct stratified subsample wrapper, so we manually subset
    obs = adata.obs.copy()
    
    keep_indices = []
    
    import numpy as np
    np.random.seed(random_state)
    
    for group in obs[groupby].unique():
        group_idx = np.where(obs[groupby] == group)[0]
        if len(group_idx) > target_cells_per_group:
            sampled_idx = np.random.choice(group_idx, size=target_cells_per_group, replace=False)
            keep_indices.extend(sampled_idx)
        else:
            keep_indices.extend(group_idx)
            
    adata_sub = adata[keep_indices].copy()
    print(f"Subsampled from {adata.n_obs} to {adata_sub.n_obs} cells.")
    
    return adata_sub
