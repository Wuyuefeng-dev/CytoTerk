"""
Basic scRNA-seq tools including PCA, UMAP, Leiden, and cell state tracking.
"""

import scanpy as sc
from anndata import AnnData
import numpy as np
import pandas as pd

def run_pca_and_neighbors(adata: AnnData, n_pcs: int = 50, n_neighbors: int = 30, random_state: int = 42) -> AnnData:
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs, random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=random_state)
    return adata

def run_umap_and_cluster(adata: AnnData, resolution: float = 0.5, random_state: int = 42) -> AnnData:
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_{resolution}', random_state=random_state)
    return adata

def link_rna_state_to_barcode(adata: AnnData, barcode_key: str = "barcode", state_key: str = "leiden_0.5") -> pd.DataFrame:
    """
    Integrates cell RNA states (e.g., clusters, pseudotime) with real lineage barcodes.
    Calculates the frequency of cell states within each clonal lineage.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    barcode_key : str
        Column containing lineage tracking barcodes.
    state_key : str
        Column containing RNA cell state annotations (e.g., cluster IDs).
        
    Returns
    -------
    pd.DataFrame
        A dataframe mapping barcodes to their constituent cell RNA states.
    """
    if barcode_key not in adata.obs or state_key not in adata.obs:
        raise ValueError(f"Need '{barcode_key}' and '{state_key}' in adata.obs.")
        
    # Group by barcode and cell state to get the clone size per state
    mapping = adata.obs.groupby([barcode_key, state_key]).size().unstack(fill_value=0)
    
    # Calculate proportions
    proportions = mapping.div(mapping.sum(axis=1), axis=0)
    
    print(f"Mapped {len(mapping)} distinct barcodes to {mapping.shape[1]} RNA states.")
    
    # Store in unstructured data
    adata.uns['barcode_state_proportions'] = proportions
    
    return proportions
