"""
Multiple sample integration wrappers.
"""

from anndata import AnnData
import scanpy as sc

def run_harmony(adata: AnnData, batch_key: str, max_iter_harmony: int = 10, **kwargs) -> None:
    """
    Wrapper around sc.external.pp.harmony_integrate to integrate across multiple samples/batches.
    Requires `harmonypy` to be installed.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Must have PCA computed (`X_pca` in obsm).
    batch_key : str
        The key in `adata.obs` corresponding to the batch/sample assignment.
    max_iter_harmony : int
        Maximum number of iterations for harmony.
    """
    if 'X_pca' not in adata.obsm:
        raise ValueError("PCA must be computed before running Harmony. Use `sccytotrek.tools.run_pca_and_neighbors`.")
        
    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs.")
        
    # Run integration directly using scanpy's external API
    sc.external.pp.harmony_integrate(
        adata, 
        batch_key, 
        max_iter_harmony=max_iter_harmony,
        **kwargs
    )
    
    # Store the result in obsm and compute new neighbors based on harmony
    # Harmony result is stored in `X_pca_harmony` by scanpy
    if 'X_pca_harmony' in adata.obsm:
        sc.pp.neighbors(adata, use_rep='X_pca_harmony')
        sc.tl.umap(adata)
