"""
Confounder correction to facilitate unbiased downstream analysis.
"""

from anndata import AnnData
import scanpy as sc

def regress_confounders(adata: AnnData, keys: list) -> AnnData:
    """
    Remove the effect of confounding factors (e.g., library size, cell cycle phase, mitochondrial fraction)
    to unravel true cell population heterogeneity.
    
    This wraps Scanpy's regress_out function optimally for the CytoTrek pipeline.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Best run on log-normalized data.
    keys : list of str
        Keys in `adata.obs` representing the confounding factors to regress out.
        
    Returns
    -------
    AnnData
        Updated AnnData with effects regressed out.
    """
    print(f"Regressing out confounding factors: {keys}...")
    
    # Verify keys exist
    valid_keys = [k for k in keys if k in adata.obs.columns]
    
    if len(valid_keys) == 0:
        print("Warning: None of the specified confounders exist in adata.obs. Skipping regression.")
        return adata
        
    if len(valid_keys) < len(keys):
        print(f"Warning: Only {valid_keys} found in adata.obs.")
        
    sc.pp.regress_out(adata, valid_keys)
    print("Confounder regression complete. Data matrix updated.")
    
    return adata
