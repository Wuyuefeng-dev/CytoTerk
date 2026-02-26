"""
In-silico gene perturbation simulation.
"""

from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def predict_knockdown_effect(
    adata: AnnData, 
    target_gene: str, 
    n_hvg: int = 500,
    knockdown_fraction: float = 0.0
) -> np.ndarray:
    """
    Predict the shift in cellular state following an in-silico gene knockdown.
    Operates by finding highly variable genes (HVGs) co-expressed with the target,
    and propagating the effect.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Should contain normalized/scaled data.
    target_gene : str
        The gene to effectively "knockdown".
    n_hvg : int
        Number of highly variable genes to use for the state shift vector.
    knockdown_fraction : float
        The fraction of expression remaining (0.0 = complete knockout).
        
    Returns
    -------
    np.ndarray
        An array of shape (n_cells, n_hvg) representing the predicted shift vector in PCA/HVG space.
    """
    if target_gene not in adata.var_names:
        raise ValueError(f"Target gene '{target_gene}' not found in the dataset.")
        
    # Get HVGs or take top variance genes
    if 'highly_variable' in adata.var.columns:
        hvg_names = adata.var_names[adata.var['highly_variable']][:n_hvg]
    else:
        variances = np.var(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, axis=0)
        top_hvg_idx = np.argsort(variances)[::-1][:n_hvg]
        hvg_names = adata.var_names[top_hvg_idx]
        
    # Calculate simple co-expression (correlation) between target gene and HVGs
    target_expr = adata[:, target_gene].X
    if hasattr(target_expr, "toarray"):
        target_expr = target_expr.toarray()
    target_expr = target_expr.flatten()
    
    hvg_expr = adata[:, hvg_names].X
    if hasattr(hvg_expr, "toarray"):
        hvg_expr = hvg_expr.toarray()
        
    # Correlation matrix between target and HVGs
    correlations = np.corrcoef(target_expr, hvg_expr, rowvar=False)[0, 1:]
    
    # Calculate the shift per cell
    # If a cell expressed the target gene at value X, hitting it to 0 means a change of -X.
    # We propagate this change to co-expressed genes. 
    # If Gene A is positively correlated with Target, a knockdown reduces Gene A.
    
    # change_in_target shape: (n_cells, 1)
    current_expr = target_expr.reshape(-1, 1)
    change_in_target = (current_expr * knockdown_fraction) - current_expr 
    
    # Shift applied to all HVGs based on correlation: shape (n_cells, n_hvg)
    shift_vector = change_in_target @ correlations.reshape(1, -1)
    
    # Store resulting vectors in adata
    adata.obsm[f'shift_{target_gene}_KD'] = shift_vector
    
    return shift_vector
