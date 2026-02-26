"""
Identify ordering effect genes along a trajectory (pseudotime).
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr

def find_ordering_genes(adata: AnnData, pseudotime_key: str = 'pseudotime', top_n: int = 100) -> pd.DataFrame:
    """
    Identify genes whose expression significantly changes along a pseudotime ordering.
    Uses Spearman rank correlation to find monotonic trends.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    pseudotime_key : str
        Column in adata.obs containing the continuous ordering/pseudotime metric.
    top_n : int
        Number of top genes to return.
        
    Returns
    -------
    pd.DataFrame
        DataFrame of genes correlated with the ordering, sorted by absolute correlation.
    """
    if pseudotime_key not in adata.obs:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs")
        
    print(f"Finding ordering effect genes along '{pseudotime_key}'...")
    
    pt = adata.obs[pseudotime_key].values
    
    # Filter cells with valid pseudotime
    valid_idx = ~np.isnan(pt)
    pt_valid = pt[valid_idx]
    
    use_raw = False
    if adata.raw is not None:
        X = adata.raw.X[valid_idx, :]
        var_names = adata.raw.var_names
        use_raw = True
    else:
        X = adata.X[valid_idx, :]
        var_names = adata.var_names

    corrs = []
    pvals = []
    
    # Simple loop for correlation (can be optimized for large sparse matrices)
    if hasattr(X, "toarray"):
        for i in range(X.shape[1]):
            expr = X[:, i].toarray().flatten()
            corr, pval = spearmanr(pt_valid, expr)
            corrs.append(corr)
            pvals.append(pval)
    else:
        for i in range(X.shape[1]):
            expr = X[:, i]
            corr, pval = spearmanr(pt_valid, expr)
            corrs.append(corr)
            pvals.append(pval)
            
    res_df = pd.DataFrame({
        'gene': var_names,
        'spearman_corr': corrs,
        'pval': pvals
    }).dropna()
    
    res_df['abs_corr'] = res_df['spearman_corr'].abs()
    res_df = res_df.sort_values('abs_corr', ascending=False).head(top_n)
    
    return res_df.reset_index(drop=True)
