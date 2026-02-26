"""
Differential expression analysis adjusting for imperfect capture efficiency (dropouts).
"""

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.stats as stats
import statsmodels.api as sm

def dropout_adjusted_de(
    adata: AnnData, 
    group_key: str, 
    group1: str, 
    group2: str,
    min_cells: int = 10,
    out_csv: str = None
) -> pd.DataFrame:
    """
    Perform differential expression analysis while adjusting for imperfect capture 
    efficiency by incorporating the cellular detection rate (CDR - proxy for dropout rate) 
    as a covariate in a linear model (similar to the MAST framework).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized).
    group_key : str
        Column in obs defining groups.
    group1 : str
        Name of the first group.
    group2 : str
        Name of the second group.
    min_cells : int
        Minimum number of cells expressing a gene in either group to test it.
    out_csv : str
        Optional filepath to save the results in a friendly CSV layout.
        
    Returns
    -------
    pd.DataFrame
        Differential expression results including log2 fold changes and p-values.
    """
    print(f"Running dropout-adjusted DE ({group1} vs {group2}) on {group_key}...")
    
    if group_key not in adata.obs:
        raise ValueError(f"'{group_key}' not in adata.obs")
        
    # Calculate Cellular Detection Rate (CDR) as a proxy for capture efficiency/dropdown
    # Proportion of genes expressed > 0 per cell
    if 'cdr' not in adata.obs:
        if hasattr(adata.X, "toarray"):
            adata.obs['cdr'] = (adata.X > 0).sum(axis=1) / adata.n_vars
        else:
            adata.obs['cdr'] = (adata.X > 0).sum(axis=1) / adata.n_vars
    
    idx_g1 = adata.obs[group_key] == group1
    idx_g2 = adata.obs[group_key] == group2
    
    if not any(idx_g1) or not any(idx_g2):
        raise ValueError("One or both groups are empty.")
        
    idx_both = idx_g1 | idx_g2
    X_sub = adata.X[idx_both, :]
    
    # Endogenous grouping variable (1 for group1, 0 for group2)
    group_var = (adata.obs.loc[idx_both, group_key] == group1).astype(int).values
    cdr_var = adata.obs.loc[idx_both, 'cdr'].values
    
    # Design matrix: [Intercept, Group, CDR]
    exog = sm.add_constant(np.column_stack((group_var, cdr_var)))
    
    results = []
    var_names = adata.var_names
    
    if hasattr(X_sub, "toarray"):
        is_sparse = True
    else:
        is_sparse = False
        
    for i in range(X_sub.shape[1]):
        if is_sparse:
            y = X_sub[:, i].toarray().flatten()
        else:
            y = X_sub[:, i]
            
        # Basic filtering
        if np.sum(y > 0) < min_cells:
            continue
            
        # Fit OLS
        try:
            model = sm.OLS(y, exog)
            fit = model.fit()
            
            # The coefficient for Group is index 1
            coef = fit.params[1]
            pval = fit.pvalues[1]
            
            # Simple Log2FC (assuming y is already log1p)
            lfc = np.mean(y[group_var == 1]) - np.mean(y[group_var == 0])
            
            results.append({
                'gene': var_names[i],
                'log2fc': lfc / np.log(2), # Adjust if data is ln(1+x)
                'coef_adjusted': coef,
                'pval': pval
            })
        except:
            pass
            
    df_res = pd.DataFrame(results)
    
    # FDR correction
    if not df_res.empty:
        df_res['padj'] = stats.false_discovery_control(df_res['pval'])
        df_res = df_res.sort_values('padj')
        
    if out_csv and not df_res.empty:
        df_res.to_csv(out_csv, index=False)
        print(f"Saved differential expression results to {out_csv}")
        
    return df_res
