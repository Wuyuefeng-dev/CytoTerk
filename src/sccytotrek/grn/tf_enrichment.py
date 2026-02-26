"""
Custom original Gene Regulatory Network inferrence and TF enrichment.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
import scipy.sparse as sp

def run_tf_enrichment(
    adata: AnnData, 
    tf_network: pd.DataFrame, 
    source_col: str, 
    target_col: str, 
    weight_col: str = None,
    min_expr_fraction: float = 0.05
) -> AnnData:
    """
    Perform Transcription Factor (TF) activity inference using a simple 
    weighted sum approach (dot product) to minimize external dependencies like Decoupler.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (log-normalized).
    tf_network : pd.DataFrame
        Dataframe containing the GRN/TF-Target relationships.
    source_col : str
        Column containing TF names.
    target_col : str
        Column containing target gene names.
    weight_col : str
    weight_col : str
        Column containing interaction weights (e.g., +1 for activator, -1 for repressor). 
        If None, assumes all weights are +1.
    min_expr_fraction : float
        Minimum fraction of cells expressing the TF RNA for it to be considered. 
        TFs below this threshold will be excluded from the activity inference.
        
    Returns
    -------
    AnnData
        The AnnData object with TF activities added to `obsm['X_tf_activity']`.
    """
    print("Running custom TF enrichment scoring...")
    
    if weight_col is None:
        tf_network = tf_network.copy()
        tf_network['weight'] = 1.0
        weight_col = 'weight'
        
    # Filter network to genes present in adata
    valid_targets = set(adata.var_names)
    net = tf_network[tf_network[target_col].isin(valid_targets)]
    
    # Get unique TFs
    tfs = net[source_col].unique()
    
    # Filter TFs by actual RNA expression in the dataset
    print(f"Filtering TFs requiring expression in at least {min_expr_fraction*100}% of cells...")
    valid_tfs = []
    
    # Calculate fraction of cells expressing each gene
    if sp.issparse(adata.X):
        expr_frac = np.array((adata.X > 0).mean(axis=0)).flatten()
    else:
        expr_frac = np.mean(adata.X > 0, axis=0)
        
    gene_to_frac = dict(zip(adata.var_names, expr_frac))
    
    for tf in tfs:
        if tf in gene_to_frac and gene_to_frac[tf] >= min_expr_fraction:
            valid_tfs.append(tf)
            
    if not valid_tfs:
        print("Warning: No TFs passed the expression filter. Returning adata unchanged.")
        return adata
        
    print(f"Retained {len(valid_tfs)} out of {len(tfs)} TFs with high RNA expression.")
    tfs = valid_tfs
    
    # Restrict network to valid TFs
    net = net[net[source_col].isin(tfs)]
    
    # Create target x TF weight matrix
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    
    W = np.zeros((adata.n_vars, len(tfs)))
    tf_to_idx = {tf: i for i, tf in enumerate(tfs)}
    
    for _, row in net.iterrows():
        g_idx = gene_to_idx[row[target_col]]
        t_idx = tf_to_idx[row[source_col]]
        W[g_idx, t_idx] = row[weight_col]
        
    # Standardize weights per TF to avoid bias from regulon size
    W_sum = np.abs(W).sum(axis=0)
    W_sum[W_sum == 0] = 1 # Avoid div by zero
    W = W / W_sum
    
    # Multiply Expression matrix by Weight matrix (Cells x Genes) @ (Genes x TFs) -> (Cells x TFs)
    if sp.issparse(adata.X):
        tf_activities = adata.X.dot(W)
    else:
        tf_activities = np.dot(adata.X, W)
        
    # User Request: Enriched TF should have high RNA expression in data
    # We multiply the inferred activity by the min-max scaled RNA expression of the TF itself
    print("Scaling inferred TF activity by actual TF RNA expression...")
    tf_rna_expr = np.zeros((adata.n_obs, len(tfs)))
    for i, tf in enumerate(tfs):
        g_idx = gene_to_idx[tf]
        if sp.issparse(adata.X):
            expr = adata.X[:, g_idx].toarray().flatten()
        else:
            expr = adata.X[:, g_idx]
            
        # Min-Max scale the RNA expression between 0 and 1
        expr_min = expr.min()
        expr_max = expr.max()
        if expr_max > expr_min:
            scaled_expr = (expr - expr_min) / (expr_max - expr_min)
        else:
            scaled_expr = np.zeros_like(expr)
            
        tf_rna_expr[:, i] = scaled_expr
        
    # Element-wise multiplication: Inferred Activity * Scaled RNA Expression
    tf_activities = tf_activities * tf_rna_expr
        
    # Store in obsm and obs
    tf_df = pd.DataFrame(tf_activities, index=adata.obs_names, columns=tfs)
    adata.obsm['X_tf_activity'] = tf_df
    
    for tf in tfs:
        adata.obs[f'tf_score_{tf}'] = tf_df[tf].values
    
    print(f"Scored {len(tfs)} TFs with RNA-expression adjustment. Stored in adata.obsm['X_tf_activity'].")
    return adata

def plot_tf_dotplot(adata: AnnData, tfs: list = None, groupby: str = 'leiden_0.5', save_path: str = None) -> None:
    """
    Plot Transcription Factor (TF) enrichment scores across groups using a dotplot.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing TF activities in `obsm['X_tf_activity']`.
    tfs : list, optional
        List of TFs to plot. If None, plots all available TFs.
    groupby : str, optional
        Key in `adata.obs` to group cells by (e.g., cell types or clusters).
    save_path : str, optional
        Path to save the generated figure.
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    if 'X_tf_activity' not in adata.obsm:
        print("TF activity not found. Please run `run_tf_enrichment` first.")
        return
        
    tf_df = adata.obsm['X_tf_activity']
    
    if tfs is None:
        tfs = tf_df.columns.tolist()
    else:
        tfs = [tf for tf in tfs if tf in tf_df.columns]
        
    if not tfs:
        print("No valid TFs found to plot.")
        return
        
    # Create a temporary AnnData object specifically for dotplot plotting
    # We set X to the TF activity matrix so that sc.pl.dotplot works naturally
    tmp_adata = AnnData(X=tf_df[tfs].values, obs=adata.obs.copy())
    tmp_adata.var_names = tfs
    
    # Generate the dotplot using standard scanpy scaling parameters
    sc.pl.dotplot(tmp_adata, var_names=tfs, groupby=groupby, cmap='viridis', standard_scale='var', show=False, title="TF Enrichment Dotplot")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

