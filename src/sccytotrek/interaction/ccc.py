"""
Cell-Cell Communication (CCC) analysis based on Ligand-Receptor interactions.
"""

from anndata import AnnData
import pandas as pd
import numpy as np
from scipy import sparse

def run_cellphonedb_scoring(
    adata: AnnData, 
    lr_pairs: pd.DataFrame, 
    group_key: str = 'leiden_0.5',
    ligand_col: str = 'ligand',
    receptor_col: str = 'receptor',
    n_perms: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Score cell-cell interactions based on the CellPhoneDB algorithm.
    This computes the mean expression of the ligand in the sender and the 
    receptor in the receiver, then performs a permutation test (shuffling cell 
    labels) to calculate an empirical p-value for the specificity of the interaction.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    lr_pairs : pd.DataFrame
        DataFrame containing at least two columns for ligand and receptor gene symbols.
    group_key : str
        The grouping variable in `adata.obs` (e.g., cell types, clusters).
    ligand_col : str
        Column name in lr_pairs for ligands.
    receptor_col : str
        Column name in lr_pairs for receptors.
    n_perms : int
        Number of permutations for empirical p-value calculation.
    random_state : int
        Random seed.
        
    Returns
    -------
    pd.DataFrame
        Interaction scores and empirical p-values between all pairs of groups for the given LR pairs.
    """
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not in adata.obs")
        
    print(f"Running CellPhoneDB algorithm ({n_perms} perms) for {len(lr_pairs)} LR pairs across groups in '{group_key}'...")
    np.random.seed(random_state)
    
    groups = np.array(adata.obs[group_key].unique())
    n_groups = len(groups)
    
    genes_in_data = set(adata.var_names)
    valid_pairs = []
    for _, row in lr_pairs.iterrows():
        if row[ligand_col] in genes_in_data and row[receptor_col] in genes_in_data:
            valid_pairs.append((row[ligand_col], row[receptor_col]))
            
    if not valid_pairs:
        return pd.DataFrame()
        
    ligands = list(set([p[0] for p in valid_pairs]))
    receptors = list(set([p[1] for p in valid_pairs]))
    unique_genes = list(set(ligands + receptors))
    
    # Pre-extract data for speed
    X_genes = adata[:, unique_genes].X
    if sparse.issparse(X_genes):
        X_genes = X_genes.toarray()
        
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}
    
    # Map cells to groups
    labels_true = adata.obs[group_key].values
    
    # Calculate true means
    mean_expr = np.zeros((n_groups, len(unique_genes)))
    for i, g in enumerate(groups):
        idx = labels_true == g
        if np.sum(idx) > 0:
            mean_expr[i, :] = np.mean(X_genes[idx, :], axis=0)
            
    # Calculate true scores (mean of means) for each LR pair and group-pair
    true_scores = {} # (ligand, receptor, sender, receiver) -> score
    for lig, rec in valid_pairs:
        l_idx = gene_to_idx[lig]
        r_idx = gene_to_idx[rec]
        for i, s in enumerate(groups):
            for j, r in enumerate(groups):
                score = (mean_expr[i, l_idx] + mean_expr[j, r_idx]) / 2.0
                true_scores[(lig, rec, s, r)] = score
                
    # Permutation test
    print("Running permutations...")
    perm_counts = {k: 0 for k in true_scores.keys()}
    
    for p in range(n_perms):
        # Shuffle labels
        labels_perm = np.random.permutation(labels_true)
        mean_expr_perm = np.zeros((n_groups, len(unique_genes)))
        
        for i, g in enumerate(groups):
            idx = labels_perm == g
            if np.sum(idx) > 0:
                mean_expr_perm[i, :] = np.mean(X_genes[idx, :], axis=0)
                
        for lig, rec in valid_pairs:
            l_idx = gene_to_idx[lig]
            r_idx = gene_to_idx[rec]
            for i, s in enumerate(groups):
                for j, r in enumerate(groups):
                    perm_score = (mean_expr_perm[i, l_idx] + mean_expr_perm[j, r_idx]) / 2.0
                    if perm_score >= true_scores[(lig, rec, s, r)]:
                        perm_counts[(lig, rec, s, r)] += 1
                        
    # Compile results
    results = []
    for k, score in true_scores.items():
        if score > 0:
            lig, rec, s, r = k
            p_val = perm_counts[k] / n_perms
            results.append({
                'sender': s,
                'receiver': r,
                'interaction': f"{lig}_{rec}",
                'ligand': lig,
                'receptor': rec,
                'score': score,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
            
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(['p_value', 'score'], ascending=[True, False]).reset_index(drop=True)
        
    return df_results

def plot_cell2cell_dotplot(
    df_res: pd.DataFrame, 
    top_n: int = 20, 
    title: str = "Cell2Cell Interaction Dotplot",
    save_path: str = None
):
    """
    Generate a Cell2Cell/CellphoneDB style dotplot of ligand-receptor interactions.
    X-axis: Sender-Receiver cell pairs
    Y-axis: Ligand-Receptor pairs
    Dot size: -log10(p-value) or significance
    Dot color: Interaction Score
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    
    if df_res.empty:
        print("Empty DataFrame provided. Nothing to plot.")
        return
        
    # Filter top interactions
    df_plot = df_res.head(top_n).copy()
    
    # Create required columns
    df_plot['cell_pair'] = df_plot['sender'] + " -> " + df_plot['receiver']
    df_plot['lr_pair'] = df_plot['ligand'] + " - " + df_plot['receptor']
    
    # Calculate log p-value for dot size (-log10(p) with cap)
    # Avoid log(0)
    min_p_val = df_plot.loc[df_plot['p_value'] > 0, 'p_value'].min() if (df_plot['p_value'] > 0).any() else 0.001
    min_p_val = min(0.001, min_p_val)
    
    df_plot['adj_p_value'] = df_plot['p_value'].replace(0, min_p_val)
    df_plot['log_p'] = -np.log10(df_plot['adj_p_value'])
    
    # Pivot tables for grid plotting
    score_pivot = df_plot.pivot(index='lr_pair', columns='cell_pair', values='score').fillna(0)
    logp_pivot = df_plot.pivot(index='lr_pair', columns='cell_pair', values='log_p').fillna(0)
    
    # Make plot
    fig, ax = plt.subplots(figsize=(max(6, len(score_pivot.columns) * 0.8), max(6, len(score_pivot.index) * 0.5)))
    
    # Grid coordinates
    X, Y = np.meshgrid(np.arange(score_pivot.shape[1]), np.arange(score_pivot.shape[0]))
    
    # Flatten for scattered plot
    x_flat = X.flatten()
    y_flat = Y.flatten()
    c_flat = score_pivot.values.flatten()
    s_flat = logp_pivot.values.flatten()
    
    # Scale dot sizes
    s_max = max(1, s_flat.max())
    s_scaled = (s_flat / s_max) * 200  # max size 200
    
    scatter = ax.scatter(x_flat, y_flat, c=c_flat, s=s_scaled, cmap='viridis', edgecolors='grey', linewidths=0.5)
    
    # Formatting
    ax.set_xticks(np.arange(score_pivot.shape[1]))
    ax.set_yticks(np.arange(score_pivot.shape[0]))
    ax.set_xticklabels(score_pivot.columns, rotation=90)
    ax.set_yticklabels(score_pivot.index)
    
    plt.colorbar(scatter, ax=ax, label='Interaction Score')
    
    # Add Size Legend
    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, num=5)
    ax.legend(handles, labels, loc="upper right", title="Significance Size", bbox_to_anchor=(1.3, 1))
    
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
    plt.close(fig)
    return fig
