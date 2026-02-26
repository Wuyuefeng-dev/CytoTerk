"""
Trajectory analysis and tipping point detection.
"""

from anndata import AnnData
import numpy as np
import scipy.sparse as sp
from scipy.stats import entropy
import pandas as pd

def compute_sandpile_entropy(
    adata: AnnData, 
    pseudotime_key: str = "dpt_pseudotime", 
    n_bins: int = 50,
    correlation_threshold: float = 0.5
) -> dict:
    """
    Compute tipping point cells using a Sandpile Model approach.
    In this model, a cell approaching a tipping point (critical state) 
    exhibits high network entropy and instability before crashing into another state.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing pseudotime.
    pseudotime_key : str
        Key in `adata.obs` where pseudotime is stored.
    n_bins : int
        Number of bins to divide the pseudotime trajectory into.
    correlation_threshold : float
        Threshold for defining an active "edge" or connection between genes 
        to compute state entropy.
        
    Returns
    -------
    dict
        Dictionary containing 'bins', 'entropy', and 'tipping_point_bin'.
        Also updates adata.obs['sandpile_entropy'].
    """
    print("Calculating Sandpile Model (Network Entropy) for Tipping Points...")
    if pseudotime_key not in adata.obs:
        raise ValueError(f"'{pseudotime_key}' not found in adata.obs.")
        
    pt = adata.obs[pseudotime_key].values
    valid_idx = ~np.isnan(pt)
    pt_valid = pt[valid_idx]
    
    # Sort cells by pseudotime
    sorted_idx = np.argsort(pt_valid)
    actual_cell_indices = np.where(valid_idx)[0][sorted_idx]
    
    X_sorted = adata.X[actual_cell_indices]
    
    if sp.issparse(X_sorted):
        X_sorted = X_sorted.toarray()
        
    # Binning along trajectory
    bin_size = max(1, len(pt_valid) // n_bins)
    entropies = []
    cell_entropy_map = np.zeros(adata.n_obs)
    
    import pandas as pd
    bin_degrees = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else len(pt_valid)
        
        # Extract expression for cells in this bin
        X_bin = X_sorted[start:end, :]
        
        # Center the data
        X_bin_centered = X_bin - X_bin.mean(axis=0)
        
        # Calculate Pearson correlation matrix for the genes in this bin
        # To avoid massive memory issues, we restrict to HVGs if available
        if 'highly_variable' in adata.var:
            X_bin_centered = X_bin_centered[:, adata.var['highly_variable']]
            
        # Add small noise to avoid division by zero
        X_bin_centered += np.random.normal(0, 1e-8, X_bin_centered.shape)
        
        # Correlation matrix
        corr_matrix = np.corrcoef(X_bin_centered, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Sandpile Model: Calculate degree distribution of the network 
        # (genes with correlation > threshold are connected)
        adjacency = np.abs(corr_matrix) > correlation_threshold
        degrees = adjacency.sum(axis=1)
        bin_degrees.append(degrees)
        
        # Calculate Shannon entropy of the degree distribution
        # High entropy = high instability = tipping point
        degree_counts = np.bincount(degrees)[1:] # Ignore isolated nodes (degree 0)
        if len(degree_counts) > 0:
            probs = degree_counts / degree_counts.sum()
            bin_entropy = entropy(probs)
        else:
            bin_entropy = 0.0
            
        entropies.append(bin_entropy)
        
        # Assign this bin's entropy to its constituent cells
        original_indices = actual_cell_indices[start:end]
        cell_entropy_map[original_indices] = bin_entropy
        
    # Tipping point is identified as the peak of entropy (maximum instability)
    tipping_bin = int(np.argmax(entropies))
    
    # Extract the key genes at the tipping point
    tipping_degrees = bin_degrees[tipping_bin]
    gene_names = adata.var_names
    if 'highly_variable' in adata.var:
        gene_names = gene_names[adata.var['highly_variable']]
        
    tipping_genes_df = pd.DataFrame({
        'gene': gene_names,
        'degree_weight': tipping_degrees
    }).sort_values(by='degree_weight', ascending=False)
    
    adata.obs['sandpile_entropy'] = cell_entropy_map
    
    print(f"Tipping point identified at bin {tipping_bin} with Max Entropy: {entropies[tipping_bin]:.3f}")
    
    return {
        "bins": np.arange(n_bins),
        "entropy": entropies,
        "tipping_point_bin": tipping_bin,
        "tipping_genes": tipping_genes_df
    }
    
def plot_tipping_genes(tipping_genes_df: pd.DataFrame, top_n: int = 20, save_path: str = None) -> None:
    """
    Plot the top genes driving the tipping point ranked by their network degree (weight).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_plot = tipping_genes_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(6, max(4, top_n * 0.25)))
    sns.barplot(data=df_plot, x='degree_weight', y='gene', ax=ax, palette='viridis')
    ax.set_title("Top Genes Driving the Tipping Point")
    ax.set_xlabel("Network Degree (Weight)")
    ax.set_ylabel("Gene")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
