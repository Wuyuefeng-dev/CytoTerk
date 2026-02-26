import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Optional
from .plotting_advanced import set_style

def plot_monocle3_trajectory(
    adata: AnnData, 
    group_key: str = 'leiden_0.5', 
    color_by: str = 'monocle3_pseudotime',
    cmap: str = 'viridis',
    title: str = "Monocle3-style Principal Graph",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the UMAP with the Monocle3 principal graph (MST) layered on top.
    """
    if 'principal_graph' not in adata.uns:
        raise ValueError("Run `run_monocle3` first to generate the principal graph.")
        
    set_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # 1. Plot cells
    if color_by in adata.obs:
        if adata.obs[color_by].dtype.name in ['category', 'object']:
            # Discrete
            sns.scatterplot(
                x=adata.obsm['X_umap'][:, 0], 
                y=adata.obsm['X_umap'][:, 1],
                hue=adata.obs[color_by],
                palette='husl',
                s=15, ax=ax, edgecolor='none', alpha=0.7
            )
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        else:
            # Continuous (e.g., pseudotime)
            scat = ax.scatter(
                adata.obsm['X_umap'][:, 0], 
                adata.obsm['X_umap'][:, 1],
                c=adata.obs[color_by],
                cmap=cmap,
                s=15, edgecolor='none', alpha=0.7
            )
            plt.colorbar(scat, ax=ax, label=color_by)
    else:
        ax.scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c='lightgrey', s=10)
        
    # 2. Plot the Principal Graph (MST)
    T = adata.uns['principal_graph']
    
    # Draw edges
    for (u, v) in T.edges():
        pos_u = T.nodes[u]['pos']
        pos_v = T.nodes[v]['pos']
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 
                color='black', linewidth=2, zorder=4)
                
    # Draw nodes (centroids)
    for node in T.nodes():
        pos = T.nodes[node]['pos']
        ax.scatter(pos[0], pos[1], s=100, c='black', edgecolor='white', zorder=5)
        # Add labels
        ax.text(pos[0] + 0.2, pos[1] + 0.2, str(node), 
                fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
                
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
    return fig

def plot_streamgraph(
    adata: AnnData, 
    time_key: str, 
    group_key: str, 
    n_bins: int = 50,
    title: str = "Streamgraph over Pseudotime",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a Streamgraph (stacked area plot) showing the proportion of 
    cell groups across pseudotime.
    
    Parameters
    ----------
    adata : AnnData
    time_key : str
        Column in adata.obs containing pseudotime (e.g., 'dpt_pseudotime', 'monocle3_pseudotime')
    group_key : str
        Column in adata.obs containing categorical cell identities (e.g., 'leiden_0.5', 'cell_type')
    n_bins : int
        Number of pseudotime bins to compute proportions over
        
    Returns
    -------
    matplotlib.pyplot.Figure
    """
    import pandas as pd
    import numpy as np
    
    if time_key not in adata.obs:
        raise ValueError(f"Pseudotime key '{time_key}' not found in adata.obs.")
    if group_key not in adata.obs:
        raise ValueError(f"Group key '{group_key}' not found in adata.obs.")
        
    # Extract data securely
    df = pd.DataFrame({
        'time': adata.obs[time_key].values,
        'group': adata.obs[group_key].values
    }).dropna()
    
    # Bin pseudotime
    df['bin'] = pd.cut(df['time'], bins=n_bins, labels=False)
    
    # Calculate group proportions per bin
    counts = df.groupby(['bin', 'group']).size().unstack(fill_value=0)
    
    # Smooth the counts to make it "stream-like"
    from scipy.ndimage import gaussian_filter1d
    smoothed_counts = counts.apply(lambda x: gaussian_filter1d(x.values, sigma=1.5), axis=0)
    
    # Normalize to proportions (0 to 1) per bin
    proportions = smoothed_counts.div(smoothed_counts.sum(axis=1), axis=0).fillna(0)
    
    # Setup plotting
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # X-axis will be the bin midpoints mapped roughly to the time range
    time_min, time_max = df['time'].min(), df['time'].max()
    x_axis = np.linspace(time_min, time_max, len(proportions))
    
    # Select dynamic seaborn palette
    groups = proportions.columns
    colors = sns.color_palette("husl", n_colors=len(groups))
    
    # Stackplot constructs the streamgraph
    ax.stackplot(x_axis, proportions.T.values, labels=groups, colors=colors, baseline='wiggle', alpha=0.8)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=group_key)
    ax.set_title(title)
    ax.set_xlabel(f"Pseudotime ({time_key})")
    ax.set_ylabel("Relative Density")
    ax.margins(x=0, y=0)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
    return fig
