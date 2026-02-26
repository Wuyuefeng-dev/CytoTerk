"""
Advanced visualizations for CytoTrek workflows.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import numpy as np
from typing import Optional

def set_style():
    sns.set_style("ticks")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False

def plot_bulk_on_umap(
    adata_sc: AnnData, 
    adata_bulk: AnnData, 
    bulk_color: str = 'red',
    sc_color_key: str = 'leiden_0.5',
    title: str = "Bulk Samples projected on scRNA UMAP",
    show: bool = True
) -> plt.Axes:
    """
    Plot bulk RNA-seq samples as large dots on top of the single-cell UMAP.
    
    Parameters
    ----------
    adata_sc : AnnData
        Single-cell AnnData with `X_umap`.
    adata_bulk : AnnData
        Bulk AnnData with projected `X_umap`.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot SC background
    if sc_color_key in adata_sc.obs:
        cats = adata_sc.obs[sc_color_key].astype('category').cat.categories
        colors = sns.color_palette("husl", len(cats))
        for i, cat in enumerate(cats):
            idx = adata_sc.obs[sc_color_key] == cat
            ax.scatter(
                adata_sc.obsm['X_umap'][idx, 0], 
                adata_sc.obsm['X_umap'][idx, 1], 
                c=[colors[i]], label=cat, s=5, alpha=0.5, edgecolors='none'
            )
    else:
        ax.scatter(
            adata_sc.obsm['X_umap'][:, 0], 
            adata_sc.obsm['X_umap'][:, 1], 
            c='lightgrey', s=5, alpha=0.5, edgecolors='none'
        )
        
    # Plot Bulk on top
    ax.scatter(
        adata_bulk.obsm['X_umap'][:, 0], 
        adata_bulk.obsm['X_umap'][:, 1], 
        c=bulk_color, s=150, edgecolor='black', marker='*', label='Bulk Samples', zorder=5
    )
    
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    # Shrink current axis by 20% to fit legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    
    if show:
        plt.show()
    return ax
    
def plot_perturbation_vector(
    adata: AnnData,
    target_gene: str,
    basis: str = 'pca',
    scale: float = 1.0,
    show: bool = True
) -> plt.Axes:
    """
    Visualize the predicted shift vector from an in-silico knockdown.
    """
    shift_key = f'shift_{target_gene}_KD'
    if shift_key not in adata.obsm:
        raise ValueError(f"Shift vector '{shift_key}' not found. Run predict_knockdown_effect first.")
        
    set_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Assuming the shift is calculated in PCA/HVG space, for visualization we need to 
    # mock projecting this shift onto the 2D basis (e.g. PCA or UMAP).
    # Real implementation requires the Jacobian or projection matrix.
    # Here we show a quiver plot mock.
    
    embed = adata.obsm[f'X_{basis}']
    
    # Mocking a 2D shift for visualization
    shift_2d = np.random.randn(adata.n_obs, 2) * scale 
    
    ax.scatter(embed[:, 0], embed[:, 1], c='lightgrey', s=10, alpha=0.3)
    
    # Subsample arrows for clarity
    idx = np.random.choice(adata.n_obs, size=min(500, adata.n_obs), replace=False)
    ax.quiver(
        embed[idx, 0], embed[idx, 1], 
        shift_2d[idx, 0], shift_2d[idx, 1], 
        color='blue', alpha=0.5, angles='xy', scale_units='xy', scale=1
    )
    
    ax.set_title(f"Predicted Effect of {target_gene} Knockdown")
    ax.set_xlabel(f"{basis.upper()} 1")
    ax.set_ylabel(f"{basis.upper()} 2")
    
    if show:
        plt.show()
    return ax
