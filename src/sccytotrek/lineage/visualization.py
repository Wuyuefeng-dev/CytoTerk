import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def plot_lineage_umap(
    adata: ad.AnnData, 
    barcode_key: str = "barcode", 
    status_key: str = "barcode_imputed_status",
    palette: str = "tab20",
    show: bool = True,
    save: str = None
):
    """
    Visualize imputed lineage tracing over the RNA UMAP embedding.
    Highlights cells that have been imputed versus original.
    """
    if "X_umap" not in adata.obsm:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Plot all clones
    sc.pl.umap(adata, color=barcode_key, ax=axes[0], show=False, palette=palette, legend_loc='on data')
    axes[0].set_title(f"Clonal Assignment ({barcode_key})")
    
    # 2. Plot imputation status
    if status_key in adata.obs:
        sc.pl.umap(adata, color=status_key, ax=axes[1], show=False, palette={"original": "lightgrey", "imputed": "red"})
        axes[1].set_title("Imputation Status")
    else:
        axes[1].axis('off')
        
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()

def plot_clone_size_distribution(
    adata: ad.AnnData, 
    barcode_key: str = "barcode", 
    status_key: str = "barcode_imputed_status",
    show: bool = True,
    save: str = None
):
    """
    Plots the distribution of clone sizes before and after imputation.
    """
    if status_key not in adata.obs:
        raise ValueError(f"Status key '{status_key}' not found. Was imputation run?")
        
    counts = adata.obs.groupby([barcode_key, status_key]).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', stacked=True, color={'original': '#1f77b4', 'imputed': '#ff7f0e'}, ax=ax)
    
    ax.set_title("Clone Sizes: Original vs Imputed")
    ax.set_xlabel("Clone Barcode")
    ax.set_ylabel("Number of Cells")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Only show top N clones if there are too many
    if len(counts) > 30:
        ax.set_xticklabels([])
        ax.set_xlabel("Clone Barcode (Top 30+)")
        
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
