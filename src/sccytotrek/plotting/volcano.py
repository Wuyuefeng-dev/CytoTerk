import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from .style import apply_seurat_theme

def plot_volcano(
    de_res: pd.DataFrame,
    gene_col: str = 'gene',
    lfc_col: str = 'log2fc',
    pval_col: str = 'padj',
    lfc_thresh: float = 1.0,
    pval_thresh: float = 0.05,
    top_n: int = 15,
    title: str = "Volcano Plot",
    figsize: tuple = (6, 5),
    show: bool = True,
    save: Optional[str] = None
):
    """
    Plot a volcano plot from differential expression results.
    """
    if de_res.empty:
        print("Empty DataFrame provided. Cannot plot Volcano.")
        return None
        
    fig, ax = plt.subplots(figsize=figsize)
    apply_seurat_theme(ax)

    x = de_res[lfc_col].values
    # Add a small epsilon to avoid log(0) if pval == 0
    y = -np.log10(de_res[pval_col].values + 1e-300)

    # Determine significance
    sig_up = (x > lfc_thresh) & (de_res[pval_col].values < pval_thresh)
    sig_dn = (x < -lfc_thresh) & (de_res[pval_col].values < pval_thresh)
    ns = ~(sig_up | sig_dn)

    ax.scatter(x[ns], y[ns], color='lightgray', s=10, alpha=0.5, label='Not Sig')
    ax.scatter(x[sig_up], y[sig_up], color='indianred', s=15, alpha=0.8, label='Up')
    ax.scatter(x[sig_dn], y[sig_dn], color='steelblue', s=15, alpha=0.8, label='Down')

    ax.axvline(lfc_thresh, color='gray', linestyle='--', lw=0.8)
    ax.axvline(-lfc_thresh, color='gray', linestyle='--', lw=0.8)
    ax.axhline(-np.log10(pval_thresh), color='gray', linestyle='--', lw=0.8)

    # Label top genes
    if top_n > 0:
        genes = de_res[gene_col].values
        
        # Top UP
        idx_up = np.where(sig_up)[0]
        if len(idx_up) > 0:
            # Sort by p-value (lowest) -> highest y
            top_up_idx = idx_up[np.argsort(y[idx_up])[-top_n:]]
            for i in top_up_idx:
                ax.text(x[i] + 0.1, y[i], genes[i], fontsize=8, ha='left', va='center', color='black')
                
        # Top DOWN
        idx_dn = np.where(sig_dn)[0]
        if len(idx_dn) > 0:
            top_dn_idx = idx_dn[np.argsort(y[idx_dn])[-top_n:]]
            for i in top_dn_idx:
                ax.text(x[i] - 0.1, y[i], genes[i], fontsize=8, ha='right', va='center', color='black')

    ax.set_title(title)
    ax.set_xlabel(f"{lfc_col} (log2)")
    ax.set_ylabel(f"-log10({pval_col})")
    ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.05, 1))

    if save:
        fig.savefig(save, bbox_inches='tight', dpi=150)
        print(f"  Saved volcano plot: {save}")
    
    if show:
        plt.show()
    
    if not show and not save:
        return fig
    else:
        plt.close(fig)
