"""
Quality control and preprocessing functions.
"""

import scanpy as sc
from anndata import AnnData

def calculate_qc_metrics(adata: AnnData, mt_prefix: str = "MT-") -> None:
    """
    Calculate basic quality control metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    mt_prefix : str, optional
        Prefix for mitochondrial genes, by default "MT-"
    """
    adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    
def plot_qc_violins(adata: AnnData, save_path: str = None) -> None:
    """
    Generate and save violin plots for basic quality control metrics.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with QC metrics calculated.
    save_path : str, optional
        Path to save the generated figure.
    """
    import matplotlib.pyplot as plt
    if not all(k in adata.obs.columns for k in ['n_genes_by_counts', 'total_counts']):
        print("Missing QC metrics. Did you run `calculate_qc_metrics` first?")
        return
        
    keys_to_plot = ['n_genes_by_counts', 'total_counts']
    if 'pct_counts_mt' in adata.obs.columns:
        keys_to_plot.append('pct_counts_mt')
        
    sc.pl.violin(adata, keys_to_plot, jitter=0.4, multi_panel=True, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
def filter_cells_and_genes(
    adata: AnnData, 
    min_genes: int = 200, 
    min_cells: int = 3,
    max_pct_mt: float = 20.0
) -> None:
    """
    Filter out low quality cells and rare genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    min_genes : int, optional
        Minimum number of expressed genes for a cell to pass filtering, by default 200
    min_cells : int, optional
        Minimum number of cells expressing a gene for it to pass filtering, by default 3
    max_pct_mt : float, optional
        Maximum allowed percentage of mitochondrial counts, by default 20.0
    """
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    # Filter by mitochondrial fraction if calculated
    if "pct_counts_mt" in adata.obs.columns:
        adata._inplace_subset_obs(adata.obs["pct_counts_mt"] < max_pct_mt)

def normalize_and_log(adata: AnnData, target_sum: float = 1e4) -> None:
    """
    Total-count normalize (library-size correct) the data matrix to 10,000 reads per cell, 
    so that counts become comparable among cells, and then logarithmize the data.
    """
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
