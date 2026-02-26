"""
Pathway analysis using GSEApy for true ssGSEA and Over-Representation Analysis.
"""

from anndata import AnnData
import pandas as pd
import numpy as np
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

def run_ssgsea(adata: AnnData, gene_sets: str | dict, outdir: str = 'gseapy_out') -> pd.DataFrame:
    """
    Run Single-Sample GSEA (ssGSEA) using GSEApy.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gene_sets : str or dict
        Name of Enrichr library (e.g., 'KEGG_2021_Human') or dict of custom gene sets.
    outdir : str
        Output directory for GSEApy.
        
    Returns
    -------
    pd.DataFrame
        ssGSEA normalized enrichment scores for each cell.
    """
    if not GSEAPY_AVAILABLE:
        raise ImportError("gseapy not installed. pip install gseapy")
        
    print(f"Running ssGSEA with GSEApy on {adata.n_obs} cells...")
    
    # GSEApy ssGSEA expects a DataFrame of Genes x Samples (Cells)
    if hasattr(adata.X, "toarray"):
        df = pd.DataFrame(adata.X.toarray().T, index=adata.var_names, columns=adata.obs_names)
    else:
        df = pd.DataFrame(adata.X.T, index=adata.var_names, columns=adata.obs_names)
        
    # Optional: GSEApy can be very slow for large cell numbers (>10k). Let's warn the user.
    if adata.n_obs > 5000:
        print("Warning: ssGSEA on large datasets might be slow.")
        
    ss = gp.ssgsea(data=df, gene_sets=gene_sets, outdir=outdir, sample_norm_method='rank', no_plot=True)
    
    # ss.res2d is the Normalized Enrichment Score (NES) dataframe
    nes_df = ss.res2d.T
    
    # Store results in AnnData directly for ease of plotting
    adata.obsm['ssGSEA'] = nes_df
    return nes_df

def run_go_enrichment(gene_list: list, background: int = 20000, gene_sets: str = 'GO_Biological_Process_2021') -> pd.DataFrame:
    """
    Run basic GO enrichment (Over-Representation Analysis) using GSEApy on a list of DE genes.
    
    Parameters
    ----------
    gene_list : list
        List of gene symbols.
    background : int or list
        Background size or list of background genes.
    gene_sets : str
        Enrichr library name.
        
    Returns
    -------
    pd.DataFrame
    """
    
    enr = gp.enrichr(gene_list=gene_list,
                     gene_sets=gene_sets,
                     organism='Human', 
                     background=background,
                     outdir=None)
    return enr.results

def plot_gsva_umap(
    adata: AnnData, 
    pathway_name: str, 
    cmap: str = 'Reds', 
    title: str = None,
    save_path: str = None
):
    """
    Plot the GSVA/ssGSEA enrichment score for a specific pathway on the UMAP.
    Requires `run_ssgsea` to have been run, saving scores in `adata.obsm['ssGSEA']`.
    """
    import scanpy as sc
    import matplotlib.pyplot as plt
    
    if 'ssGSEA' not in adata.obsm:
        raise ValueError("ssGSEA scores not found. Run `run_ssgsea` first.")
        
    nes_df = adata.obsm['ssGSEA']
    
    if pathway_name not in nes_df.columns:
        raise ValueError(f"Pathway '{pathway_name}' not found in ssGSEA results.")
        
    # Temporarily add the score to obs for easy Scanpy plotting
    tmp_col = f"GSVA_{pathway_name.replace(' ', '_')}"
    adata.obs[tmp_col] = nes_df[pathway_name].values
    
    plot_title = title if title else f"GSVA: {pathway_name}"
    
    fig = sc.pl.umap(
        adata, 
        color=tmp_col, 
        cmap=cmap, 
        title=plot_title,
        show=False,
        return_fig=True
    )
    
    # Cleanup
    del adata.obs[tmp_col]
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        
    plt.close(fig)
    return fig
