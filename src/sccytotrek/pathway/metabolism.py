"""
Metabolism analysis to evaluate release and sense (uptake/response) of metabolites.
"""

from anndata import AnnData
import pandas as pd
import scanpy as sc

def score_metabolism_status(
    adata: AnnData, 
    metabolism_signatures: dict, 
    group_key: str = 'leiden_0.5'
) -> AnnData:
    """
    Score major metabolism pathways (e.g., release vs sense) using signature gene sets.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    metabolism_signatures : dict
        A dictionary where keys are pathway names (e.g., 'Glucose_Sense', 'Lactate_Release')
        and values are lists of gene symbols.
    group_key : str
        Column in adata.obs for grouping (optional usage but good for downstream plotting).
        
    Returns
    -------
    AnnData
        Updated AnnData with metabolism scores in `adata.obs`.
    """
    print("Scoring metabolism signatures...")
    
    for pathway, genes in metabolism_signatures.items():
        genes_present = [g for g in genes if g in adata.var_names]
        
        if not genes_present:
            print(f"Warning: No genes found for pathway '{pathway}'. Skipping.")
            continue
            
        print(f"Scoring '{pathway}' ({len(genes_present)}/{len(genes)} genes present)...")
        # Use Scanpy's built-in scoring feature for general expression scoring
        sc.tl.score_genes(adata, gene_list=genes_present, score_name=pathway)
        
    return adata
