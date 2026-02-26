"""
Identify cell types based on canonical markers (e.g. malignant cells).
"""

from anndata import AnnData
import scanpy as sc
import numpy as np

def score_cell_types(adata: AnnData, marker_dict: dict, groupby: str = 'leiden_0.5') -> AnnData:
    """
    Score and assign putative cell types (e.g., Malignant, T-cell, Macrophage) 
    using a dictionary of marker genes.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    marker_dict : dict
        Dictionary where keys are cell types and values are lists of marker genes.
    groupby : str
        Cluster labels to summarize scores over.
        
    Returns
    -------
    AnnData
        Updated AnnData with assigned cell types in `adata.obs`.
    """
    print("Scoring cell types based on marker signatures...")
    
    score_names = []
    for celltype, genes in marker_dict.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            score_name = f'score_{celltype}'
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name)
            score_names.append(score_name)
            
    if not score_names:
        print("No valid marker genes found.")
        return adata
        
    # Assign cluster identities based on max average score
    cluster_mapping = {}
    for cluster in adata.obs[groupby].unique():
        cluster_cells = adata.obs[adata.obs[groupby] == cluster]
        
        # Calculate mean scores for this cluster
        means = {score: cluster_cells[score].mean() for score in score_names}
        best_match = max(means, key=means.get).replace('score_', '')
        
        # If all scores are negative, might be unknown
        if all(m < 0 for m in means.values()):
            cluster_mapping[cluster] = "Unknown"
        else:
            cluster_mapping[cluster] = best_match
            
    # Apply mapping
    adata.obs['cell_type_prediction'] = adata.obs[groupby].map(cluster_mapping)
    print(f"Cell types mapped to `adata.obs['cell_type_prediction']`. Identified: {set(cluster_mapping.values())}")
    
    return adata
