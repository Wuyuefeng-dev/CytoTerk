"""
Cross-species integration (e.g., Human vs Mouse).
"""

from anndata import AnnData
import pandas as pd

def convert_orthologs(
    adata: AnnData, 
    ortholog_table: pd.DataFrame, 
    from_species: str = "mouse", 
    to_species: str = "human"
) -> AnnData:
    """
    Convert gene names in an AnnData object between species using an ortholog table.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    ortholog_table : pd.DataFrame
        DataFrame containing mapping between species. Must have columns matching species names.
    from_species : str
        Source species column name in `ortholog_table`.
    to_species : str
        Target species column name in `ortholog_table`.
        
    Returns
    -------
    AnnData
        A new AnnData object with translated gene names and subsets to 1:1 orthologs.
    """
    if from_species not in ortholog_table.columns or to_species not in ortholog_table.columns:
        raise ValueError(f"Ortholog table must contain '{from_species}' and '{to_species}' columns.")
        
    # Create mapping dictionary
    mapping = dict(zip(ortholog_table[from_species], ortholog_table[to_species]))
    
    # Find overlapping genes
    genes_in_data = adata.var_names
    mapped_genes = []
    keep_indices = []
    
    for i, gene in enumerate(genes_in_data):
        if gene in mapping:
            mapped_genes.append(mapping[gene])
            keep_indices.append(i)
            
    # Subset and rename
    adata_mapped = adata[:, keep_indices].copy()
    adata_mapped.var_names = mapped_genes
    
    print(f"Mapped {len(mapped_genes)} genes from {from_species} to {to_species}.")
    return adata_mapped
