"""
Drug target prediction leveraging DrugBank annotations.
"""

from anndata import AnnData
import pandas as pd
import numpy as np

def score_drug_targets(
    adata: AnnData, 
    drugbank_df: pd.DataFrame, 
    group_key: str = 'leiden_0.5',
    target_gene_col: str = 'Gene_Name',
    drug_name_col: str = 'Drug_Name'
) -> pd.DataFrame:
    """
    Score potential drug efficacy based on the expression of drug target genes 
    in different cell clusters/groups.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    drugbank_df : pd.DataFrame
        DataFrame containing drug-target relationships derived from DrugBank.
    group_key : str
        Column in adata.obs defining cell groups.
    target_gene_col : str
        Column in drugbank_df for the target gene name.
    drug_name_col : str
        Column in drugbank_df for the drug name.
        
    Returns
    -------
    pd.DataFrame
        Scores for each drug across the specified groups based on aggregated target expression.
    """
    if group_key not in adata.obs:
        raise ValueError(f"Group '{group_key}' not in adata.obs")
        
    print(f"Scoring {len(drugbank_df[drug_name_col].unique())} drugs against cell groups in '{group_key}'...")
    
    # Filter targets to those present in the dataset
    genes_in_data = set(adata.var_names)
    valid_targets = drugbank_df[drugbank_df[target_gene_col].isin(genes_in_data)].copy()
    
    groups = adata.obs[group_key].unique()
    results = []
    
    for drug, targets in valid_targets.groupby(drug_name_col):
        target_genes = targets[target_gene_col].unique()
        
        # Calculate mean expression of all targets for this drug per cluster
        # Using scanpy's built-in score_genes but simplified here for direct access
        
        for g in groups:
            idx = adata.obs[group_key] == g
            if np.sum(idx) == 0:
                continue
                
            # Mean expression across all targets for the drug
            if hasattr(adata.X, "toarray"):
                expr = adata[idx, target_genes].X.mean()
            else:
                expr = np.mean(adata[idx, target_genes].X)
                
            results.append({
                'Drug': drug,
                'Group': g,
                'Score': expr,
                'Target_Count': len(target_genes)
            })
            
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        # Pivot for easier viewing: Drugs x Groups
        df_results_pivot = df_results.pivot(index='Drug', columns='Group', values='Score').fillna(0)
        return df_results_pivot
    
    return pd.DataFrame()
