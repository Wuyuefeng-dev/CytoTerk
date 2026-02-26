import numpy as np
import pandas as pd
import anndata as ad
from collections import Counter

def impute_barcodes_knn(
    adata: ad.AnnData, 
    barcode_key: str = "barcode", 
    missing_val: str = "NA", 
    n_neighbors: int = 15, 
    use_rep: str = "X_pca",
    inplace: bool = False
) -> ad.AnnData:
    """
    Impute missing lineage barcodes using k-Nearest Neighbors on RNA expression.
    
    Parameters
    ----------
    adata : anndata.AnnData
        The annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to cells.
    barcode_key : str
        The key in `adata.obs` where barcode information is stored.
    missing_val : str
        The string representing missing barcode values.
    n_neighbors : int
        Number of neighbors to consider for majority vote imputation.
    use_rep : str
        The representation in `adata.obsm` to use for kNN (e.g., 'X_pca', 'X_umap').
    inplace : bool
        Whether to modify `adata` in place or return a copy.
        
    Returns
    -------
    anndata.AnnData (if inplace=False)
        Updated with imputed barcodes and metadata.
    """
    if not inplace:
        adata = adata.copy()

    if barcode_key not in adata.obs:
        raise ValueError(f"Key '{barcode_key}' not found in adata.obs")
        
    if use_rep not in adata.obsm:
        import scanpy as sc
        print(f"Representation '{use_rep}' not found. Computing PCA...")
        if 'log1p' not in adata.uns:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        # Fix PCA by selecting highly variable genes subset only for PCA or just compute pca directly.
        sc.pp.pca(adata)
        
    # Split into missing and non-missing
    # We account for NaN or literal missing_val string
    mask_missing = (adata.obs[barcode_key] == missing_val) | (adata.obs[barcode_key].isna()) | (adata.obs[barcode_key] == "")
    
    if not mask_missing.any():
        print("No missing barcodes found. Returning original object.")
        return adata
        
    mask_present = ~mask_missing
    
    X_train = adata.obsm[use_rep][mask_present]
    y_train = adata.obs[barcode_key][mask_present].values
    
    X_test = adata.obsm[use_rep][mask_missing]
    
    # Fit KNN Classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    # Store imputed vs original status
    status_key = f"{barcode_key}_imputed_status"
    adata.obs[status_key] = "original"
    
    # Explicit conversion to object/category to avoid assignment errors on subsets
    if pd.api.types.is_categorical_dtype(adata.obs[barcode_key]):
        adata.obs[barcode_key] = adata.obs[barcode_key].astype(str)
        
    # Fill in the missing values
    adata.obs.loc[mask_missing, status_key] = "imputed"
    adata.obs.loc[mask_missing, barcode_key] = y_pred
    
    # Add info on imputation confidence (prediction probability entropy)
    proba = knn.predict_proba(X_test)
    # the max probability assigned to the majority class
    prob_max = proba.max(axis=1)
    
    confidence_key = f"{barcode_key}_imputation_confidence"
    if confidence_key not in adata.obs:
        adata.obs[confidence_key] = 1.0 # 1.0 for original, since we are sure
    
    adata.obs.loc[mask_missing, confidence_key] = prob_max

    return adata if not inplace else None
