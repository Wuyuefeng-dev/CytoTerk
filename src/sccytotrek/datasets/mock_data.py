"""
Dataset generation and loading utilities.
"""

import numpy as np
import anndata as ad
import pandas as pd
try:
    import mudata as md
except ImportError:
    md = None

def make_mock_scrna(n_cells: int = 2000, n_genes: int = 2500, n_clusters: int = 5, random_state: int = 42) -> ad.AnnData:
    """
    Generate a mock scRNA-seq AnnData object with defined clusters.
    
    Parameters
    ----------
    n_cells : int
        Number of cells
    n_genes : int
        Number of genes
    n_clusters : int
        Number of synthetic cell types/clusters to simulate
        
    Returns
    -------
    anndata.AnnData
    """
    from sklearn.datasets import make_blobs
    np.random.seed(random_state)
    
    # 1. Generate base cluster centers using make_blobs
    # We use a smaller number of informative features, then pad with noise
    n_informative = min(50, n_genes)
    X, y = make_blobs(n_samples=n_cells, n_features=n_informative, centers=n_clusters, 
                      cluster_std=1.0, random_state=random_state)
    
    # Shift to ensure strictly positive rates
    X = X - X.min() + 0.1 
    
    # 2. Pad with baseline noise for the remaining non-informative genes
    if n_genes > n_informative:
        noise = np.random.lognormal(mean=0.5, sigma=0.5, size=(n_cells, n_genes - n_informative))
        X = np.hstack([X, noise])
        
    # Scale to typical library sizes (~5000 mean UMI per cell)
    X = X / X.sum(axis=1, keepdims=True) * np.random.normal(5000, 1000, size=(n_cells, 1))
    X[X < 0] = 0.1
    
    # 3. Sample from Poisson to get integer counts
    # This simulates technical sampling noise typical of scRNA-seq
    X_counts = np.random.poisson(X).astype(np.float32)
    
    # Predefined list of common human marker genes for actual biological context
    real_genes = [
        "CD3D", "CD3E", "CD8A", "CD4", "IL7R", "SELL", 
        "GNLY", "NKG7", "MS4A1", "CD79A", "CD14", "LYZ", 
        "FCGR3A", "MS4A7", "PPBP", "FCER1A", "CST3", 
        "CD8B", "LEF1", "CCR7", "TRAC", "GZMB", "GZMH",
        "PRF1", "CD27", "HLA-DRA", "HLA-DRB1", "CD68", "S100A9",
        "S100A8", "NCAM1", "CD19", "BANK1", "PAX5", "CD38",
        "SDC1", "MKI67", "TOP2A", "PCNA", "AURKA", "BIRC5",
        "EPCAM", "KRT18", "KRT19", "KRT8", "MUC1", "CDH1",
        "VIM", "ACTA2", "COL1A1"
    ]
    np.random.shuffle(real_genes)
    
    # 4. Construct AnnData
    
    # Generate Lineage Tracing Barcodes (simulate 50% dropout)
    n_clones = n_clusters * 2 # Few distinct clones
    base_barcodes = [f"clone_{np.random.randint(100, 999)}" for _ in range(n_cells)]
    
    # Assign true clones roughly matching clusters for biological realism
    for i in range(n_cells):
        base_barcodes[i] = f"clone_A{y[i]}" # Simple linkage to cluster
        
    # Induce 50% missing dropout
    is_missing = np.random.rand(n_cells) < 0.5
    for i in np.where(is_missing)[0]:
        base_barcodes[i] = "NA"
        
    obs = {
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "true_cluster": [str(c) for c in y],
        "barcode": base_barcodes
    }
    
    # Generate gene names: Use real genes for informative features, pad the rest
    gene_names = []
    for i in range(n_genes):
        if i < len(real_genes):
            gene_names.append(real_genes[i])
        else:
            # Pad with realistic sounding background genes
            prefix = np.random.choice(["RPL", "RPS", "MT-", "AC0", "AL0", "LINC"])
            gene_names.append(f"{prefix}{np.random.randint(1, 9999)}")
            
    # Ensure uniqueness in background genes
    gene_names = list(pd.Series(gene_names).drop_duplicates())
    while len(gene_names) < n_genes:
        prefix = np.random.choice(["RPL", "RPS", "MT-", "AC0", "AL0", "LINC"])
        new_gene = f"{prefix}{np.random.randint(1, 99999)}"
        if new_gene not in gene_names:
            gene_names.append(new_gene)
            
    var = {"gene_id": gene_names}
    
    adata = ad.AnnData(X=X_counts, obs=pd.DataFrame(obs), var=pd.DataFrame(var))
    adata.obs_names = adata.obs["cell_id"]
    adata.var_names = adata.var["gene_id"]
    
    return adata

def make_mock_multiome(
    n_cells: int = 2000, 
    n_genes: int = 2500, 
    n_peaks: int = 2000,
    n_batches: int = 3,
    random_state: int = 42
):
    """
    Generate a mock scMultiome MuData object (RNA + ATAC) with batch effects.
    
    Parameters
    ----------
    n_cells : int
        Total number of cells across all batches
    n_genes : int
        Number of genes in the RNA modality
    n_peaks : int
        Number of peaks in the ATAC modality
    n_batches : int
        Number of independent batches to simulate
        
    Returns
    -------
    mudata.MuData
    """
    if md is None:
        raise ImportError("Please install mudata to use multiome features.")
        
    from sklearn.datasets import make_blobs
    np.random.seed(random_state)
    
    cells_per_batch = n_cells // n_batches
    all_rna_X, all_atac_X, all_y = [], [], []
    batch_labels = []
    
    for b in range(n_batches):
        # Shift random state per batch to induce batch effect
        b_seed = random_state + b * 10 
        
        # 1. Base clusters layout (5 clusters)
        n_informative = min(50, n_genes)
        X, y = make_blobs(n_samples=cells_per_batch, n_features=n_informative, centers=5, 
                          cluster_std=1.0 + (b * 0.2), random_state=b_seed)
        
        # Shift to positive and add batch-specific linear shift
        batch_shift = np.random.uniform(-0.5, 0.5)
        X = X - X.min() + 0.1 + batch_shift
        
        # --- Build RNA Matrix for this batch ---
        X_rna = X.copy()
        if n_genes > n_informative:
            noise = np.random.lognormal(mean=0.5, sigma=0.5, size=(cells_per_batch, n_genes - n_informative))
            X_rna = np.hstack([X_rna, noise])
            
        X_rna = X_rna / X_rna.sum(axis=1, keepdims=True) * np.random.normal(5000, 1000, size=(cells_per_batch, 1))
        X_rna[X_rna < 0] = 0.1
        batch_rna_X = np.random.poisson(X_rna).astype(np.float32)
        all_rna_X.append(batch_rna_X)
        
        # --- Build ATAC Matrix for this batch ---
        X_atac = X.copy()
        if n_peaks > n_informative:
            # ATAC is much sparser
            noise_atac = np.random.exponential(scale=0.1, size=(cells_per_batch, n_peaks - n_informative))
            # Shift ATAC features slightly to break perfect collinearity with RNA
            shift_matrix = np.random.normal(0, 0.5, size=(cells_per_batch, n_informative))
            X_atac = np.hstack([X_atac + shift_matrix, noise_atac])
            
        # Sparsify ATAC
        X_atac[X_atac < np.percentile(X_atac, 85)] = 0
        X_atac = X_atac / (X_atac.sum(axis=1, keepdims=True) + 1e-6) * np.random.normal(10000, 2000, size=(cells_per_batch, 1))
        batch_atac_X = np.random.poisson(X_atac).astype(np.float32)
        all_atac_X.append(batch_atac_X)
        
        all_y.extend(list(y))
        batch_labels.extend([f"Batch_{b+1}"] * cells_per_batch)
        
    rna_X = np.vstack(all_rna_X)
    atac_X = np.vstack(all_atac_X)
    y = np.array(all_y)
    n_cells_actual = rna_X.shape[0]
    
    # 2. Complete RNA feature space
    if n_genes > n_informative:
        noise = np.random.lognormal(mean=0.5, sigma=0.5, size=(n_cells, n_genes - n_informative))
        X = np.hstack([X, noise])
        
    X = X / X.sum(axis=1, keepdims=True) * np.random.normal(5000, 1000, size=(n_cells, 1))
    X[X < 0] = 0.1
    rna_X = np.random.poisson(X).astype(np.float32)
    
    # 3. Gene names
    real_genes = [
        "CD3D", "CD3E", "CD8A", "CD4", "IL7R", "SELL", 
        "GNLY", "NKG7", "MS4A1", "CD79A", "CD14", "LYZ", 
        "FCGR3A", "MS4A7", "PPBP", "FCER1A", "CST3", 
        "CD8B", "LEF1", "CCR7", "TRAC", "GZMB", "GZMH",
        "PRF1", "CD27", "HLA-DRA", "HLA-DRB1", "CD68", "S100A9",
        "S100A8", "NCAM1", "CD19", "BANK1", "PAX5", "CD38",
        "SDC1", "MKI67", "TOP2A", "PCNA", "AURKA", "BIRC5",
        "EPCAM", "KRT18", "KRT19", "KRT8", "MUC1", "CDH1",
        "VIM", "ACTA2", "COL1A1"
    ]
    np.random.shuffle(real_genes)
    gene_names = []
    for i in range(n_genes):
        if i < len(real_genes):
            gene_names.append(real_genes[i])
        else:
            prefix = np.random.choice(["RPL", "RPS", "MT-", "AC0", "AL0", "LINC"])
            gene_names.append(f"{prefix}{np.random.randint(1, 9999)}")
            
    gene_names = list(pd.Series(gene_names).drop_duplicates())
    while len(gene_names) < n_genes:
        prefix = np.random.choice(["RPL", "RPS", "MT-", "AC0", "AL0", "LINC"])
        new_gene = f"{prefix}{np.random.randint(1, 99999)}"
        if new_gene not in gene_names:
            gene_names.append(new_gene)
            
    # 4. Create RNA AnnData
    rna_obs = {"cell_id": [f"cell_{i}" for i in range(n_cells)], "true_cluster": [str(c) for c in y]}
    rna_var = {"gene_id": gene_names}
    rna_adata = ad.AnnData(X=rna_X, obs=pd.DataFrame(rna_obs), var=pd.DataFrame(rna_var))
    rna_adata.obs_names = rna_adata.obs["cell_id"]
    rna_adata.var_names = rna_adata.var["gene_id"]
    
    # 5. ATAC
    # For ATAC, just generate noise directly (minimal linkage for demo)
    atac_X = np.random.poisson(lam=0.5, size=(n_cells, n_peaks)).astype(np.float32)
    atac_obs = {"cell_id": [f"cell_{i}" for i in range(n_cells)], "true_cluster": [str(c) for c in y]}
    atac_var = {"peak_id": [f"chr1_{i*100}_{i*100+50}" for i in range(n_peaks)]}
    atac_adata = ad.AnnData(X=atac_X, obs=pd.DataFrame(atac_obs), var=pd.DataFrame(atac_var))
    atac_adata.obs_names = atac_adata.obs["cell_id"]
    atac_adata.var_names = atac_adata.var["peak_id"]
    
    # MuData
    mdata = md.MuData({"rna": rna_adata, "atac": atac_adata})
    return mdata
