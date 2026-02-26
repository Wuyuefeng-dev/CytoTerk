"""
Multiome (RNA+ATAC, RNA+Methylation) support functions.
"""

from anndata import AnnData
import scanpy as sc

try:
    import mudata as md
except ImportError:
    md = None
    
def run_wnn(mdata, rna_key: str = "rna", atac_key: str = "atac") -> None:
    """
    Placeholder for Weighted Nearest Neighbors (WNN) joint multi-modal integration.
    Often, muon (seurat-like) or scvi-tools are used for this.
    For demonstration, we establish a combined neighbor graph.
    
    Parameters
    ----------
    mdata : mudata.MuData
        Multiome dataset container.
    rna_key : str
        Key for the RNA modality.
    atac_key : str
        Key for the ATAC/Methylation modality.
    """
    if md is None:
        raise ImportError("Please install mudata to use multiome features.")
        
    print(f"Running WNN across {rna_key} and {atac_key} modalities...")
    
    # 1. Process each modality independently (PCA for RNA, LSI for ATAC)
    if 'X_pca' not in mdata.mod[rna_key].obsm:
        sc.pp.pca(mdata.mod[rna_key])
        
    if 'X_lsi' not in mdata.mod[atac_key].obsm:
        # Dummy LSI / PCA for this example
        sc.pp.pca(mdata.mod[atac_key], n_comps=50)
        mdata.mod[atac_key].obsm['X_lsi'] = mdata.mod[atac_key].obsm['X_pca']
        
    # 2. Compute neighbors for both
    sc.pp.neighbors(mdata.mod[rna_key])
    sc.pp.neighbors(mdata.mod[atac_key], use_rep='X_lsi')
    
    # 3. Create a unified representation (Placeholder logic, mudata handles this structurally)
    mdata.update()
    # E.g., a real implementation would use muon.pp.neighbors(mdata)
    
def run_cca_integration(mdata, rna_key: str = "rna", atac_key: str = "atac", n_components: int = 20) -> None:
    """
    Canonical Correlation Analysis (CCA) for joint embedding.
    Extracts the maximally correlated subspace between RNA and ATAC modalities.
    """
    print(f"Running CCA Integration across {rna_key} and {atac_key}...")
    from sklearn.cross_decomposition import CCA
    import numpy as np
    
    rna_X = mdata.mod[rna_key].X.toarray() if hasattr(mdata.mod[rna_key].X, "toarray") else mdata.mod[rna_key].X
    atac_X = mdata.mod[atac_key].X.toarray() if hasattr(mdata.mod[atac_key].X, "toarray") else mdata.mod[atac_key].X
    
    # We use PCA first to reduce to manageable feature space before CCA for efficiency
    if 'X_pca' not in mdata.mod[rna_key].obsm:
        sc.pp.pca(mdata.mod[rna_key])
    if 'X_lsi' not in mdata.mod[atac_key].obsm:
        sc.pp.pca(mdata.mod[atac_key], n_comps=50)
        mdata.mod[atac_key].obsm['X_lsi'] = mdata.mod[atac_key].obsm['X_pca']
        
    cca = CCA(n_components=n_components)
    rna_c, atac_c = cca.fit_transform(mdata.mod[rna_key].obsm['X_pca'], mdata.mod[atac_key].obsm['X_lsi'])
    
    # The joint representation is the average of the canonical variates
    mdata.obsm['X_cca'] = (rna_c + atac_c) / 2.0
    
def run_concat_pca_integration(mdata, rna_key: str = "rna", atac_key: str = "atac") -> None:
    """
    Early Integration: Row-wise concatenation of modalities followed by Joint PCA.
    """
    print(f"Running Concatenated PCA Integration across {rna_key} and {atac_key}...")
    import numpy as np
    from sklearn.decomposition import PCA
    
    rna_pca = mdata.mod[rna_key].obsm['X_pca']
    atac_lsi = mdata.mod[atac_key].obsm['X_lsi']
    
    # L2 Norm normalize both spaces before concatenating to balance modality variance
    rna_pca = rna_pca / np.linalg.norm(rna_pca, axis=1, keepdims=True)
    atac_lsi = atac_lsi / np.linalg.norm(atac_lsi, axis=1, keepdims=True)
    
    concat = np.hstack([rna_pca, atac_lsi])
    
    pca = PCA(n_components=30)
    mdata.obsm['X_concat_pca'] = pca.fit_transform(concat)

def run_procrustes_integration(mdata, rna_key: str = "rna", atac_key: str = "atac") -> None:
    """
    Procrustes Alignment: Linearly transforms one embedding to optimally match the 
    shape of the other through translation, scaling, and rotation.
    Assumes paired cells (co-assays).
    """
    print(f"Running Procrustes Alignment across {rna_key} and {atac_key}...")
    from scipy.spatial import procrustes
    import numpy as np
    
    # Ensure they have the same dimensions for Procrustes (pad with zeros if necessary)
    rna_pca = mdata.mod[rna_key].obsm['X_pca']
    atac_lsi = mdata.mod[atac_key].obsm['X_lsi']
    
    min_dim = min(rna_pca.shape[1], atac_lsi.shape[1])
    
    # Align ATAC to RNA
    mtx1, mtx2, disparity = procrustes(rna_pca[:, :min_dim], atac_lsi[:, :min_dim])
    
    # Joint representation is simply the aligned target space (or their average)
    mdata.obsm['X_procrustes'] = (mtx1 + mtx2) / 2.0

def run_snf_integration(mdata, rna_key: str = "rna", atac_key: str = "atac") -> None:
    """
    Similarity Network Fusion (SNF) approximation.
    Fuses the kNN adjacency matrices of both modalities into a consensus network.
    """
    print(f"Running Similarity Network Fusion (SNF) across {rna_key} and {atac_key}...")
    import numpy as np
    
    if 'connectivities' not in mdata.mod[rna_key].obsp:
        sc.pp.neighbors(mdata.mod[rna_key])
    if 'connectivities' not in mdata.mod[atac_key].obsp:
        sc.pp.neighbors(mdata.mod[atac_key], use_rep='X_lsi')
        
    W_rna = mdata.mod[rna_key].obsp['connectivities']
    W_atac = mdata.mod[atac_key].obsp['connectivities']
    
    # SNF merges edge weights (simplified iterative updating approximation via averaging)
    # Since they share the same cells, we can directly average their sparse neighbor graphs
    W_joint = (W_rna + W_atac) / 2.0
    
    # Store at mudata level
    if mdata.obsp is None:
        mdata.obsp = {}
    mdata.obsp['snf_connectivities'] = W_joint

def run_joint_harmony(mdata, rna_key: str = "rna", atac_key: str = "atac", batch_key: str = None) -> None:
    """
    Joint Harmony Integration.
    Harmonizes the Concatenated PCA space.
    """
    print(f"Running Joint Harmony Integration across {rna_key} and {atac_key}...")
    try:
        import harmonypy as hm
    except ImportError:
        print("harmonypy not available. Skipping Joint Harmony.")
        return
        
    if 'X_concat_pca' not in mdata.obsm:
        run_concat_pca_integration(mdata, rna_key, atac_key)
        
    # If no batch key is provided, we use Harmony to "align" the modalities themselves 
    # if they were un-paired, but here they are paired multi-ome. 
    # For paired data, batch_key from metadata is expected. We create a dummy if None.
    if batch_key is None or batch_key not in mdata.obs:
        import numpy as np
        mdata.obs['dummy_batch'] = np.random.choice(['batch1', 'batch2'], size=mdata.shape[0])
        batch_key = 'dummy_batch'
        
    print(f"Harmonizing joint space by `{batch_key}`...")
    # harmonypy takes data_mat (n_components, n_cells) naturally, 
    # but the python wrapper `run_harmony` signature internally expects (n_cells, n_components) 
    # and then might transpose. Let's pass the standard Cell x PC.
    ho = hm.run_harmony(mdata.obsm['X_concat_pca'], mdata.obs, batch_key)
    
    # harmonypy typically returns `Z_corr` shaped as (n_components, n_cells).
    # mdata.obsm expects (n_cells, n_components).
    if ho.Z_corr.shape[1] == mdata.shape[0]:
        mdata.obsm['X_joint_harmony'] = ho.Z_corr.T
    else:
        mdata.obsm['X_joint_harmony'] = ho.Z_corr
