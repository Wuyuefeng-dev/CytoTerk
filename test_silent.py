import scanpy as sc
import anndata
from sccytotrek.tools.pca_umap import run_pca_and_neighbors

try:
    print("Loading PBMC3k")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    print("Subsampling to 1000 cells")
    sc.pp.subsample(adata, n_obs=1000)

    print("Normalising")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)

    # Let's isolate PCA
    print("Running PCA")
    sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
    print("PCA success")

    # Let's isolate Neighbors
    print("Running Neighbors")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
    print("Neighbors success")

except Exception as e:
    print(f"FAILED: {e}")
