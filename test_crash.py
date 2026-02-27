import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import scanpy as sc
import numpy as np

print("Loading PBMC3k")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
print("Data type:", adata.X.dtype)

# adata.X = adata.X.astype(np.float32)

print("Normalising")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)

print("PCA")
sc.tl.pca(adata, svd_solver='arpack')
print("Neighbors")
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
print("UMAP")
sc.tl.umap(adata)
print("Done!")
