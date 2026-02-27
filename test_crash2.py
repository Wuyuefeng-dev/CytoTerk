import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import scanpy as sc
# Force threadpoolctl to 1 thread, which fixes pynndescent mac segfaults
from threadpoolctl import threadpool_limits
with threadpool_limits(limits=1):
    print("Loading PBMC3k")
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()
    print("Normalising")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    print("PCA")
    sc.tl.pca(adata, svd_solver='arpack')
    print("Neighbors")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20, method='umap', use_rep='X_pca')
    print("UMAP")
    sc.tl.umap(adata)
    print("Done!")
