import scanpy as sc
import anndata
import warnings

print("Loading PBMC3k")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)

for solver in ['propack', 'lobpcg']:
    print(f"\nTrying SVD Solver: {solver}")
    adata_test = adata.copy()
    try:
        sc.tl.pca(adata_test, svd_solver=solver, n_comps=20)
        print(" -> PCA Success!")
        sc.pp.neighbors(adata_test, n_neighbors=15, n_pcs=20)
        print(" -> Neighbors Success!")
        sc.tl.umap(adata_test)
        print(" -> UMAP Success!")
        break
    except Exception as e:
        print(f" -> Failed: {e}")
