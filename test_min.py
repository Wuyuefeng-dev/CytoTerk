import scanpy as sc
import sccytotrek as ct
import scipy.sparse as sp

sc.settings.n_jobs = 1

adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
if sp.issparse(adata.X):
    adata.X = adata.X.toarray()

# Subsampling 2564 cells down to 1000...
# Run original QC calculation
ct.preprocessing.calculate_qc_metrics(adata)
adata = ct.tools.identify_doublets(adata, expected_rate=0.05)
adata = adata[~adata.obs['predicted_doublet']].copy()
adata = ct.preprocessing.subsample_cells(adata, target_cells=1000)

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)

print("Running PCA and neighbors")
# Run PCA
adata = ct.tools.run_pca_and_neighbors(adata, n_pcs=20, n_neighbors=15, svd_solver='arpack')
print("PCA and neighbors successful")

