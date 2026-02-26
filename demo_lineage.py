import os
import scanpy as sc
import sccytotrek as ct
from sccytotrek.datasets.mock_data import make_mock_scrna

def main():
    print("Generating demo lineage tracing data...")
    adata = make_mock_scrna(n_cells=2000, n_genes=2500, n_clusters=5, random_state=42)
    
    # Preprocessing
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    
    os.makedirs("demo_figs", exist_ok=True)
    
    print("Running kNN Barcode Imputation...")
    adata = ct.lineage.impute_barcodes_knn(adata, barcode_key="barcode", use_rep="X_pca", inplace=False)
    
    print("Plotting Lineage UMAP...")
    ct.lineage.plot_lineage_umap(
        adata, 
        barcode_key="barcode", 
        status_key="barcode_imputed_status",
        save="demo_figs/lineage_imputation_umap.png",
        show=False
    )
    
    print("Plotting Clone Size Distribution...")
    ct.lineage.plot_clone_size_distribution(
        adata,
        barcode_key="barcode",
        status_key="barcode_imputed_status",
        save="demo_figs/lineage_clone_sizes.png",
        show=False
    )
    
    print("Lineage Tracing Demo Complete. Figures saved to demo_figs/.")

if __name__ == "__main__":
    main()
