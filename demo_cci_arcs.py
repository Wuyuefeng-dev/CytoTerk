import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import scanpy as sc
sc.settings.n_jobs = 1
import sccytotrek as ct
import sccytotrek as ct
from sccytotrek.datasets.mock_data import make_mock_scrna

def main():
    print("Generating mock scRNA-seq data...")
    adata = make_mock_scrna(n_cells=1000, n_genes=2000, n_clusters=4, random_state=42)
    
    # Preprocess
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    adata.obs["cluster"] = kmeans.fit_predict(adata.obsm["X_pca"]).astype(str)
    
    # Create a mock CCI results dataframe
    # We will use the generated KMeans clusters ('0', '1', '2', '3')
    cci_data = [
        {"sender": "0", "receiver": "1", "score": 2.5, "pvalue": 0.01},
        {"sender": "0", "receiver": "2", "score": 1.8, "pvalue": 0.04},
        {"sender": "1", "receiver": "3", "score": 3.0, "pvalue": 0.001},
        {"sender": "2", "receiver": "3", "score": 2.2, "pvalue": 0.02},
        {"sender": "3", "receiver": "0", "score": 1.5, "pvalue": 0.05},
        {"sender": "2", "receiver": "1", "score": 3.5, "pvalue": 0.005},
    ]
    cci_df = pd.DataFrame(cci_data)
    
    os.makedirs("demo_figs", exist_ok=True)
    out_file = "demo_figs/cci_umap_arcs.png"
    
    print("Generating CCI UMAP Arc Plot...")
    ct.interaction.plot_cell2cell_umap(
        adata, 
        lr_results=cci_df,
        groupby="cluster",
        score_col="score",
        save=out_file,
        show=False
    )
    
    print(f"Done! Figure saved to {out_file}")

if __name__ == "__main__":
    main()
