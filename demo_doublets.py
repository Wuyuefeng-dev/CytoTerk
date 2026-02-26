import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import scanpy as sc
sc.settings.n_jobs = 1
import numpy as np
import sccytotrek as ct
from sccytotrek.datasets.mock_data import make_mock_scrna


def main():
    print("=== Doublet Detection Demo ===")
    print("Generating mock scRNA-seq data...")
    adata = make_mock_scrna(n_cells=2000, n_genes=2500, n_clusters=5, random_state=42)

    # Save raw counts for doublet detection BEFORE normalisation
    adata.layers["counts"] = adata.X.copy()

    # --- Step 1: Doublet Detection on raw counts ---
    print("Running doublet detection...")
    adata = ct.tl.identify_doublets(adata, expected_rate=0.06, n_neighbors=20)

    # --- Step 2: Preprocessing for UMAP ---
    print("Preprocessing for UMAP embedding...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)

    os.makedirs("demo_figs", exist_ok=True)

    # --- Step 3: Statistical Summary ---
    summary = ct.tl.doublet_statistical_summary(adata)
    print("\n--- Doublet Detection Statistical Summary ---")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    # --- Step 4: Plot ---
    print("\nGenerating doublet analysis plots...")
    ct.tl.plot_doublet_scores(
        adata,
        use_rep="X_umap",
        save="demo_figs/doublet_analysis.png",
        show=False
    )
    print("Done! Figures saved to demo_figs/doublet_analysis.png")


if __name__ == "__main__":
    main()
