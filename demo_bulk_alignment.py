"""
Demo: Bulk RNA-seq Alignment to Single-Cell UMAP Reference
===========================================================
Generates SeuratExtend-style 4-panel figure:
  A — SC UMAP coloured by cluster with bulk samples overlaid as large stars
  B — Per-bulk-sample neighbourhood cluster pie charts
  C — Bulk vs SC cluster correlation heatmap
  D — Marker gene dot plot (expression size × fraction)

Output: demo_figs/bulk_alignment.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
sc.settings.n_jobs = 1

from sccytotrek.datasets.mock_data import make_mock_scrna
from sccytotrek.integration.bulk import project_bulk_to_umap, plot_bulk_alignment


def make_mock_bulk(adata_sc, n_bulk_samples: int = 6, random_state: int = 42):
    """
    Simulate bulk RNA-seq by averaging SC expression within clusters,
    with added biological + technical noise.
    Creates samples representing a mix of 1-2 dominant clusters each.
    """
    rng = np.random.default_rng(random_state)
    common_genes = adata_sc.var_names.tolist()
    clusters = adata_sc.obs['cluster'].unique()

    X_sc = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else adata_sc.X
    profiles = []
    sample_names = []

    for i in range(n_bulk_samples):
        # Each sample is a mixture of 1-2 clusters
        n_cl = rng.integers(1, 3)
        chosen = rng.choice(sorted(clusters), size=n_cl, replace=False)
        weights = rng.dirichlet(np.ones(n_cl))

        expr = np.zeros(len(common_genes))
        for cl, w in zip(chosen, weights):
            mask = adata_sc.obs['cluster'].values == cl
            cl_mean = X_sc[mask].mean(axis=0)
            expr += w * cl_mean

        # Add Gaussian noise and library-size variation
        noise = rng.normal(0, 0.3 * expr.std(), expr.shape)
        scale = rng.uniform(0.8, 1.2)
        expr  = np.clip((expr + noise) * scale, 0, None)

        profiles.append(expr)
        label = "+".join([f"C{c}" for c in chosen])
        sample_names.append(f"Bulk_{i+1}_{label}")

    X_bulk = np.vstack(profiles).astype(np.float32)
    obs = pd.DataFrame({'sample': sample_names}, index=sample_names)
    adata_bulk = ad.AnnData(
        X=X_bulk,
        obs=obs,
        var=pd.DataFrame(index=common_genes)
    )
    return adata_bulk


def main():
    os.makedirs("demo_figs", exist_ok=True)

    # 1. Generate and preprocess single-cell reference
    print("Generating mock scRNA-seq reference...")
    adata_sc = make_mock_scrna(n_cells=1500, n_genes=2000, n_clusters=5, random_state=42)
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.highly_variable_genes(adata_sc, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.pca(adata_sc, n_comps=30)
    sc.pp.neighbors(adata_sc, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_sc)
    print(f"  SC reference: {adata_sc.shape}, clusters: {adata_sc.obs['cluster'].nunique()}")

    # 2. Generate mock bulk samples
    print("Generating mock bulk RNA-seq samples...")
    adata_bulk = make_mock_bulk(adata_sc, n_bulk_samples=8, random_state=42)
    print(f"  Bulk: {adata_bulk.shape}")

    # 3. Project bulk onto SC UMAP
    print("Projecting bulk onto SC UMAP (true PCA-loading projection)...")
    adata_bulk = project_bulk_to_umap(adata_sc, adata_bulk, n_neighbors=15)
    print(f"  Bulk UMAP coords shape: {adata_bulk.obsm['X_umap'].shape}")

    # 4. SeuratExtend-style visualization
    print("Generating SeuratExtend-style alignment figure...")
    plot_bulk_alignment(
        adata_sc, adata_bulk,
        cluster_key="cluster",
        bulk_label_key="sample",
        save="demo_figs/bulk_alignment.png",
        show=False
    )

    print("\nDone!")
    print("  ✓ demo_figs/bulk_alignment.png" if os.path.exists("demo_figs/bulk_alignment.png") else "  ✗ figure not saved")


if __name__ == "__main__":
    main()
