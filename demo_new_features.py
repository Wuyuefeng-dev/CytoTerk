"""
Demo: Clonal Streamgraph, Tipping Point UMAP, and updated Doublet Detection.
Generates:
  - demo_figs/lineage_clonal_streamgraph.png
  - demo_figs/tipping_point_umap.png
  - demo_figs/doublet_analysis_white.png  (re-generated with white bg)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import scanpy as sc
sc.settings.n_jobs = 1
import sccytotrek as ct
from sccytotrek.datasets.mock_data import make_mock_scrna


def main():
    os.makedirs("demo_figs", exist_ok=True)
    print("Generating mock data...")
    adata = make_mock_scrna(n_cells=2000, n_genes=2500, n_clusters=5, random_state=42)
    adata.layers["counts"] = adata.X.copy()

    # ── Doublet detection (white bg) ─────────────────────────────────────────────
    print("Running doublet detection...")
    adata = ct.tl.identify_doublets(adata, expected_rate=0.06, n_neighbors=20)

    # ── Preprocessing + UMAP ─────────────────────────────────────────────────────
    print("Preprocessing...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    # Use cluster labels already present in mock data
    if 'cluster' not in adata.obs.columns:
        from sklearn.preprocessing import LabelEncoder
        adata.obs['cluster'] = adata.obs.get('leiden', adata.obs.get('louvain', '0')).astype(str)

    # ── Doublet plot white bg ────────────────────────────────────────────────────
    print("Plotting doublet scores (white background)...")
    ct.tl.plot_doublet_scores(
        adata, use_rep="X_umap",
        save="demo_figs/doublet_analysis.png",
        show=False
    )

    # ── Barcode imputation ───────────────────────────────────────────────────────
    print("Running barcode imputation...")
    adata = ct.lineage.impute_barcodes_knn(adata, barcode_key="barcode", use_rep="X_pca", inplace=False)

    # ── Pseudotime (Monocle3-style) ──────────────────────────────────────────────
    print("Computing pseudotime...")
    root_cluster = adata.obs['cluster'].unique()[0]
    adata = ct.trajectory.run_monocle3(adata, groupby="cluster", root_cluster=root_cluster)

    # ── Tipping Point ────────────────────────────────────────────────────────────
    print("Computing sandpile entropy (tipping points)...")
    tp_result = ct.trajectory.compute_sandpile_entropy(adata, pseudotime_key="monocle3_pseudotime", n_bins=30)

    print("Plotting tipping point UMAP...")
    ct.trajectory.plot_tipping_point_umap(
        adata,
        pseudotime_key="monocle3_pseudotime",
        entropy_key="sandpile_entropy",
        tipping_point_result=tp_result,
        n_top_tipping_cells=100,
        save="demo_figs/tipping_point_umap.png",
        show=False
    )

    # ── Clonal streamgraph ───────────────────────────────────────────────────────
    print("Plotting clonal streamgraph along pseudotime...")
    ct.lineage.plot_clonal_streamgraph(
        adata,
        pseudotime_key="monocle3_pseudotime",
        barcode_key="barcode",
        top_n_clones=8,
        save="demo_figs/lineage_clonal_streamgraph.png",
        show=False
    )

    print("\nAll done! Figures saved:")
    for f in ["demo_figs/doublet_analysis.png",
              "demo_figs/tipping_point_umap.png",
              "demo_figs/lineage_clonal_streamgraph.png"]:
        exists = "✓" if os.path.exists(f) else "✗"
        print(f"  {exists} {f}")


if __name__ == "__main__":
    main()
