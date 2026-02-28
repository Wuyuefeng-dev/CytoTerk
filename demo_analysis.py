"""
scCytoTrek Comprehensive Demo Analysis Script.

This script demonstrates common scRNA-seq workflows utilizing scCytoTrek
and generates figures to be used in the documentation.
"""

import os
import scanpy as sc

# Prevent macOS Apple Silicon (ARM64) segmentation faults and save memory/power 
# across all systems (Linux/Windows/Mac) by limiting pynndescent/arpack thread spawning
sc.settings.n_jobs = 1

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import scCytoTrek modules
import sccytotrek as ct

def main():
    # 1. Setup output directories
    fig_dir = "demo_figs"
    os.makedirs(fig_dir, exist_ok=True)
    sc.settings.figdir = fig_dir
    
    # Open a markdown file to log the demo results
    md_file = open("demo_figs/demo_report.md", "w")
    md_file.write("# scCytoTrek Demonstration Pipeline\n\n")
    
    # 2. Load demo data
    print("\n---------------------------------------------------------")
    print("Please choose a dataset for the demonstration pipeline:")
    print("1: scanpy.datasets.pbmc3k() (Real PBMC data, ~2.7k cells)")
    print("2: Generated simulated demo data (Mock data)")
    print("---------------------------------------------------------")
    try:
        choice = input("Enter 1 or 2 [default: 1]: ").strip()
    except EOFError:
        choice = '1'
    
    md_file.write("## 1. Data Loading\n")
    if choice == '2':
        if not os.path.exists("demo_data/sccytotrek_demo_scrna.h5ad"):
            print("Demo data not found. Please run 'python generate_demo_data.py' first.")
            return
        adata = sc.read_h5ad("demo_data/sccytotrek_demo_scrna.h5ad")
        data_source = "mock"
        md_file.write(f"Loaded generated mock dataset: `{adata}`\n\n")
    else:
        print("Downloading/Loading scanpy.datasets.pbmc3k()...")
        adata = sc.datasets.pbmc3k()
        adata.var_names_make_unique()
        # Ensure matrix is dense to prevent sparse PCA segmentation faults on Mac ARM64/OpenBLAS
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            adata.X = adata.X.toarray()
            
        data_source = "pbmc3k"
        md_file.write(f"Loaded `scanpy.datasets.pbmc3k()` dataset: `{adata}`\n\n")
    print(adata)
    
    # 3. Preprocessing (Doublets, Subsample, Impute, Normalize)
    print("\n--- Running Preprocessing ---")
    md_file.write("## 2. Preprocessing\n")
    
    # Run original QC calculation
    ct.preprocessing.calculate_qc_metrics(adata)
    ct.preprocessing.plot_qc_violins(adata, save_path=os.path.join(fig_dir, "qc_violins.png"))
    md_file.write(f"We begin by evaluating total molecular distributions (UMI Depth) to remove empty droplets or multi-nucleated aberrations using QC Violin Plots.\n\n")
    md_file.write(f"![QC Violins](qc_violins.png)\n\n")
    
    # Custom Doublet ID
    adata = ct.tools.identify_doublets(adata, expected_rate=0.05)
    
    # --- Doublet Plotting Before/After ---
    if 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='predicted_doublet', show=False, title="Before Doublet Removal")
        plt.savefig(os.path.join(fig_dir, "doublets_before.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
    n_before = adata.shape[0]
    adata = adata[~adata.obs['predicted_doublet']].copy()
    n_after = adata.shape[0]
    
    if 'X_umap' in adata.obsm:
        sc.pl.umap(adata, color='true_cluster', show=False, title=f"After Doublet Removal (n={n_after})")
        plt.savefig(os.path.join(fig_dir, "doublets_after.png"), bbox_inches='tight', dpi=150)
        plt.close()
    
    # Subsample (for demo purposes)
    adata = ct.preprocessing.subsample_cells(adata, target_cells=1000)
    
    # Standard Scanpy normalization block before imputation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)

    md_file.write(f"- **Doublet Detection:** Custom kNN-based doublet finding. Identified and removed {n_before - n_after} doublets.\n")
    md_file.write("  ![Before Doublet Removal](doublets_before.png)\n")
    md_file.write("  ![After Doublet Removal](doublets_after.png)\n\n")
    md_file.write("- **Subsampling:** Subsampled remaining cells for demonstration purposes.\n")
    md_file.write("- **Normalization & HVG:** Standard log1p and 1000 highly variable genes selected.\n\n")
    
    # 4. Dimensionality Reduction & Clustering
    print("\n--- Running Dimensionality Reduction ---")
    md_file.write("## 3. Dimensionality Reduction\n")
    adata = ct.tools.run_pca_and_neighbors(adata, n_pcs=20, n_neighbors=15)
    
    # Custom Gene Imputation (smoothing dropout)
    # adata = ct.preprocessing.impute_knn_smoothing(adata, n_neighbors=15, use_rep='X_pca')
    
    adata = ct.tools.run_umap_and_cluster(adata, resolution=0.5)
    fig = ct.plotting.dim_plot(adata, color='leiden_0.5', title="Leiden Clusters (UMAP)", show=False)
    fig.figure.savefig(os.path.join(fig_dir, "umap_clusters.png"), bbox_inches='tight', dpi=150)
    plt.close()

    md_file.write("Standard PCA, Neighborhood Graph, and UMAP computation. Leiden clustering (resolution=0.5) is used as the base labeling.\n\n")
    md_file.write("![UMAP Leiden Clusters](umap_clusters.png)\n\n")
    
    # 4.5 Alternative Clustering Methods
    print("\n--- Running Alternative Clustering Methods ---")
    md_file.write("## 3.5 Alternative Clustering Methods\n")
    md_file.write("scCytoTrek supports several standard and specialized clustering algorithms. Below is an overview of each method's pros, cons, and ideal working situations:\n\n")

    # 1. K-Means
    print("  -> K-Means")
    try:
        adata = ct.clustering.run_kmeans(adata, n_clusters=5)
        fig = ct.plotting.dim_plot(adata, color='kmeans', title="K-Means Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "kmeans_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 1. K-Means Clustering\n")
        md_file.write("- **Pros:** Extremely fast and scalable to massive single-cell datasets. Simple to interpret.\n")
        md_file.write("- **Cons:** Assumes spherical clusters of similar size. Fails on complex, elongated trajectory manifolds.\n")
        md_file.write("- **Working Situation:** Best used as an initial rapid quantization step or when cell types are highly distinct and globular (e.g., peripheral blood mononuclear cells where major lineages are highly separated).\n\n")
        md_file.write("![K-Means](kmeans_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"K-Means failed: {e}")

    # 2. Agglomerative (Hierarchical)
    print("  -> Agglomerative")
    try:
        adata = ct.clustering.run_agglomerative(adata, n_clusters=5)
        fig = ct.plotting.dim_plot(adata, color='agglomerative', title="Agglomerative Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "agglomerative_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 2. Agglomerative (Hierarchical) Clustering\n")
        md_file.write("- **Pros:** Does not require a pre-specified number of clusters (if a distance threshold is used). Captures hierarchical relationships between cell types (e.g., T-cell subtypes grouping under a pan-T lineage).\n")
        md_file.write("- **Cons:** High memory and computational complexity ($O(N^2)$ or $O(N^3)$), making it very slow for datasets > 10,000 cells without subsampling.\n")
        md_file.write("- **Working Situation:** Ideal for smaller scRNA-seq datasets or bulk RNA-seq where establishing an evolutionary or developmental relationship between the populations is critical.\n\n")
        md_file.write("![Agglomerative](agglomerative_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"Agglomerative failed: {e}")

    # 3. Spectral Clustering
    print("  -> Spectral")
    try:
        adata = ct.clustering.run_spectral(adata, n_clusters=5)
        fig = ct.plotting.dim_plot(adata, color='spectral', title="Spectral Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "spectral_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 3. Spectral Clustering\n")
        md_file.write("- **Pros:** Excellent at identifying non-convex, arbitrarily shaped clusters. Mathematically similar to graph-based approaches like Louvain.\n")
        md_file.write("- **Cons:** Computationally expensive due to eigenvalue decomposition. Can be sensitive to the choice of affinity matrix parameters.\n")
        md_file.write("- **Working Situation:** Useful when cells form dense topological manifolds that are non-globular, such as interconnected developmental branches where standard K-Means fails.\n\n")
        md_file.write("![Spectral](spectral_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"Spectral failed: {e}")

    # 4. Gaussian Mixture Models (GMM)
    print("  -> GMM")
    try:
        adata = ct.clustering.run_gmm(adata, n_components=5)
        fig = ct.plotting.dim_plot(adata, color='gmm', title="GMM Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "gmm_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 4. Gaussian Mixture Models (GMM)\n")
        md_file.write("- **Pros:** Provides soft assignments (probabilities) for cell membership, reflecting biological ambiguity. Can model clusters with different variances.\n")
        md_file.write("- **Cons:** Prone to local maxima. Can be unstable in highly dimensional spaces if not sufficiently reduced via PCA.\n")
        md_file.write("- **Working Situation:** Excellent for modeling transitional states in continuous differentiation processes where a cell might partially belong to two distinct states.\n\n")
        md_file.write("![GMM](gmm_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"GMM failed: {e}")

    # 5. DBSCAN
    print("  -> DBSCAN")
    try:
        adata = ct.clustering.run_dbscan(adata, eps=2.0, min_samples=10)
        fig = ct.plotting.dim_plot(adata, color='dbscan', title="DBSCAN Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "dbscan_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 5. DBSCAN (Density-Based)\n")
        md_file.write("- **Pros:** Does not force every cell into a cluster (can robustly identify noise/outliers). Excellent at finding clusters of arbitrary shape based on local density.\n")
        md_file.write("- **Cons:** Extremely sensitive to the `eps` (distance) parameter. Fails if the dataset has vastly different densities across the manifold.\n")
        md_file.write("- **Working Situation:** Best for filtering out anomalous/outlier cells or processing spatial transcriptomics where distinct anatomical regions correspond to dense clusters separated by empty space.\n\n")
        md_file.write("![DBSCAN](dbscan_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"DBSCAN failed: {e}")

    # 6. Louvain
    print("  -> Louvain")
    try:
        adata = ct.clustering.run_louvain(adata, resolution=0.5)
        fig = ct.plotting.dim_plot(adata, color='louvain', title="Louvain Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "louvain_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 6. Louvain Clustering\n")
        md_file.write("- **Pros:** Conventional, widely adopted graph-based community detection. Extremely fast and identifies non-linear structural communities effectively.\n")
        md_file.write("- **Cons:** Resolution limit problem (can miss small sub-clusters in large datasets).\n")
        md_file.write("- **Working Situation:** The standard gold standard for initial general cell-type clustering in most scRNA-seq workflows.\n\n")
        md_file.write("![Louvain](louvain_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"Louvain failed: {e}")

    # 7. Leiden
    print("  -> Leiden")
    try:
        adata = ct.clustering.run_leiden(adata, resolution=0.5)
        fig = ct.plotting.dim_plot(adata, color='leiden', title="Leiden Clustering", show=False)
        fig.figure.savefig(os.path.join(fig_dir, "leiden_clusters.png"), bbox_inches='tight', dpi=150)
        plt.close()
        md_file.write("### 7. Leiden Clustering\n")
        md_file.write("- **Pros:** An improvement over Louvain. Guarantees well-connected communities and avoids badly connected sub-components within clusters.\n")
        md_file.write("- **Cons:** Slightly slower than Louvain depending on implementation.\n")
        md_file.write("- **Working Situation:** Highly recommended modern alternative to Louvain for nuanced, robust sub-clustering operations.\n\n")
        md_file.write("![Leiden](leiden_clusters.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"Leiden failed: {e}")
        
    # Trajectories
    print("\n--- Trajectory Inference Methods ---")
    md_file.write("### 3. Trajectory Inference & Streamgraphs\n")
    try:
        root_cluster = '0'
        # 3.1 Monocle3
        adata = ct.trajectory.run_monocle3(adata, groupby='leiden_0.5', root_cluster=root_cluster)
        fig = ct.pl_monocle.plot_monocle3_trajectory(adata, group_key='leiden_0.5', color_by='monocle3_pseudotime', title="Monocle3 Principal Graph")
        fig.savefig(os.path.join(fig_dir, "trajectory_monocle3.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        # 3.2 Streamgraph
        fig = ct.pl_monocle.plot_streamgraph(adata, time_key='monocle3_pseudotime', group_key='leiden_0.5', title="Cell Density over Monocle3 Pseudotime")
        fig.savefig(os.path.join(fig_dir, "trajectory_streamgraph.png"), bbox_inches='tight', dpi=150)
        plt.close()

        # 3.3 Additional Pseudotime Methods
        adata = ct.trajectory.run_slingshot_pseudotime(adata, groupby='leiden_0.5', root_cluster=root_cluster)
        adata = ct.trajectory.run_palantir_pseudotime(adata, root_cell=adata.obs_names[0])
        adata = ct.trajectory.run_cellrank_pseudotime(adata, root_cell=adata.obs_names[0])
        
        # Plot Side-by-Side to compare pseudotime projections
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        scat1 = axes[0].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c=adata.obs['slingshot_pseudotime'], cmap='magma', s=15, alpha=0.8)
        axes[0].set_title("Slingshot Pseudotime (Approximated)")
        plt.colorbar(scat1, ax=axes[0])
        
        scat2 = axes[1].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c=adata.obs['palantir_pseudotime'], cmap='plasma', s=15, alpha=0.8)
        axes[1].set_title("Palantir Pseudotime (Approximated)")
        plt.colorbar(scat2, ax=axes[1])
        
        scat3 = axes[2].scatter(adata.obsm['X_umap'][:, 0], adata.obsm['X_umap'][:, 1], c=adata.obs['cellrank_pseudotime'], cmap='viridis', s=15, alpha=0.8)
        axes[2].set_title("CellRank Pseudotime (Approximated)")
        plt.colorbar(scat3, ax=axes[2])
        
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "trajectory_comparison.png"), bbox_inches='tight', dpi=150)
        plt.close()

        md_file.write("Implemented a suite of trajectory inference engines including **Monocle3**, **Slingshot**, **Palantir**, and **CellRank** approximations to trace cellular development.\n\n")
        md_file.write("![Monocle3 Trajectory](trajectory_monocle3.png)\n\n")
        md_file.write("We also provide a **Streamgraph** to visualize the continuous transitions of cell proportions across the pseudotime axis.\n\n")
        md_file.write("![Streamgraph](trajectory_streamgraph.png)\n\n")
        md_file.write("Comparison of different pseudotime projections on UMAP:\n\n")
        md_file.write("![Trajectory Comparison](trajectory_comparison.png)\n\n")
        md_file.flush()
    except Exception as e:
        print(f"Trajectory analysis failed: {e}")

    # 5. Cell Type Identity
    print("\n--- Assigning Cell Types ---")
    md_file.write("## 4. Cell Type Identification\n")
    if data_source == "pbmc3k":
        marker_dict = {
            'T-cell': ['CD3D', 'CD3E', 'IL32'],
            'B-cell': ['CD79A', 'MS4A1'],
            'Monocyte': ['FCGR3A', 'LZTFL1'],
            'NK-cell': ['GNLY', 'NKG7']
        }
    else:
        marker_dict = {
            'Malignant': [adata.var_names[0], adata.var_names[1]], # mock markers
            'T-cell': [adata.var_names[2], adata.var_names[3]]
        }
    adata = ct.tools.score_cell_types(adata, marker_dict=marker_dict, groupby='leiden_0.5')
    
    fig = ct.plotting.dim_plot(adata, color='cell_type_prediction', title="Cell Type Mapping", show=False)
    fig.figure.savefig(os.path.join(fig_dir, "cell_types.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    md_file.write("Assigned cell types based on simple gene module scoring of known markers over cluster consensus.\n\n")
    md_file.write("![Cell Types](cell_types.png)\n\n")
    
    # 6. Dropout-Adjusted Differential Expression
    print("\n--- Dropout-Adjusted Differential Expression ---")
    md_file.write("## 5. Differential Expression\n")
    groups = adata.obs['leiden_0.5'].unique()
    if len(groups) >= 2:
        try:
            de_res = ct.tools.dropout_adjusted_de(
                adata, group_key='leiden_0.5', group1=groups[0], group2=groups[1], 
                out_csv=os.path.join(fig_dir, "differential_expression.csv")
            )
            print("Successfully extracted differentially expressed genes.")
            
            if not de_res.empty:
                # Generate Volcano Plot
                fig = ct.plotting.plot_volcano(
                    de_res, 
                    title=f"Volcano: {groups[0]} vs {groups[1]}",
                    lfc_thresh=1.0, 
                    pval_thresh=0.05,
                    show=False,
                    save=os.path.join(fig_dir, "de_volcano.png")
                )
            
            md_file.write(f"Conducted Dropout-Adjusted Differential Expression between `{groups[0]}` and `{groups[1]}`. ")
            md_file.write("Results outputted to `differential_expression.csv`.\n\n")
            md_file.write("![Volcano Plot](de_volcano.png)\n\n")
        except Exception as e:
            print(f"DE skipped: {e}")
            md_file.write(f"> Differential Expression failed/skipped: {e}\n\n")

    # 7. Custom TF Enrichment (No decoupler)
    # 8. Trajectory Inference & Tipping Points
    print("\n--- Trajectory Inference (Tipping Points) ---")
    md_file.write("## 7. Trajectory Inference & Sandpile Entropy\n")
    try:
        root_cell = adata.obs_names[0]
        try:
            adata = ct.trajectory.run_trajectory_inference(adata, root_cell=root_cell, groupby='leiden_0.5')
            pseudo_key = 'dpt_pseudotime'
        except Exception as e:
            print(f"Standard DPT skipped ({e}). Falling back to Palantir pseudotime for Sandpile Entropy.")
            pseudo_key = 'palantir_pseudotime'
            
        # Sandpile Entropy Tipping Point Detection
        tipping_res = ct.trajectory.compute_sandpile_entropy(adata, pseudotime_key=pseudo_key, n_bins=20)
        tipping_bin = tipping_res['tipping_point_bin']
        max_entropy = tipping_res['entropy'][tipping_bin]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(tipping_res['bins'], tipping_res['entropy'], marker='o', color='darkorange')
        ax.axvline(tipping_bin, color='red', linestyle='--', label=f'Tipping Point (Bin {tipping_bin})')
        ax.set_title("Sandpile Model: Network Entropy along Trajectory")
        ax.set_xlabel("Pseudotime Bins")
        ax.set_ylabel("Shannon Entropy")
        ax.legend()
        fig.savefig(os.path.join(fig_dir, "sandpile_entropy_trajectory.png"), bbox_inches='tight', dpi=150)
        plt.close()
        
        md_file.write(f"Computed trajectory and Sandpile Network Entropy. Tipping point found at bin {tipping_bin} with entropy {max_entropy:.3f}.\n\n")
        md_file.write(f"![Sandpile Entropy Line Graph](sandpile_entropy_trajectory.png)\n\n")

        # Project Sandpile Entropy onto UMAP space
        if 'X_umap' in adata.obsm:
            sc.pl.umap(adata, color='sandpile_entropy', cmap='magma', show=False, title='Sandpile Entropy (Tipping Points)')
            plt.savefig(os.path.join(fig_dir, "sandpile_umap.png"), bbox_inches='tight', dpi=150)
            plt.close()
            md_file.write(f"![Sandpile Entropy UMAP](sandpile_umap.png)\n\n")
            
        # Plot Key Genes Driving the Tipping Point
        if 'tipping_genes' in tipping_res:
            ct.trajectory.plot_tipping_genes(tipping_res['tipping_genes'], top_n=20, save_path=os.path.join(fig_dir, "tipping_genes_barplot.png"))
            md_file.write(f"Identified the critical genes driving the network configuration at the tipping point:\n\n")
            md_file.write(f"![Tipping Genes](tipping_genes_barplot.png)\n\n")

        # Find genes correlated with pseudotime ordering
        ordering_genes = ct.trajectory.find_ordering_genes(adata, pseudotime_key=pseudo_key, top_n=10)
        ordering_genes.to_csv(os.path.join(fig_dir, "ordering_effect_genes.csv"), index=False)
        print("Trajectory Ordering Genes extracted to CSV.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Trajectory analysis skipped entirely: {e}")

    # 9. Custom TF Enrichment
    print("\n--- Custom TF Enrichment ---")
    md_file.write("## 6. Transcription Factor Enrichment\n")
    # Build a relevant TF network using genes that actually exist in the data (PBMC specific)
    tf_pairs_list = [
        ('SPI1', 'LYZ', 1.0),
        ('SPI1', 'CD14', 0.8),
        ('STAT1', 'ISG15', 1.0),
        ('STAT1', 'CXCL10', 0.9),
        ('PAX5', 'CD79A', 1.0),
        ('PAX5', 'MS4A1', 0.8)
    ]
    
    valid_pairs = [(tf, tgt, w) for tf, tgt, w in tf_pairs_list if tf in adata.var_names and tgt in adata.var_names]
    
    # Fallback to variable genes if the specific ones were filtered out by HVG
    if not valid_pairs:
        valid_genes = adata.var_names[:5].tolist()
        valid_pairs = [
            (valid_genes[0], valid_genes[2], 1.0),
            (valid_genes[0], valid_genes[3], 0.8),
            (valid_genes[1], valid_genes[2], -0.5),
            (valid_genes[1], valid_genes[3], 0.9)
        ]
        
    tfs, tgts, wts = zip(*valid_pairs)
    tf_df = pd.DataFrame({
        'tf': tfs,
        'target': tgts,
        'weight': wts
    })
    
    if len(tf_df) >= 1:
        primary_tf = tf_df['tf'].iloc[0]
        # Use a zero threshold to ensure the mock genes pass
        adata = ct.grn.run_tf_enrichment(adata, tf_network=tf_df, source_col='tf', target_col='target', min_expr_fraction=0.0)
        
        # Plot TF enrichment UMAP
        if f"tf_score_{primary_tf}" in adata.obs:
            sc.pl.umap(adata, color=f"tf_score_{primary_tf}", cmap='viridis', show=False, title=f"TF Enrichment: {primary_tf}")
            plt.savefig(os.path.join(fig_dir, "tf_enrichment_umap.png"), bbox_inches='tight', dpi=150)
            plt.close()
            md_file.write(f"Evaluated transcription factor activities using expression-weighted network scoring for `{primary_tf}`.\n\n")
            md_file.write(f"![TF Enrichment UMAP](tf_enrichment_umap.png)\n\n")
            
        # Plot TF enrichment Dotplot across clusters
        ct.grn.plot_tf_dotplot(adata, groupby='leiden_0.5', save_path=os.path.join(fig_dir, "tf_enrichment_dotplot.png"))
        md_file.write(f"Visualized cluster-specific TF enrichment scores:\n\n")
        md_file.write(f"![TF Enrichment Dotplot](tf_enrichment_dotplot.png)\n\n")
    
    # 10. GSEApy Pathway Analysis & GSVA UMAP
    print("\n--- GSEApy ssGSEA & GSVA Visualization ---")
    md_file.write("## 8. Pathway Analysis (GSVA)\n")
    if ct.pathway.enrichment.GSEAPY_AVAILABLE:
        # Create a tiny mock gene set dictionary for testing ssGSEA locally using real adata genes
        valid_genes_b = adata.var_names[5:10].tolist()
        mock_gmt = {
            "Mock_Pathway_1": valid_genes_b[:3],
            "Mock_Pathway_2": [valid_genes_b[3], valid_genes_b[4]]
        }
        
        try:
            nes_df = ct.pathway.run_ssgsea(adata, gene_sets=mock_gmt, outdir=fig_dir)
            print("ssGSEA complete. Computed for mock pathways.")
            
            # Plot GSVA Enrichment on UMAP
            fig = ct.pathway.plot_gsva_umap(adata, pathway_name='Mock_Pathway_1', cmap='viridis')
            if fig is not None:
                fig.savefig(os.path.join(fig_dir, "gsva_pathway_a_umap.png"), bbox_inches='tight', dpi=150)
                plt.close()
                md_file.write("Single-cell GSVA enrichment scores projected onto the UMAP.\n\n")
                md_file.write("![GSVA Enrichment UMAP](gsva_pathway_a_umap.png)\n\n")
                md_file.flush()
        except Exception as e:
            print(f"GSEApy mapping failed: {e}")
            
    # 11. CellPhoneDB Algorithm & Cell2Cell Plot
    print("\n--- CellPhoneDB Ligand-Receptor Scoring ---")
    md_file.write("## 9. Cell-Cell Communication (CellPhoneDB Algorithm & Cell2Cell Plot)\n")
    try:
        # Instead of arbitrary first genes, we supply real immune Ligand-Receptor pairs
        # relevant to PBMC3k (e.g. Antigen Presentation, T-cell activation, Chemokines)
        lr_pairs_list = [
            ('HLA-DRA', 'CD4'),
            ('B2M', 'CD3E'),
            ('CD86', 'CD28'),
            ('CCL5', 'CCR5'),
            ('IL32', 'CD4'),
            ('HLA-DPB1', 'CD4'),
            ('CD74', 'CD44')
        ]
        
        # Filter pairs to only those where both genes exist in the PBMC dataset
        valid_pairs = [(l, r) for l, r in lr_pairs_list if l in adata.var_names and r in adata.var_names]
        
        if valid_pairs:
            ligands, receptors = zip(*valid_pairs)
            lr_df = pd.DataFrame({
                'ligand': ligands,
                'receptor': receptors
            })
            
            # Run custom CellPhoneDB
            ccc_res = ct.interaction.run_cellphonedb_scoring(adata, lr_pairs=lr_df, group_key='leiden_0.5', n_perms=100)
        
        # Plot Cell2Cell style
        if not ccc_res.empty:
            fig = ct.interaction.plot_cell2cell_dotplot(ccc_res, top_n=20)
            fig.savefig(os.path.join(fig_dir, "cell2cell_interaction.png"), bbox_inches='tight', dpi=150)
            plt.close()
            
            try:
                ct.interaction.plot_cell2cell_umap(adata, ccc_res, group_key='leiden_0.5', top_n=10, save=os.path.join(fig_dir, "cci_umap_arcs.png"), show=False)
            except Exception as e:
                print(f"Arc plotting failed: {e}")
                
            md_file.write("We evaluate extracellular communication by running non-parametric label permutations against biologically relevant immune Ligand-Receptor pairs (e.g., HLA-DRA to CD4, CCL5 to CCR5) on the PBMC data.\n\n")
            md_file.write("![Cell2Cell Interaction](cell2cell_interaction.png)\n\n")
            md_file.write("![Cell2Cell UMAP Arcs](cci_umap_arcs.png)\n\n")
            md_file.flush()
    except Exception as e:
        print(f"Interaction analysis skipped: {e}")

    md_file.close()
    print("\n✅ Demo execution complete. Figures and report saved to 'demo_figs/'.")

if __name__ == "__main__":
    main()
