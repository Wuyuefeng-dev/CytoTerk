import os
import sys
import matplotlib.pyplot as plt
import mudata as md
import scanpy as sc

# Ensure the local `sccytotrek` package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import sccytotrek as ct

def main():
    print("="*50)
    print("  scCytoTrek: Multiome (RNA + ATAC) Analysis Demo  ")
    print("="*50)

    # 1. Setup Data Directory
    data_dir = "demo_data"
    fig_dir = "demo_figs"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    multiome_file = os.path.join(data_dir, "sccytotrek_demo_multiome.h5mu")
    
    # Generate data if not exists
    if not os.path.exists(multiome_file):
        print("Generating mock Multiome dataset (RNA + ATAC)...")
        mdata = ct.datasets.make_mock_multiome(n_cells=2000, n_genes=3000, n_peaks=5000)
        mdata.write(multiome_file)
    else:
        print(f"Loading existing Multiome dataset from {multiome_file}...")
        mdata = md.read_h5mu(multiome_file)
        
    print(f"Dataset loaded: {mdata}")
    
    # Setup markdown report
    report_path = os.path.join(fig_dir, "demo_multiome_report.md")
    md_file = open(report_path, "w")
    md_file.write("# scCytoTrek Multiome (RNA+ATAC) Demo Report\n\n")
    md_file.write("This report demonstrates the joint analysis of scRNA-seq and scATAC-seq data using `scCytoTrek` and its underlying `mudata` backend.\n\n")
    
    # 2. RNA Processing
    print("\n--- Processing RNA Modality ---")
    rna = mdata.mod['rna']
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, n_top_genes=2000)
    sc.pp.pca(rna)
    sc.pp.neighbors(rna)
    sc.tl.umap(rna)
    # Cluster RNA
    ct.clustering.run_leiden(rna, resolution=0.5)
    
    # 3. ATAC Processing
    print("\n--- Processing ATAC Modality ---")
    atac = mdata.mod['atac']
    # Simplified TF-IDF / LSI analog for demo purposes
    sc.pp.normalize_total(atac, target_sum=1e4)
    sc.pp.log1p(atac)
    sc.pp.highly_variable_genes(atac, n_top_genes=2000)
    sc.pp.pca(atac, n_comps=50) # Serves as our LSI
    atac.obsm['X_lsi'] = atac.obsm['X_pca'] 
    
    sc.pp.neighbors(atac, use_rep='X_lsi')
    sc.tl.umap(atac)
    ct.clustering.run_leiden(atac, resolution=0.5)
    
    # 4. Joint Multi-modal WNN Integration
    print("\n--- WNN Joint Integration ---")
    md_file.write("## 1. Modality-Specific processing and WNN Integration\n")
    ct.multiome.run_wnn(mdata, rna_key='rna', atac_key='atac')
    md_file.write("`scCytoTrek.multiome.run_wnn` processes both modalities and establishes a joint representation of the Multiome data.\n\n")
    
    # 4.5 Additional Integration Methods
    print("\n--- Running 5 Novel Integration Methods ---")
    md_file.write("## 1.5 Additional Multiome Integration Methods\n")
    ct.multiome.run_cca_integration(mdata)
    ct.multiome.run_concat_pca_integration(mdata)
    ct.multiome.run_procrustes_integration(mdata)
    ct.multiome.run_snf_integration(mdata)
    ct.multiome.run_joint_harmony(mdata)
    
    md_file.write("scCytoTrek also provides wrapper implementations for 5 distinct joint integration modalities:\n")
    md_file.write("1. **CCA (Canonical Correlation Analysis)**: Extracts maximally correlated biological subspaces.\n")
    md_file.write("2. **Concatenated PCA**: Simple early-fusion dimensionality reduction post-L2 normalization.\n")
    md_file.write("3. **Procrustes Alignment**: Rotation, translation, and scaling to overlay cross-modality matrices.\n")
    md_file.write("4. **SNF (Similarity Network Fusion)**: Aggregation of kNN adjacency graphs into a consensus network.\n")
    md_file.write("5. **Joint Harmony**: Batch-correction mechanism applied natively to the multi-modal space.\n\n")

    # 5. Plotting Modalities side-by-side
    print("\n--- Plotting Multiome UMAPs ---")
    ct.pl_adv.set_style()
    
    # Generate Joint UMAP based on one of the integrated spaces (e.g. Concat PCA)
    sc.pp.neighbors(mdata, use_rep='X_concat_pca')
    sc.tl.umap(mdata)
    mdata.obs['leiden'] = rna.obs['leiden']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot RNA UMAP
    sc.pl.umap(rna, color='leiden', ax=axes[0], show=False, title='RNA UMAP (Leiden)')
    # Plot ATAC UMAP
    sc.pl.umap(atac, color='leiden', ax=axes[1], show=False, title='ATAC UMAP (Leiden)')
    # Plot Joint UMAP
    sc.pl.umap(mdata, color='leiden', ax=axes[2], show=False, title='Joint UMAP (ConcatPCA)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "multiome_umaps.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    md_file.write("Side-by-side comparison of the independent embeddings prior to joint-graph resolution, alongside a Joint Embedding (ConcatPCA).\n\n")
    md_file.write("![Multiome UMAPs](multiome_umaps.png)\n\n")
    
    # Copy true clusters to mdata obs if available
    if 'true_cluster' in rna.obs:
        mdata.obs['true_cluster'] = rna.obs['true_cluster']
    
    md_file.write("## 2. Conclusion\n")
    md_file.write("The Multiome wrapper successfully manages both AnnData structures within a unified MuData container, allowing parallel modular workflows and cross-modality anchoring.\n")
    md_file.close()
    
    print("\nMultiome Pipeline completed successfully.")
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    main()
