import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text = """\
# scCytoTrek Demonstration Pipeline

This notebook runs the full demonstration pipeline of the scCytoTrek package using standard Scanpy datasets (like pbmc3k) and outputs the results step by step. It is fully compatible with Linux, Windows, and macOS.
"""

code_imports = """\
import os

import scanpy as sc

# Prevent macOS Apple Silicon (ARM64) segmentation faults and save memory/power 
# across all systems (Linux/Windows/Mac) by limiting pynndescent/arpack thread spawning
sc.settings.n_jobs = 1

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add src to Python Path to ensure local module resolution
import sys
sys.path.append('src')

# Import scCytoTrek modules
import sccytotrek as ct

fig_dir = "demo_figs"
os.makedirs(fig_dir, exist_ok=True)
sc.settings.figdir = fig_dir
"""

text_load = """\
## 1. Data Loading

We will load the standard 3k PBMC dataset from 10x Genomics via `scanpy.datasets.pbmc3k()`.
"""

code_load = """\
print("Downloading/Loading scanpy.datasets.pbmc3k()...")
adata = sc.datasets.pbmc3k()
adata.var_names_make_unique()
# Ensure matrix is dense to prevent sparse PCA segmentation faults on Mac ARM64/OpenBLAS
import scipy.sparse as sp
if sp.issparse(adata.X):
    adata.X = adata.X.toarray()
print(adata)
"""

text_prep = """\
## 2. Preprocessing & Quality Control

We calculate standard QC metrics and visualize them. After identifying and removing doublets, the cells are subsampled (for speed/demonstration), normalized, and highly variable genes are isolated.
"""

code_prep = """\
# Run original QC calculation
ct.preprocessing.calculate_qc_metrics(adata)
ct.preprocessing.plot_qc_violins(adata, save_path=os.path.join(fig_dir, "qc_violins.png"))

# Custom Doublet ID
adata = ct.tools.identify_doublets(adata, expected_rate=0.04)

# Filter out doublets
adata = adata[~adata.obs['predicted_doublet']].copy()

# Subsample (for demo purposes)
adata = ct.preprocessing.subsample_cells(adata, target_cells=1500)

# Standard Scanpy normalization block
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=1000)

print(adata)
"""

text_dim = """\
## 3. Dimensionality Reduction & Clustering

We perform PCA, build the spatial neighborhood graph, project the UMAP, and then run Leiden clustering.
"""

code_dim = """\
import warnings
warnings.filterwarnings("ignore")

sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20)
sc.tl.umap(adata)

sc.tl.leiden(adata, resolution=0.5, key_added='leiden_0.5')
fig = ct.plotting.dim_plot(adata, color='leiden_0.5', title="Leiden Clusters (UMAP)", show=True)
"""

text_types = """\
## 4. Cell Type Assignment

We score these clusters against a known marker dictionary for PBMCs.
"""

code_types = """\
marker_dict = {
    'T-cell': ['CD3D', 'CD3E', 'IL32'],
    'B-cell': ['CD79A', 'MS4A1'],
    'Monocyte': ['FCGR3A', 'LZTFL1'],
    'NK-cell': ['GNLY', 'NKG7']
}
adata = ct.tools.score_cell_types(adata, marker_dict=marker_dict, groupby='leiden_0.5')

fig = ct.plotting.dim_plot(adata, color='cell_type_prediction', title="Cell Type Mapping", show=True)
"""

text_traj = """\
## 5. Trajectory Inference (Sandpile Network Entropy & Monocle3)

We utilize advanced trajectory tools, including Sandpile entropy computations for critical state tipping points.
"""

code_traj = """\
root_cell = adata.obs_names[0]
sc.tl.dpt(adata) # Simple Pseudotime proxy
# Sandpile Entropy Tipping Point Detection
try:
    tipping_res = ct.trajectory.compute_sandpile_entropy(adata, pseudotime_key='dpt_pseudotime', n_bins=20)
    tipping_bin = tipping_res['tipping_point_bin']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tipping_res['bins'], tipping_res['entropy'], marker='o', color='darkorange')
    ax.axvline(tipping_bin, color='red', linestyle='--', label=f'Tipping Point (Bin {tipping_bin})')
    ax.set_title("Sandpile Model: Network Entropy along Trajectory")
    plt.show()
    
    sc.pl.umap(adata, color='sandpile_entropy', cmap='magma', title='Sandpile Entropy (Tipping Points)', show=True)
except Exception as e:
    print(f"Skipped tipping point due to error: {e}")
"""

text_grn = """\
## 6. Transcription Factor Enrichment

We can dynamically calculate transcription factor enrichments using synthetic or derived networks.
"""

code_grn = """\
tf_pairs_list = [
    ('SPI1', 'LYZ', 1.0),
    ('SPI1', 'CD14', 0.8),
    ('STAT1', 'ISG15', 1.0),
    ('STAT1', 'CXCL10', 0.9),
    ('PAX5', 'CD79A', 1.0),
    ('PAX5', 'MS4A1', 0.8)
]

valid_pairs = [(tf, tgt, w) for tf, tgt, w in tf_pairs_list if tf in adata.var_names and tgt in adata.var_names]
if not valid_pairs:
    valid_genes = adata.var_names[:5].tolist()
    valid_pairs = [(valid_genes[0], valid_genes[2], 1.0), (valid_genes[0], valid_genes[3], 0.8)]

tfs, tgts, wts = zip(*valid_pairs)
tf_df = pd.DataFrame({'tf': tfs, 'target': tgts, 'weight': wts})

if len(tf_df) >= 1:
    primary_tf = tf_df['tf'].iloc[0]
    adata = ct.grn.run_tf_enrichment(adata, tf_network=tf_df, source_col='tf', target_col='target', min_expr_fraction=0.0)
    
    if f"tf_score_{primary_tf}" in adata.obs:
        sc.pl.umap(adata, color=f"tf_score_{primary_tf}", cmap='viridis', title=f"TF Enrichment: {primary_tf}", show=True)
"""

text_cci = """\
## 7. Extracellular Communication (CellPhoneDB Scoring & Cell2Cell Plot)

We evaluate cell-cell communication by running non-parametric label permutations against biologically relevant Ligand-Receptor pairs.
"""

code_cci = """\
lr_pairs_list = [
    ('HLA-DRA', 'CD4'),
    ('B2M', 'CD3E'),
    ('CD86', 'CD28'),
    ('CCL5', 'CCR5'),
    ('IL32', 'CD4'),
    ('HLA-DPB1', 'CD4'),
    ('CD74', 'CD44')
]

valid_pairs = [(l, r) for l, r in lr_pairs_list if l in adata.var_names and r in adata.var_names]

if valid_pairs:
    ligands, receptors = zip(*valid_pairs)
    lr_df = pd.DataFrame({'ligand': ligands, 'receptor': receptors})
    
    ccc_res = ct.interaction.run_cellphonedb_scoring(adata, lr_pairs=lr_df, group_key='leiden_0.5', n_perms=100)
    if not ccc_res.empty:
        fig = ct.interaction.plot_cell2cell_dotplot(ccc_res, top_n=20)
        # Adding UMAP Arc plotting 
        try:
            ct.interaction.plot_cell2cell_umap(adata, ccc_res, group_key='leiden_0.5', top_n=10, save=os.path.join(fig_dir, "cci_umap_arcs.png"), show=True)
        except Exception as e:
            print(f"Arc plot failed: {e}")
"""

text_fin = """\
## 8. Differential Expression

We perform Dropout-Adjusted DE across major cell clusters.
"""

code_fin = """\
groups = adata.obs['leiden_0.5'].unique()
if len(groups) >= 2:
    try:
        de_res = ct.tools.dropout_adjusted_de(
            adata, group_key='leiden_0.5', group1=groups[0], group2=groups[1], 
            out_csv=os.path.join(fig_dir, "differential_expression_pbmc3k.csv")
        )
        print(f"Differentially expressed genes between {groups[0]} and {groups[1]} extracted successfully.")
    except Exception as e:
        print(f"DE skipped: {e}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_prep),
    nbf.v4.new_code_cell(code_prep),
    nbf.v4.new_markdown_cell(text_dim),
    nbf.v4.new_code_cell(code_dim),
    nbf.v4.new_markdown_cell(text_types),
    nbf.v4.new_code_cell(code_types),
    nbf.v4.new_markdown_cell(text_traj),
    nbf.v4.new_code_cell(code_traj),
    nbf.v4.new_markdown_cell(text_grn),
    nbf.v4.new_code_cell(code_grn),
    nbf.v4.new_markdown_cell(text_cci),
    nbf.v4.new_code_cell(code_cci),
    nbf.v4.new_markdown_cell(text_fin),
    nbf.v4.new_code_cell(code_fin),
]

with open('demo_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
