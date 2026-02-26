# scCytoTrek 🚀

**scCytoTrek** is a comprehensive, scalable, and multi-functional Python package specifically designed for advanced single-cell and multi-omic data analysis. Built on top of `anndata` and `scanpy`, scCytoTrek minimizes external dependencies by providing custom, tailored algorithms for a wide variety of analytical workflows.

It is designed to give researchers deep insights into cellular heterogeneity, trajectory dynamics, cell-cell communication, and regulatory networks.

---

## 🌟 Key Features & Modules

scCytoTrek is organized into distinct functional modules:

### 1. Preprocessing (`sccytotrek.preprocessing`)
- **Fast Cell Subsampling:** Downsample massive datasets while preserving cluster diversity.
- **KNN Imputation:** Advanced k-nearest neighbors smoothing to recover lost signal from dropout events.
- **Robust QC & Normalization:** Standard single-cell QC wrappers integrated smoothly.

### 2. Tools & Analysis (`sccytotrek.tools`)
- **Custom Doublet Identification:** Detect artificial doublets using PCA-based neighborhood density (no generic dependencies required).
- **Dropout-Adjusted Differential Expression:** Perform DE analysis while correcting for varying cellular capture efficiencies (Cellular Detection Rate - CDR).
- **Confounder Regression:** Remove technical batch effects or uninteresting covariates directly from the expression matrix.
- **Cell Type Identification:** Score and assign biological cell types based on known marker gene lists.
- **Clinical Survival Correlation:** Integrate scRNA-seq expression with patient survival data using Lifelines.

### 3. Advanced Clustering (`sccytotrek.clustering`)
- **Non-Negative Matrix Factorization (NMF):** Discover additive, interpretable gene expression programs (meta-genes).
- **K-Means:** Rapid quantization for massive datasets or highly distinct, globular clusters.
- **Agglomerative (Hierarchical):** Uncover evolutionary or developmental hierarchies holding sub-lineages.
- **Spectral Clustering:** Identify non-convex and continuous manifold structures where K-Means fails.
- **Gaussian Mixture Models (GMM):** Model continuous differentiation bridges using soft-probabilistic cell assignments.
- **DBSCAN:** Density-based clustering, perfect for spatial data and isolating outlier/noise cells.

### 4. Trajectory Inference & Tipping Points (`sccytotrek.trajectory`)
- **Pseudotime Inference:** Automated wrappers for constructing DPT progression trajectories, alongside **Slingshot**, **Palantir**, and **CellRank** approximations.
- **Trajectory Visualization:** Streamgraphs visualizing dynamic cell-type proportion flows across pseudotime axes (`plot_streamgraph`), complementing Monocle3 principal graphs.
- **Ordering Effect Genes:** Identify which genes drive progression alongside the pseudotime axis.
- **Sandpile Model for Tipping Points:** Detect critical states ('tipping points') predicting sudden shifts in cellular fate using a robust network-entropy algorithm.
- **Lineage Extraction:** Extract differentiation graphs and parse complex barcoding data (e.g., Polylox, DARLIN arrays).

### 5. Cell-Cell Communication (`sccytotrek.interaction`)
- **Custom CellPhoneDB Algorithm:** Natively score Ligand-Receptor pair significance via randomized permutation tests.
- **Cell2Cell Plotting:** Exquisite visual dot-plots mapping sending/receiving populations to interaction strength and precision (-log10(p)).

### 6. Lineage Tracing & Integration (`sccytotrek.lineage`)
- **Barcode Dropout Imputation:** Use robust non-linear kNN expression topological imputation to fill in up to 50% missing cellular barcodes.
- **Lineage Visualization:** Specialized UMAPs and bar plots comparing cell clone densities before and after imputation.

### 7. Gene Regulatory Networks & GRN (`sccytotrek.grn`)
- **Custom TF Enrichment:** Predict transcription factor activity solely via dot-product network weights (no heavy dependencies).
- **In Silico Gene Knockdown:** Predict how genetic perturbations shift cell states on the PCA/UMAP manifold.

### 8. Pathway Analysis (`sccytotrek.pathway`)
- **GSVA / ssGSEA:** True single-cell pathway enrichment with `gseapy`, projected effortlessly onto UMAP embeddings.
- **GO Biological Process Over-Representation:** Direct API for gene set enrichment testing.

### 9. Multi-modal Integration (`sccytotrek.integration` & `sccytotrek.multiome`)
- **scVI Deep Learning Integration:** Perform batch integration and complex latent sub-clustering using state-of-the-art variational autoencoders.
- **Cross-Species Mapping:** Map Human and Mouse gene orthologs directly on the AnnData object.
- **Bulk projection:** Project sorted Bulk RNA-seq samples directly into the single-cell expression embedding.
- **RNA+ATAC Joint Integration:** Complete 6-algorithm suite for multi-omics structural mapping, including **WNN**, **CCA**, **Concatenated PCA**, **Procrustes Alignment**, **Similarity Network Fusion (SNF)**, and **Joint Harmony**.

---

## 💾 Installation

It is recommended to use an isolated Python environment. scCytoTrek officially requires Python >= 3.9.

### Option 1: Install from Source (pip)
```bash
git clone https://github.com/Wuyuefeng-dev/CytoTerk.git
cd CytoTerk
pip install -e .
```

### Option 2: Conda Environment Setup
```bash
conda create -n sccytotrek_env python=3.10
conda activate sccytotrek_env
pip install git+https://github.com/Wuyuefeng-dev/CytoTerk.git
```

**Required Core Dependencies:**
`scanpy`, `anndata`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `networkx`, `scipy`

**Extended Functional Dependencies:**
- Pathway analysis (`pathway`): `gseapy`
- Deep Learning (`integration`): `scvi-tools`
- Survival stats (`tools`): `lifelines`

## 📖 Quick Start & Documentation

To run a full breakdown analysis spanning Preprocessing, Normalization, Tipping Point prediction, GSVA, and Interaction Scoring, run the generated script:

```bash
python generate_demo_data.py
python demo_analysis.py
```

This generates `demo_figs/demo_report.md` along with dozens of automatically stylized analytical plots showing the capabilities of `scCytoTrek`.

---

## 🚀 scCytoTrek Comprehensive Demonstration & Walkthrough

Welcome to the definitive walkthrough for **scCytoTrek**. This document outlines our end-to-end multi-functional pipeline. Every major module available in `scCytoTrek` was utilized sequentially on generated paired single-cell RNA (scRNA) and Multiome datasets to validate their algorithmic robustness.

### Demo Data Generation
The raw inputs are artificially synthesized using `sccytotrek.datasets.make_mock_data` and `make_mock_multiome`. These yield rich `AnnData` and `MuData` objects possessing ground-truth temporal branches, multiple cell modalities, simulated batch variance, and complex lineage tracing barcodes.

---

## 1. Preprocessing (`sccytotrek.preprocessing` & `sccytotrek.tools`)

The initial steps heavily refine the raw count matrices.

### Downsampling & Normalization
- `ct.preprocessing.subsample_cells`: Randomly subsets data while attempting to preserve geometric shapes.
- `sc.pp.normalize_total` / `log1p` / `highly_variable_genes`: Classic scanpy-backed normalization scaling to exactly 1e4 counts/cell.
- `ct.preprocessing.impute_knn`: (Optional) specifically smooths dropout events by averaging normalized local densities prior to DE testing.

### Quality Control (QC) Distribution
Pre-processing requires robust visualization of total UMIs (`total_counts`) and detected genes (`n_genes_by_counts`) to trim low-quality cells or empty droplets effectively.
![QC Violin Plots](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/qc_violins.png)

### Custom Doublet Detection
`ct.tools.identify_doublets` simulates artificial cross-cluster droplets to build a density-aware kNN graph identifying potential dual-cell captures. Below is the un-filtered overlap space versus the cleaned manifold.

| Before Doublet Removing | Cleaned Dataset |
| :---: | :---: |
| ![Before Doublet Removing](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/doublets_before.png) | ![Cleaned Dataset](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/doublets_after.png) |

---

## 2. Advanced Clustering Algorithms (`sccytotrek.clustering`)

Standard pipelines default to **Leiden algorithms**. However, `scCytoTrek` empowers researchers by integrating an expansive suite of non-standard topological classifiers to dissect arbitrary expression shapes.

### Baseline Unsupervised Grouping

| Leiden (Base) | Louvain Algorithm |
| :---: | :---: |
| ![Leiden](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/leiden_clusters.png) | ![Louvain](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/louvain_clusters.png) |

### Specialized Clustering Methods
If discrete separation is fuzzy, standard methods fail. Here, alternative implementations highlight the geometry:

| K-Means (Fast) | GMM (Probabilistic) | Spectral (Topographic) |
| :---: | :---: | :---: |
| ![K-Means](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/kmeans_clusters.png) | ![GMM](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/gmm_clusters.png) | ![Spectral](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/spectral_clusters.png) |

| DBSCAN (Noise Filter) | Agglomerative (Hierarchical) |
| :---: | :---: |
| ![DBSCAN](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/dbscan_clusters.png) | ![Agglomerative](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/agglomerative_clusters.png) |

> [!NOTE]
> We also implement Non-Negative Matrix Factorization (NMF) via `ct.clustering.run_nmf` to identify co-varying continuous meta-gene programs rather than discrete cellular blobs.

---

## 3. Cell Identity and Scoring

### Automated Type Assignment
Rather than manual curation, `ct.tools.score_cell_types` cross-references a dictionary of known marker profiles against normalized cellular arrays.
![Inferred Cellular Typology](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/cell_types.png)

### Dropout-Adjusted DE Analysis
`ct.tools.run_differential_expression` uses a unique Cellular Detection Rate (CDR) linear model to isolate biological signals strictly away from capture-specific sequencer biases.

---

## 4. Trajectory Inference & Tipping Points (`sccytotrek.trajectory`)

Differentiation is rarely instantaneous. Trajectories establish macroscopic cell maturity indices (pseudotime) away from assigned progenitor roots.

### Method Comparison
`scCytoTrek` natively bridges assumptions from **Slingshot** (Principal Curves), **Palantir** (Markov Shortest Path), and **CellRank** (Velocity Flows) to map differentiation alongside baseline **Monocle3** networks.

| Model Comparisons | Monocle3 Principal Graph |
| :---: | :---: |
| ![Comparisons](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/trajectory_comparison.png) | ![Monocle3](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/trajectory_monocle3.png) |

### Visualizing Timeline Shifts (Streamgraphs)
To rapidly interpret density changes as time progresses: `ct.trajectory.plot_streamgraph` smoothens population variance along the computed chronological axis.
![Dynamic Population Streamgraph](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/trajectory_streamgraph.png)

### Tipping Point Calculation (Sandpile Model)
Using `ct.trajectory.compute_sandpile_entropy`, the timeline is discretely binned, dynamically tracking global network variability. A spike in non-linear "regulatory entropy" flags critical transitional tipping points preceding differentiation splits that cannot be visually recovered from a UMAP trace.

| Entropy Trajectory | Top Genes Driving Instability |
| :---: | :---: |
| ![Entropy](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/sandpile_entropy_trajectory.png) | ![Instability](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/tipping_genes_barplot.png) |

---

## 5. Extracellular Communication (`sccytotrek.interaction`)

### Ligand-Receptor Scoring
Rather than relying on closed-source databases, `ct.interaction.run_cellphonedb_scoring` uses customized non-parametric label permutations identifying cross-cluster protein-protein signals.

### Interaction Mapping
`ct.interaction.plot_cell2cell_dotplot` condenses millions of connections into targeted, mathematically sound receptor networks. Size indicates reliability; color corresponds to the product strength.
![Receptor Crosslinking Grid](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/cell2cell_interaction.png)

---

## 6. Multi-omics Architecture (`sccytotrek.multiome`)

To natively resolve modern `10x Multiome` assays, standard transcriptomic matrices must be rigidly pinned against `scATAC-seq` epigenetic peaks.

Implemented Integration Arrays acting heavily upon structured `mudata` inputs:
1. **Weighted Nearest Neighbors (WNN)**: Reconciles independent distance boundaries.
2. **Canonical Correlation (CCA)**: Selects sub-vectors correlating both input assays.
3. **Procrustes Alignment**: Performs mathematical affine geometry rotation to mirror RNA over ATAC.
4. **SNF** & **Concatenated PCA** & **Joint Harmony**.

![6-Way Multiome Model Outputs](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/multiome_umaps.png)

---

## 7. Lineage Tracing & Imputation (`sccytotrek.lineage`)

Allows users to simulate, benchmark, and impute complex lineage tracing assays where a significant percentage of cellular barcodes fail to capture correctly but the underlying cellular transcriptomes are preserved. **Impute missing cells based on topological RNA similarities.**

---

## 8. Enrichment and Mapping Features

- **scVI Deep Learning Integration** (`run_scvi_integration`): Variational Encoders extracting robust underlying latent traits immune to sequencing lane disruptions.
- **Cross-Species Alignment** (`map_human_mouse_orthologs`): Bridging mouse experimental datasets against clinical human cohorts via hardcoded Ensembl relationships. 
- **Bulk Alignment Visualization**: Leveraging UMAP's `.transform()` behavior to drop Bulk-RNA tissues neatly into the underlying single-cell visual reference map.
- **TF Enrichment**: `run_tf_enrichment` establishes pseudo bulk dependencies without importing resource-intensive secondary libraries. It scores network edge weights directly over single-cell RNA transcript inputs.
![TF Enrichment Module](https://raw.githubusercontent.com/Wuyuefeng-dev/CytoTerk/main/demo_figs/tf_enrichment_umap.png)

---

## 🤝 Contributing & License

scCytoTrek is developed as open-source software. Pull requests for novel algorithms, bug fixes, or optimizations are welcome.

