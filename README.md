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

### 6. Gene Regulatory Networks & GRN (`sccytotrek.grn`)
- **Custom TF Enrichment:** Predict transcription factor activity solely via dot-product network weights (no heavy dependencies).
- **In Silico Gene Knockdown:** Predict how genetic perturbations shift cell states on the PCA/UMAP manifold.

### 7. Pathway Analysis (`sccytotrek.pathway`)
- **GSVA / ssGSEA:** True single-cell pathway enrichment with `gseapy`, projected effortlessly onto UMAP embeddings.
- **GO Biological Process Over-Representation:** Direct API for gene set enrichment testing.

### 8. Multi-modal Integration (`sccytotrek.integration` & `sccytotrek.multiome`)
- **scVI Deep Learning Integration:** Perform batch integration and complex latent sub-clustering using state-of-the-art variational autoencoders.
- **Cross-Species Mapping:** Map Human and Mouse gene orthologs directly on the AnnData object.
- **Bulk projection:** Project sorted Bulk RNA-seq samples directly into the single-cell expression embedding.
- **RNA+ATAC Joint Integration:** Complete 6-algorithm suite for multi-omics structural mapping, including **WNN**, **CCA**, **Concatenated PCA**, **Procrustes Alignment**, **Similarity Network Fusion (SNF)**, and **Joint Harmony**.

---

## 💾 Installation

It is recommended to use an isolated Python environment. scCytoTrek officially requires Python >= 3.9.

### Option 1: Install from Source (pip)
```bash
git clone https://github.com/your-username/sccytotrek.git
cd sccytotrek
pip install -e .
```

### Option 2: Conda Environment Setup
```bash
conda create -n sccytotrek_env python=3.10
conda activate sccytotrek_env
pip install git+https://github.com/your-username/sccytotrek.git
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

## 🤝 Contributing & License

scCytoTrek is developed as open-source software. Pull requests for novel algorithms, bug fixes, or optimizations are welcome.
