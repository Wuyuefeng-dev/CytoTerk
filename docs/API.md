# scCytoTrek API Reference

Welcome to the `scCytoTrek` API documentation. This document provides a comprehensive overview of the core modules and functions available in the package.

## 1. `sccytotrek.preprocessing`

Modules for initial data manipulation, QC, and imputation.

### `subsample_cells(adata: AnnData, fraction: float = 0.5, random_state: int = 42) -> AnnData`
Randomly downsamples the AnnData object to a specified fraction of cells. Useful for accelerating downstream algorithms on massive datasets while maintaining representation.
- **Parameters:**
  - `adata`: Input AnnData object.
  - `fraction`: Proportion of cells to keep (0.0 to 1.0).
  - `random_state`: Seed for reproducibility.

### `impute_knn(adata: AnnData, n_neighbors: int = 15, use_rep: str = "X_pca") -> AnnData`
Imputes missing gene expression (dropout events) by taking the mean expression of each cell's $k$-nearest neighbors in a reduced dimensional space. Creates a new layer `imputed` to preserve original raw or normalized counts.
- **Parameters:**
  - `adata`: Input AnnData object.
  - `n_neighbors`: Number of neighbors to average over.
  - `use_rep`: The representation (e.g., PCA, scVI latent space) to compute neighbor distances.

## 2. `sccytotrek.tools`

Core analysis and biological interrogation tools.

### `identify_doublets_custom(adata: AnnData, n_neighbors: int = 20, use_rep: str = "X_pca", threshold: float = 0.95) -> AnnData`
Custom algorithm to identify artificial doublets without external dependencies. Calculates local density (distance to $k$-th neighbor) in PCA space. Cells in abnormally high-density regions are flagged.
- **Parameters:**
  - `adata`: Input AnnData object.
  - `threshold`: Density quantile above which cells are flagged as doublets.

### `run_differential_expression(adata: AnnData, group_col: str, group1: str, group2: str = None, adjust_dropout: bool = True) -> pd.DataFrame`
Performs differential expression using Welch's t-test or Wilcoxon Rank-Sum. If `adjust_dropout` is true, the test incorporates the Cellular Detection Rate (CDR) to correct for capture efficiency variation between droplets.

### `score_cell_types(adata: AnnData, marker_dict: dict, use_raw: bool = False)`
Calculates a signature score (mean normalized expression) for provided cell type marker sets and assigns the highest-scoring identity to each cell.

### `run_survival_analysis(adata: AnnData, time_col: str, event_col: str, feature: str = "leiden_0.5") -> pd.DataFrame`
Uses `lifelines` to compute Cox Proportional Hazards linking single-cell features (e.g., cluster abundance or gene expression per patient) to clinical survival curves.

## 3. `sccytotrek.clustering`

Advanced unsupervised grouping methodologies. 

### `run_nmf(adata: AnnData, n_components: int = 10, random_state: int = 42) -> AnnData`
Performs Non-Negative Matrix Factorization (NMF) to find additive gene expression modules ('meta-genes'). Useful for continuous phenotype discovery. Adds basis vectors to `varm` and coefficients to `obsm`.

### `run_kmeans(adata: AnnData, n_clusters: int, use_rep: str = 'X_pca') -> AnnData`
Standard hard clustering. Extremely fast, assigns cells to distinct blobs based on Euclidean distance to centroids.

### `run_agglomerative(adata: AnnData, n_clusters: int, use_rep: str = 'X_pca') -> AnnData`
Hierarchical clustering building a tree of cell relationships. Captures nested sub-lineages but scales poorly to millions of cells.

### `run_spectral(adata: AnnData, n_clusters: int, use_rep: str = 'X_pca') -> AnnData`
Maps cells via Laplacian eigenmaps before applying K-Means. Exceptional at detecting complex, non-convex manifolds (like intertwined developmental branches).

### `run_gmm(adata: AnnData, n_components: int, use_rep: str = 'X_pca') -> AnnData`
Probabilistic clustering. Assigns soft membership probabilities, making it optimal for cells sitting 'between' distinct states.

### `run_dbscan(adata: AnnData, eps: float, min_samples: int, use_rep: str = 'X_pca') -> AnnData`
Density-based clustering. excellent for identifying outlier noise and finding arbitrarily shaped spatial blobs.

## 4. `sccytotrek.trajectory`

Differentiation, pseudotime, and non-linear dynamics.

### `compute_sandpile_entropy(adata: AnnData, pseudotime_key: str, n_bins: int = 50) -> dict`
Detects critical transition states (**Tipping Points**) in differentiation trajectories. Bins cells by pseudotime and calculates the Shannon entropy of inferred regulatory networks. Spikes in entropy designate tipping points preceding radical fate shifts.

### `extract_lineage_graph(adata: AnnData, barcode_col: str) -> nx.DiGraph`
Constructs a directed graph representing cellular evolution based on inherited static barcodes (e.g., Polylox, DARLIN). 

### `find_ordering_genes(adata: AnnData, pseudotime_col: str, n_genes: int = 100) -> pd.DataFrame`
Identifies genes whose expression is strongly rank-correlated with the inferred pseudotime, pinpointing the genetic drivers of trajectory progression.

### `run_slingshot_pseudotime(adata: AnnData, groupby: str, root_cluster: str = None) -> AnnData`
Approximates the Slingshot trajectory inference algorithm. Computes cluster centroids in UMAP space and fits a 1D Principal Curve through them, projecting individual cells onto the curve to estimate global pseudotime progression.

### `run_palantir_pseudotime(adata: AnnData, root_cell: str) -> AnnData`
Simulates a Palantir-like Markov Chain pseudotime framework. Calculates the shortest path distances across the sparse k-Nearest Neighbors (kNN) connectivity graph originating from the specified root cell using Dijkstra's algorithm.

### `run_cellrank_pseudotime(adata: AnnData, root_cell: str) -> AnnData`
Approximation of CellRank's directed macroscopic flow. Applies a weighted, non-linear expansion to UMAP spatial distances from a designated root cell to mimic late-stage commitment probabilities across continuous landscapes.

### `plot_streamgraph(adata: AnnData, time_key: str, group_key: str, n_bins: int = 50) -> plt.Figure`
Generates a continuous Streamgraph visualization, heavily smoothing categorical proportions (e.g., cell types or clusters) dynamically plotted across a specified pseudotime axis using 1D Gaussian filters.

## 5. `sccytotrek.interaction`

Ligand-Receptor inference and Cell-Cell Communication.

### `run_cellphonedb_scoring(adata: AnnData, lr_pairs: pd.DataFrame, group_key: str, n_perms: int = 1000) -> pd.DataFrame`
A fully custom, dependency-free implementation of the CellPhoneDB algorithm. Calculates the mean expression of ligand and receptor across two clusters, then performs `n_perms` label-shuffling permutations to derive an empirical p-value for the interaction strength.

### `plot_cell2cell_dotplot(df_res: pd.DataFrame, top_n: int = 20, save_path: str = None)`
Visualizes the output of `run_cellphonedb_scoring` mimicking exquisite Cell2Cell visualizations. X-axis represents interacting cluster pairs, Y-axis represents the Ligand-Receptor pair. Color indicates interaction score, size indicates $-\log_{10}(p-value)$.

## 6. `sccytotrek.grn`

Gene Regulatory Networks and Transcription Factor mapping.

### `run_tf_enrichment(adata: AnnData, tf_network: pd.DataFrame, min_expr_fraction: float = 0.05) -> AnnData`
Infers Transcription Factor activity by performing a weighted dot-product between normalized cellular RNA expression and a TF-Target regulon matrix. **Crucially**, it scales the inferred target-based activity by the normalized RNA expression of the TF gene itself, ensuring that only transcriptionally active TFs are scored highly.

## 7. `sccytotrek.pathway`

Gene Set Enrichment Analysis (GSEA).

### `plot_gsva_umap(adata: AnnData, pathway_name: str, cmap: str = 'Reds') -> Figure`
Extracts `gseapy` computed single-cell level pathway enrichment scores (ssGSEA/GSVA) from `adata.obsm` and effortlessly projects them directly onto the scRNA UMAP for spatial visualization of pathway activation.

## 8. `sccytotrek.integration` & `sccytotrek.multiome`

Multi-modal and Cross-dataset harmonization.

### `run_scvi_integration(adata: AnnData, batch_key: str) -> AnnData`
Wrapper around `scvi-tools` Variational Autoencoder. Performs deep learning-based harmonization across technical batches, creating a corrected latent space (`X_scvi`) for downstream trajectory or clustering tasks.

### `map_human_mouse_orthologs(adata: AnnData, conversion_dict: dict) -> AnnData`
Seamlessly translates feature names between Human and Mouse ensembl/symbol nomenclatures using a provided bipartite conversion dictionary, enabling cross-species atlas integration.

### `run_wnn(mdata: MuData, rna_key: str = "rna", atac_key: str = "atac") -> None`
Derives independent neighbor graphs for dual modalities and initiates a Weighted Nearest Neighbors (WNN) joint multi-modal graph framework.

### `run_cca_integration(mdata: MuData)`
Calculates Canonical Correlation Analysis (CCA) between scRNA-seq PCA and scATAC-seq LSI matrices, deriving an abstract, maximally correlated joint biological subspace.

### `run_concat_pca_integration(mdata: MuData)`
Performs an early-fusion dimensionality reduction. L2-normalizes independent embeddings, concatenates them row-wise, and outputs a joint Principal Component axis (`X_concat_pca`).

### `run_procrustes_integration(mdata: MuData)`
Applies Procrustes Shape Alignment. Through optimal linear transformations (scaling, rotation, translation), the spatial structure of ATAC embeddings is geometrically forcibly aligned and averaged against the RNA embeddings.

### `run_snf_integration(mdata: MuData)`
Calculates Similarity Network Fusion (SNF). Merges independent adjacency graphs (kNN) from each modality iteratively, arriving at a robust, fused non-linear consensus neighbor network.

### `run_joint_harmony(mdata: MuData, batch_key: str = None)`
Utilizes `harmonypy` to actively harmonize the multimodal dataset natively in the joint `ConcatPCA` dimensionality space, effectively suppressing modality-specific biases or external confounding batch artifacts.
