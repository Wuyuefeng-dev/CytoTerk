"""
Custom implementation for doublet identification, statistical analysis, and visualization.
"""

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import scipy.sparse as sp


def identify_doublets(adata: AnnData, expected_rate: float = 0.05, n_neighbors: int = 30) -> AnnData:
    """
    Identify potential doublets by simulating doublets from data and using a k-NN graph.
    This is an original, dependency-reduced implementation inspired by Scrublet.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix. Best run on raw, unnormalized counts.
    expected_rate : float
        The estimated doublet rate.
    n_neighbors : int
        Number of nearest neighbors to construct the graph.

    Returns
    -------
    AnnData
        Updated AnnData with 'doublet_score' and 'predicted_doublet' in obs.
    """
    print(f"Running custom original doublet detection (Expected Rate: {expected_rate})...")

    X = adata.X
    n_obs = X.shape[0]

    # 1. Simulate doublets
    n_sim = n_obs * 2
    idx1 = np.random.choice(n_obs, n_sim, replace=True)
    idx2 = np.random.choice(n_obs, n_sim, replace=True)

    if sp.issparse(X):
        X_sim = X[idx1] + X[idx2]
        X_comb = sp.vstack([X, X_sim])
    else:
        X_sim = X[idx1] + X[idx2]
        X_comb = np.vstack([X, X_sim])

    # 2. Basic dimensionality reduction (PCA on log-counts)
    if sp.issparse(X_comb):
        X_comb_log = X_comb.copy()
        X_comb_log.data = np.log1p(X_comb_log.data)
    else:
        X_comb_log = np.log1p(X_comb)

    try:
        pca = PCA(n_components=30, random_state=42)
        if sp.issparse(X_comb_log):
            X_comb_pca = pca.fit_transform(X_comb_log.toarray())
        else:
            X_comb_pca = pca.fit_transform(X_comb_log)
    except MemoryError:
        print("MemoryError during custom doublet detection. Try downsampling or using highly variable genes first.")
        return adata

    X_obs_pca = X_comb_pca[:n_obs]

    # 3. k-NN graph and scoring
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X_comb_pca)
    distances, indices = nn.kneighbors(X_obs_pca)

    # Score: proportion of simulated doublets in the neighborhood
    scores = np.mean(indices >= n_obs, axis=1)

    # Adjust scores relative to expected rate (simplified thresholding)
    threshold = np.quantile(scores, 1.0 - expected_rate)

    adata.obs['doublet_score'] = scores
    adata.obs['predicted_doublet'] = scores >= threshold

    n_doublets = adata.obs['predicted_doublet'].sum()
    print(f"Custom algorithm identified {n_doublets} doublets ({n_doublets/adata.n_obs*100:.1f}%).")

    return adata


def doublet_statistical_summary(adata: AnnData) -> dict:
    """
    Compute statistical summary of doublet scores and return as a dict.

    Parameters
    ----------
    adata : AnnData
        Must have 'doublet_score' and 'predicted_doublet' in obs.

    Returns
    -------
    dict with keys: n_cells, n_doublets, doublet_rate, score_mean, score_std,
                    score_median, score_q25, score_q75, score_max
    """
    if 'doublet_score' not in adata.obs:
        raise ValueError("Run identify_doublets() first to compute doublet_score.")

    scores = adata.obs['doublet_score'].values
    is_doublet = adata.obs['predicted_doublet'].values
    n_doublets = int(is_doublet.sum())

    summary = {
        'n_cells': adata.n_obs,
        'n_doublets': n_doublets,
        'n_singlets': adata.n_obs - n_doublets,
        'doublet_rate_%': round(n_doublets / adata.n_obs * 100, 2),
        'score_mean': round(float(np.mean(scores)), 4),
        'score_std': round(float(np.std(scores)), 4),
        'score_median': round(float(np.median(scores)), 4),
        'score_q25': round(float(np.percentile(scores, 25)), 4),
        'score_q75': round(float(np.percentile(scores, 75)), 4),
        'score_max': round(float(np.max(scores)), 4),
        'threshold': round(float(np.min(scores[is_doublet])) if n_doublets > 0 else float(np.max(scores)), 4),
    }
    return summary


def plot_doublet_scores(
    adata: AnnData,
    use_rep: str = 'X_umap',
    show: bool = True,
    save: str | None = None
):
    """
    Plot doublet scores and classification on the UMAP embedding.
    Generates a multi-panel figure in SeuratExtend style:
      - Panel 1: UMAP colored by continuous doublet_score
      - Panel 2: UMAP colored by binary predicted_doublet
      - Panel 3: Score distribution histogram (singlets vs doublets)
      - Panel 4: Box plot of doublet scores by cluster
      - Panel 5: Cumulative score distribution
      - Panel 6: UMAP with scores mapped to point size

    Parameters
    ----------
    adata : AnnData
        Must have 'doublet_score', 'predicted_doublet', and X_umap in obsm.
    use_rep : str
        Embedding to use for plotting.
    show : bool
        Whether to display the figure.
    save : str or None
        Path to save the figure.
    """
    import matplotlib
    if save:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import scanpy as sc

    try:
        from sccytotrek.plotting.style import (
            apply_seurat_theme, SEURAT_DISCRETE, SEURAT_EXPR_CMAP,
            FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
        )
    except ImportError:
        FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE = "white", 13, 9, 8
        SEURAT_DISCRETE = ["#4DBBD5","#E64B35","#00A087","#3C5488","#F39B7F",
                           "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]
        SEURAT_EXPR_CMAP = "YlOrRd"
        def apply_seurat_theme(ax, **kw): return ax

    if 'doublet_score' not in adata.obs:
        raise ValueError("Run identify_doublets() first.")

    if use_rep not in adata.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found. Run sc.tl.umap() first.")

    scores = adata.obs['doublet_score'].values
    is_doublet = adata.obs['predicted_doublet'].values
    umap_coords = adata.obsm[use_rep]

    fig = plt.figure(figsize=(18, 12), facecolor=FIG_BG)
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.3, hspace=0.35)

    COLOR_SINGLET = "#B09C85" # Warm beige
    COLOR_DOUBLET = "#DC0000" # Crimson

    # --- Panel 1: Continuous doublet score on UMAP ---
    ax1 = fig.add_subplot(gs[0, 0])
    apply_seurat_theme(ax1, spines="none")
    sc1 = ax1.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=scores, cmap=SEURAT_EXPR_CMAP, s=4, alpha=0.7, rasterized=True, edgecolors="none"
    )
    plt.colorbar(sc1, ax=ax1, label='Doublet Score', fraction=0.046, pad=0.04)
    ax1.set_title('Doublet Score (Continuous)', fontsize=TITLE_SIZE, fontweight='bold')
    ax1.set_xlabel('UMAP 1', fontsize=LABEL_SIZE); ax1.set_ylabel('UMAP 2', fontsize=LABEL_SIZE)

    # --- Panel 2: Binary predicted doublets on UMAP ---
    ax2 = fig.add_subplot(gs[0, 1])
    apply_seurat_theme(ax2, spines="none")
    mask_s = ~is_doublet
    
    # Plot singlets first, then doublets on top
    ax2.scatter(umap_coords[mask_s, 0], umap_coords[mask_s, 1],
                c=COLOR_SINGLET, s=4, alpha=0.6, rasterized=True, edgecolors="none", label="Singlet")
    ax2.scatter(umap_coords[is_doublet, 0], umap_coords[is_doublet, 1],
                c=COLOR_DOUBLET, s=8, alpha=0.9, rasterized=True, edgecolors="none", label="Doublet")
    
    ax2.legend(fontsize=TICK_SIZE, loc='upper right', frameon=False, markerscale=2)
    ax2.set_title('Predicted Doublets', fontsize=TITLE_SIZE, fontweight='bold')
    ax2.set_xlabel('UMAP 1', fontsize=LABEL_SIZE); ax2.set_ylabel('UMAP 2', fontsize=LABEL_SIZE)

    # --- Panel 3: Score histogram ---
    ax3 = fig.add_subplot(gs[0, 2])
    apply_seurat_theme(ax3, grid=True)
    ax3.hist(scores[mask_s], bins=50, alpha=0.75, color=COLOR_SINGLET, label='Singlet', density=True)
    ax3.hist(scores[is_doublet],  bins=50, alpha=0.75, color=COLOR_DOUBLET, label='Doublet', density=True)
    if is_doublet.sum() > 0:
        threshold = np.min(scores[is_doublet])
        ax3.axvline(threshold, color='#00A087', linestyle='--', lw=1.5, label=f'Threshold={threshold:.3f}')
    ax3.set_title('Score Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    ax3.set_xlabel('Doublet Score', fontsize=LABEL_SIZE); ax3.set_ylabel('Density', fontsize=LABEL_SIZE)
    ax3.legend(fontsize=TICK_SIZE, frameon=False)

    # --- Panel 4: Stats summary table ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('tight')
    ax4.axis('off')
    summary = doublet_statistical_summary(adata)
    table_data = [
        ['Metric', 'Value'],
        ['Total Cells', f"{summary['n_cells']:,}"],
        ['Doublets', f"{summary['n_doublets']:,}"],
        ['Singlets', f"{summary['n_singlets']:,}"],
        ['Doublet Rate', f"{summary['doublet_rate_%']}%"],
        ['Score Mean', f"{summary['score_mean']:.4f}"],
        ['Score Median', f"{summary['score_median']:.4f}"],
        ['Score Max', f"{summary['score_max']:.4f}"],
    ]
    
    # Simple table using text directly for better aesthetic matching Seurat
    y_pos = 0.9
    for row in table_data:
        fw = 'bold' if y_pos == 0.9 else 'normal'
        ax4.text(0.1, y_pos, row[0], fontsize=LABEL_SIZE + 1, fontweight=fw, transform=ax4.transAxes)
        ax4.text(0.6, y_pos, row[1], fontsize=LABEL_SIZE + 1, fontweight=fw, transform=ax4.transAxes)
        y_pos -= 0.1
    ax4.set_title('Statistical Summary', fontsize=TITLE_SIZE, fontweight='bold', pad=10)

    # --- Panel 5: Cumulative score distribution ---
    ax5 = fig.add_subplot(gs[1, 1])
    apply_seurat_theme(ax5, grid=True)
    sorted_scores = np.sort(scores)
    ax5.plot(sorted_scores, np.linspace(0, 1, len(sorted_scores)),
             color='#3C5488', lw=2, label='Empirical CDF')
    if is_doublet.sum() > 0:
        thr = np.min(scores[is_doublet])
        ax5.axvline(thr, color='#00A087', linestyle='--', lw=1.5, label=f'Threshold={thr:.3f}')
    ax5.set_title('Cumulative Score Distribution', fontsize=TITLE_SIZE, fontweight='bold')
    ax5.set_xlabel('Doublet Score', fontsize=LABEL_SIZE); ax5.set_ylabel('Fraction of Cells', fontsize=LABEL_SIZE)
    ax5.legend(fontsize=TICK_SIZE, frameon=False)

    # --- Panel 6: UMAP — scores as size ---
    ax6 = fig.add_subplot(gs[1, 2])
    apply_seurat_theme(ax6, spines="none")
    ax6.scatter(umap_coords[mask_s, 0], umap_coords[mask_s, 1],
                c=COLOR_SINGLET, s=3, alpha=0.4, label='Singlet', rasterized=True, edgecolors="none")
    ax6.scatter(umap_coords[is_doublet, 0], umap_coords[is_doublet, 1],
                c=scores[is_doublet], cmap=SEURAT_EXPR_CMAP, s=scores[is_doublet] * 80 + 10,
                alpha=0.9, label='Doublet', rasterized=True, edgecolors="none")
    ax6.set_title('Doublet Probability (Size=Score)', fontsize=TITLE_SIZE, fontweight='bold')
    ax6.set_xlabel('UMAP 1', fontsize=LABEL_SIZE); ax6.set_ylabel('UMAP 2', fontsize=LABEL_SIZE)
    ax6.legend(fontsize=TICK_SIZE, frameon=False, markerscale=1, loc='upper right')

    fig.suptitle('scCytoTrek — Doublet Detection Analysis',
                 fontsize=TITLE_SIZE + 2, fontweight='bold', y=1.02)
                 
    if save:
        plt.savefig(save, dpi=180, bbox_inches='tight', facecolor=FIG_BG)
        print(f"Saved doublet plot to {save}")
    if show:
        plt.show()
    plt.close()
