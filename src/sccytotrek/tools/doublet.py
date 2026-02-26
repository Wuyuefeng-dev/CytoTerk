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
    save: str = None
):
    """
    Plot doublet scores and classification on the UMAP embedding.
    Generates a multi-panel figure:
      - Panel 1: UMAP colored by continuous doublet_score
      - Panel 2: UMAP colored by binary predicted_doublet
      - Panel 3: Score distribution histogram (singlets vs doublets)
      - Panel 4: Box plot of doublet scores by cluster (if leiden/louvain exist)

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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import scanpy as sc

    if 'doublet_score' not in adata.obs:
        raise ValueError("Run identify_doublets() first.")

    if use_rep not in adata.obsm:
        raise ValueError(f"Embedding '{use_rep}' not found. Run sc.tl.umap() first.")

    scores = adata.obs['doublet_score'].values
    is_doublet = adata.obs['predicted_doublet'].values
    umap_coords = adata.obsm[use_rep]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0e0e0e')

    # --- Panel 1: Continuous doublet score on UMAP ---
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor('#0e0e0e')
    sc1 = ax1.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=scores, cmap='plasma', s=4, alpha=0.7, rasterized=True
    )
    fig.colorbar(sc1, ax=ax1, label='Doublet Score', fraction=0.046, pad=0.04)
    ax1.set_title('Doublet Score (Continuous)', color='white', fontsize=12, fontweight='bold')
    ax1.set_xlabel('UMAP 1', color='white'); ax1.set_ylabel('UMAP 2', color='white')
    ax1.tick_params(colors='white'); [s.set_color('white') for s in ax1.spines.values()]

    # --- Panel 2: Binary predicted doublets on UMAP ---
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor('#0e0e0e')
    colors_binary = ['#444466' if not d else '#ff4444' for d in is_doublet]
    ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c=colors_binary, s=4, alpha=0.7, rasterized=True)
    from matplotlib.patches import Patch
    legend_els = [Patch(fc='#444466', label='Singlet'), Patch(fc='#ff4444', label='Doublet')]
    ax2.legend(handles=legend_els, loc='upper right', framealpha=0.3, labelcolor='white')
    ax2.set_title('Predicted Doublets', color='white', fontsize=12, fontweight='bold')
    ax2.set_xlabel('UMAP 1', color='white'); ax2.set_ylabel('UMAP 2', color='white')
    ax2.tick_params(colors='white'); [s.set_color('white') for s in ax2.spines.values()]

    # --- Panel 3: Score histogram ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor('#1a1a2e')
    ax3.hist(scores[~is_doublet], bins=50, alpha=0.75, color='#4fc3f7', label='Singlet', density=True)
    ax3.hist(scores[is_doublet], bins=50, alpha=0.75, color='#ff4444', label='Doublet', density=True)
    if is_doublet.sum() > 0:
        threshold = np.min(scores[is_doublet])
        ax3.axvline(threshold, color='gold', linestyle='--', lw=1.5, label=f'Threshold={threshold:.3f}')
    ax3.set_title('Score Distribution', color='white', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Doublet Score', color='white'); ax3.set_ylabel('Density', color='white')
    ax3.legend(framealpha=0.3, labelcolor='white')
    ax3.tick_params(colors='white'); [s.set_color('white') for s in ax3.spines.values()]
    ax3.set_facecolor('#1a1a2e')

    # --- Panel 4: Stats summary table ---
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.axis('off')
    ax4.set_facecolor('#0e0e0e')
    summary = doublet_statistical_summary(adata)
    table_data = [
        ['Metric', 'Value'],
        ['Total Cells', f"{summary['n_cells']:,}"],
        ['Doublets', f"{summary['n_doublets']:,}"],
        ['Singlets', f"{summary['n_singlets']:,}"],
        ['Doublet Rate', f"{summary['doublet_rate_%']}%"],
        ['Score Mean', f"{summary['score_mean']:.4f}"],
        ['Score Median', f"{summary['score_median']:.4f}"],
        ['Score Std', f"{summary['score_std']:.4f}"],
        ['Score Q25', f"{summary['score_q25']:.4f}"],
        ['Score Q75', f"{summary['score_q75']:.4f}"],
    ]
    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#555555')
        if row == 0:
            cell.set_facecolor('#2a2a4a')
            cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif row % 2 == 0:
            cell.set_facecolor('#1a1a2e')
            cell.get_text().set_color('#cccccc')
        else:
            cell.set_facecolor('#111130')
            cell.get_text().set_color('#cccccc')
    ax4.set_title('Statistical Summary', color='white', fontsize=12, fontweight='bold', pad=10)

    # --- Panel 5: Cumulative score distribution ---
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('#1a1a2e')
    sorted_scores = np.sort(scores)
    ax5.plot(sorted_scores, np.linspace(0, 1, len(sorted_scores)),
             color='#29b6f6', lw=2, label='Empirical CDF')
    if is_doublet.sum() > 0:
        thr = np.min(scores[is_doublet])
        ax5.axvline(thr, color='gold', linestyle='--', lw=1.5, label=f'Threshold={thr:.3f}')
    ax5.set_title('Cumulative Score Distribution', color='white', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Doublet Score', color='white'); ax5.set_ylabel('Fraction of Cells', color='white')
    ax5.legend(framealpha=0.3, labelcolor='white')
    ax5.tick_params(colors='white'); [s.set_color('white') for s in ax5.spines.values()]

    # --- Panel 6: UMAP — scores as size ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor('#0e0e0e')
    # Plot singlets first, then doublets on top
    mask_s = ~is_doublet
    ax6.scatter(umap_coords[mask_s, 0], umap_coords[mask_s, 1],
                c='#334466', s=3, alpha=0.4, label='Singlet', rasterized=True)
    ax6.scatter(umap_coords[is_doublet, 0], umap_coords[is_doublet, 1],
                c=scores[is_doublet], cmap='hot', s=scores[is_doublet] * 80 + 10,
                alpha=0.9, label='Doublet', rasterized=True)
    ax6.set_title('Doublet Probability (Size=Score)', color='white', fontsize=11, fontweight='bold')
    ax6.set_xlabel('UMAP 1', color='white'); ax6.set_ylabel('UMAP 2', color='white')
    ax6.tick_params(colors='white'); [s.set_color('white') for s in ax6.spines.values()]
    ax6.legend(framealpha=0.3, labelcolor='white', markerscale=2)

    fig.suptitle('scCytoTrek — Doublet Detection Analysis', color='white',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=180, bbox_inches='tight', facecolor='#0e0e0e')
        print(f"Saved doublet plot to {save}")
    if show:
        plt.show()
    plt.close()
