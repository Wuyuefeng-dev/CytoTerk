"""
Bulk RNA-seq → single-cell UMAP projection
==========================================

True PCA-projection method (not a mock):
1. Scale bulk data using SC gene means/stds
2. Project onto SC PCA loadings to obtain bulk PCA coords
3. Use kNN regressor in PCA space to interpolate UMAP coords for each bulk sample

Visualization: SeuratExtend-style multi-panel figure
  Panel A — SC UMAP coloured by cluster with bulk samples overlaid as large stars
  Panel B — Per-bulk-sample pie charts showing nearest-neighbour cluster composition
  Panel C — Heatmap of bulk vs SC reference cluster correlation
  Panel D — Dot plot of top differentially expressed marker genes across bulk/SC groups
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Core projection logic
# ─────────────────────────────────────────────────────────────────────────────

def project_bulk_to_umap(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
) -> AnnData:
    """
    Project bulk RNA-seq samples onto an existing single-cell UMAP.

    Uses the actual PCA loadings from `adata_sc.varm['PCs']` to transform
    bulk expression into the SC PCA space, then uses a kNN regressor to
    predict UMAP coordinates.

    Parameters
    ----------
    adata_sc : AnnData
        Preprocessed single-cell data; must have X_pca and X_umap in obsm,
        plus `varm['PCs']` (set by sc.pp.pca).
    adata_bulk : AnnData
        Raw or normalized bulk RNA-seq (samples × genes). Shared gene space
        with adata_sc is automatically identified and used.
    n_neighbors : int
        kNN neighbours for UMAP coordinate regression.
    use_rep : str
        obsm key for the SC reference embedding (default "X_pca").

    Returns
    -------
    adata_bulk with X_pca and X_umap added to obsm, and
    'nn_clusters', 'nn_cluster_weights' added to obs/obsm.
    """
    if 'X_umap' not in adata_sc.obsm:
        raise ValueError("adata_sc must have X_umap; run sc.tl.umap() first.")
    if use_rep not in adata_sc.obsm:
        raise ValueError(f"adata_sc must have {use_rep} in obsm.")

    # Common genes
    common = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    if len(common) < 20:
        raise ValueError(f"Only {len(common)} common genes — ensure matching gene space.")
    print(f"  Projecting bulk using {len(common)} shared genes.")

    sc_sub  = adata_sc[:, common]
    bulk_sub = adata_bulk[:, common]

    # Bulk expression matrix (dense)
    X_bulk = bulk_sub.X.toarray() if sp.issparse(bulk_sub.X) else bulk_sub.X.copy()

    # Scale bulk to SC statistics
    X_sc = sc_sub.X.toarray() if sp.issparse(sc_sub.X) else sc_sub.X
    sc_means = np.asarray(X_sc.mean(axis=0)).ravel()
    sc_stds  = np.asarray(X_sc.std(axis=0)).ravel()
    sc_stds[sc_stds == 0] = 1.0
    X_bulk_scaled = (X_bulk - sc_means) / sc_stds

    # Project onto PCA loadings (n_genes × n_components)
    if 'PCs' in sc_sub.varm:
        loadings = sc_sub.varm['PCs']                              # (n_common, n_pcs)
    else:
        # Fall back: estimate loadings from the SC PCA embedding via pseudoinverse
        X_sc_full = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else adata_sc.X
        sc_pca_full = adata_sc.obsm[use_rep]
        loadings, *_ = np.linalg.lstsq(
            (X_sc_full - X_sc_full.mean(0)) / (X_sc_full.std(0) + 1e-9),
            sc_pca_full, rcond=None
        )

    # Align loadings to common gene set
    sc_gene_list = list(sc_sub.var_names)
    common_idx   = [sc_gene_list.index(g) for g in common]
    loadings_common = loadings[common_idx, :]   # (n_common, n_pcs)

    n_pcs = min(loadings_common.shape[1], adata_sc.obsm[use_rep].shape[1])
    bulk_pca = X_bulk_scaled @ loadings_common[:, :n_pcs]        # (n_bulk, n_pcs)
    adata_bulk.obsm['X_pca'] = bulk_pca.astype(np.float32)

    # kNN regression in PCA space → UMAP coordinates
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
    knn.fit(adata_sc.obsm[use_rep][:, :n_pcs], adata_sc.obsm['X_umap'])
    bulk_umap = knn.predict(bulk_pca)
    adata_bulk.obsm['X_umap'] = bulk_umap.astype(np.float32)

    # Cluster composition of kNN neighbourhood
    if 'cluster' in adata_sc.obs:
        nn_model = NearestNeighbors(n_neighbors=n_neighbors).fit(
            adata_sc.obsm[use_rep][:, :n_pcs]
        )
        _, indices = nn_model.kneighbors(bulk_pca)
        sc_clusters = adata_sc.obs['cluster'].values.astype(str)
        unique_cl   = sorted(np.unique(sc_clusters))
        comp_matrix  = np.zeros((adata_bulk.n_obs, len(unique_cl)))
        for i, nn_idx in enumerate(indices):
            nn_cl = sc_clusters[nn_idx]
            for j, cl in enumerate(unique_cl):
                comp_matrix[i, j] = (nn_cl == cl).sum() / len(nn_idx)
        adata_bulk.uns['nn_cluster_composition'] = pd.DataFrame(
            comp_matrix, index=adata_bulk.obs_names, columns=unique_cl
        )

    return adata_bulk


# ─────────────────────────────────────────────────────────────────────────────
# SeuratExtend-style visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_bulk_alignment(
    adata_sc: AnnData,
    adata_bulk: AnnData,
    cluster_key: str = "cluster",
    bulk_label_key: str = None,
    save: str = None,
    show: bool = True,
):
    """
    SeuratExtend-style bulk ↔ single-cell alignment figure.

    Four-panel layout (white background):
    A — SC UMAP coloured by cluster + bulk stars
    B — Per-bulk-sample neighbourhood cluster pie charts
    C — Bulk vs SC cluster correlation heatmap
    D — Marker gene dot plot (expression × fraction)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Wedge
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    umap_sc   = adata_sc.obsm['X_umap']
    umap_bulk = adata_bulk.obsm['X_umap']
    n_bulk    = adata_bulk.n_obs

    cluster_labels = adata_sc.obs[cluster_key].values.astype(str) \
        if cluster_key in adata_sc.obs else np.zeros(adata_sc.n_obs, dtype=str)
    unique_cl = sorted(np.unique(cluster_labels))
    n_cl      = len(unique_cl)

    cmap_cl   = plt.cm.get_cmap('tab10', n_cl)
    cl_colors = {c: cmap_cl(i) for i, c in enumerate(unique_cl)}

    bulk_names = adata_bulk.obs_names.tolist()
    if bulk_label_key and bulk_label_key in adata_bulk.obs:
        bulk_names = adata_bulk.obs[bulk_label_key].astype(str).tolist()

    fig = plt.figure(figsize=(22, 18), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── Panel A: SC UMAP + Bulk overlay ────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('white')
    for cl in unique_cl:
        mask = cluster_labels == cl
        ax_a.scatter(umap_sc[mask, 0], umap_sc[mask, 1],
                     c=[cl_colors[cl]], s=4, alpha=0.5, rasterized=True,
                     label=f'Cluster {cl}')
    # Bulk stars
    bulk_cmap = plt.cm.get_cmap('Set1', n_bulk)
    for i, (bx, by) in enumerate(umap_bulk):
        ax_a.scatter(bx, by, s=260, marker='*', color=bulk_cmap(i),
                     edgecolors='#333333', linewidth=0.8, zorder=10,
                     label=bulk_names[i])
    ax_a.set_title('Bulk Samples on SC UMAP', fontsize=12, fontweight='bold')
    ax_a.set_xlabel('UMAP 1', fontsize=9); ax_a.set_ylabel('UMAP 2', fontsize=9)
    # Legend: clusters below stars
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles, labels, fontsize=6, ncol=2, frameon=False,
                loc='lower right', borderpad=0.5)

    # ── Panel B: Per-bulk pie charts of cluster composition ──────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('white')
    ax_b.axis('off')
    ax_b.set_title('Bulk Neighbourhood Cluster Composition', fontsize=12, fontweight='bold',
                   y=1.0)

    if 'nn_cluster_composition' in adata_bulk.uns:
        comp_df = adata_bulk.uns['nn_cluster_composition']
        pie_cols = 4
        pie_rows = int(np.ceil(n_bulk / pie_cols))
        pie_gs   = gridspec.GridSpecFromSubplotSpec(pie_rows, pie_cols,
                                                    subplot_spec=gs[0, 1],
                                                    hspace=0.6, wspace=0.4)
        for bi in range(n_bulk):
            row_i, col_i = divmod(bi, pie_cols)
            pie_ax = fig.add_subplot(pie_gs[row_i, col_i])
            pie_ax.set_facecolor('white')
            vals   = comp_df.iloc[bi].values
            colors = [cl_colors[c] for c in comp_df.columns]
            wedges, _ = pie_ax.pie(vals, colors=colors, startangle=90,
                                   wedgeprops=dict(linewidth=0.4, edgecolor='white'))
            pie_ax.set_title(bulk_names[bi], fontsize=7, pad=2)
        # Shared legend
        legend_patches = [plt.matplotlib.patches.Patch(color=cl_colors[c], label=f'C{c}')
                          for c in unique_cl]
        ax_b.legend(handles=legend_patches, loc='lower center', ncol=n_cl,
                    fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.05))
    else:
        ax_b.text(0.5, 0.5, 'Run project_bulk_to_umap() first',
                  ha='center', va='center', transform=ax_b.transAxes)

    # ── Panel C: Bulk × SC cluster correlation heatmap ───────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_facecolor('white')

    # Compute pseudo-bulk profiles per SC cluster and correlate with bulk samples
    common_genes = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    if len(common_genes) > 10:
        sc_sub   = adata_sc[:, common_genes]
        bulk_sub = adata_bulk[:, common_genes]
        X_sc_raw = sc_sub.X.toarray() if sp.issparse(sc_sub.X) else sc_sub.X
        X_bulk_r = bulk_sub.X.toarray() if sp.issparse(bulk_sub.X) else bulk_sub.X

        pseudobulk = np.zeros((n_cl, len(common_genes)))
        for j, cl in enumerate(unique_cl):
            mask  = cluster_labels == cl
            pseudobulk[j] = X_sc_raw[mask].mean(axis=0)

        # Pearson correlation: bulk samples × SC clusters
        corr_mat = np.zeros((n_bulk, n_cl))
        for i in range(n_bulk):
            for j in range(n_cl):
                if pseudobulk[j].std() > 0 and X_bulk_r[i].std() > 0:
                    corr_mat[i, j] = np.corrcoef(X_bulk_r[i], pseudobulk[j])[0, 1]

        im = ax_c.imshow(corr_mat, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax_c, label='Pearson r', fraction=0.046, pad=0.04)
        ax_c.set_xticks(range(n_cl)); ax_c.set_xticklabels([f'C{c}' for c in unique_cl],
                                                              fontsize=8, rotation=45)
        ax_c.set_yticks(range(n_bulk)); ax_c.set_yticklabels(bulk_names, fontsize=8)
        ax_c.set_title('Bulk ↔ SC Cluster Correlation', fontsize=12, fontweight='bold')
        ax_c.set_xlabel('SC Cluster', fontsize=9); ax_c.set_ylabel('Bulk Sample', fontsize=9)
    else:
        ax_c.text(0.5, 0.5, 'Insufficient common genes', ha='center', va='center',
                  transform=ax_c.transAxes)
        ax_c.axis('off')

    # ── Panel D: Dot plot — top marker genes × bulk/SC groups ────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('white')

    if len(common_genes) > 10:
        # Pick top N variable genes across SC clusters as reference markers
        gene_var = X_sc_raw.var(axis=0)
        top_idx  = np.argsort(gene_var)[::-1][:20]
        top_genes = [common_genes[i] for i in top_idx]

        # Build dot data: mean expression + fraction expressing per SC cluster
        dot_means  = np.zeros((n_cl, len(top_genes)))
        dot_frac   = np.zeros((n_cl, len(top_genes)))
        for j, cl in enumerate(unique_cl):
            mask = cluster_labels == cl
            sub  = X_sc_raw[np.ix_(mask, top_idx)]
            dot_means[j] = sub.mean(axis=0)
            dot_frac[j]  = (sub > 0).mean(axis=0)

        # Normalise expression per gene for colour
        mu_min, mu_max = dot_means.min(0), dot_means.max(0)
        mu_range = mu_max - mu_min; mu_range[mu_range == 0] = 1
        dot_norm = (dot_means - mu_min) / mu_range

        ax_d.set_facecolor('white')
        norm = Normalize(vmin=0, vmax=1)
        dot_cmap = cm.YlOrRd

        for gi, gene in enumerate(top_genes):
            for ci, cl in enumerate(unique_cl):
                size  = dot_frac[ci, gi] * 300
                color = dot_cmap(norm(dot_norm[ci, gi]))
                ax_d.scatter(gi, ci, s=size, c=[color], edgecolors='#555555',
                             linewidth=0.4, zorder=3)

        ax_d.set_xticks(range(len(top_genes)))
        ax_d.set_xticklabels(top_genes, rotation=60, ha='right', fontsize=6)
        ax_d.set_yticks(range(n_cl))
        ax_d.set_yticklabels([f'Cluster {c}' for c in unique_cl], fontsize=7)
        ax_d.set_title('Marker Gene Dot Plot\n(size=% expressing, colour=mean exp)',
                       fontsize=10, fontweight='bold')
        ax_d.grid(True, alpha=0.2, linewidth=0.5)

        # Colourbar
        sm = cm.ScalarMappable(cmap=dot_cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax_d, label='Normalised Expr.', fraction=0.046, pad=0.04)
    else:
        ax_d.text(0.5, 0.5, 'Insufficient common genes', ha='center', va='center',
                  transform=ax_d.transAxes)
        ax_d.axis('off')

    fig.suptitle('scCytoTrek — Bulk RNA Alignment to Single-Cell Reference',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved bulk alignment figure to {save}")
    if show:
        plt.show()
    plt.close()
