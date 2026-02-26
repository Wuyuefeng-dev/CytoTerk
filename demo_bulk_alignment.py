"""
Demo: Bulk RNA-seq Alignment to Single-Cell Reference (no scanpy UMAP — uses sklearn t-SNE)
Output: demo_figs/bulk_alignment.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
sc.settings.n_jobs = 1

from sccytotrek.datasets.mock_data import make_mock_scrna


def _tsne_from_pca(X_pca, n_comps=30, perplexity=30, random_state=42):
    """Compute t-SNE on PCA embedding — avoids numba/UMAP segfault."""
    from sklearn.manifold import TSNE
    n = X_pca.shape[0]
    n_comp = min(n_comps, X_pca.shape[1])
    perp    = min(perplexity, max(5, n // 5))
    kwargs  = dict(n_components=2, perplexity=perp, random_state=random_state)
    try:
        tsne = TSNE(max_iter=500, **kwargs)
    except TypeError:
        tsne = TSNE(n_iter=500, **kwargs)
    return tsne.fit_transform(X_pca[:, :n_comp])


def make_mock_bulk_samples(adata_sc, n_bulk=8, random_state=42):
    """Make bulk samples as weighted cluster averages + noise."""
    rng = np.random.default_rng(random_state)
    clusters = sorted(adata_sc.obs['cluster'].unique())
    X_sc = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else adata_sc.X
    profiles, names = [], []
    for i in range(n_bulk):
        n_cl = rng.integers(1, 3)
        chosen  = list(rng.choice(clusters, size=n_cl, replace=False))
        weights = rng.dirichlet(np.ones(n_cl))
        expr = np.zeros(X_sc.shape[1])
        for cl, w in zip(chosen, weights):
            mask = adata_sc.obs['cluster'].values == cl
            expr += w * X_sc[mask].mean(axis=0)
        expr = np.clip(expr + rng.normal(0, 0.3 * expr.std(), expr.shape), 0, None)
        profiles.append(expr.astype(np.float32))
        names.append(f"Bulk{i+1}_C{'+'.join(chosen)}")
    obs = pd.DataFrame({'sample': names}, index=names)
    return ad.AnnData(X=np.vstack(profiles), obs=obs,
                      var=pd.DataFrame(index=adata_sc.var_names))


def project_bulk(adata_sc, adata_bulk, n_neighbors=15):
    """Project bulk onto SC PCA space via PCA loadings, then t-SNE."""
    from sklearn.neighbors import NearestNeighbors

    common = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    print(f"  Projecting {adata_bulk.n_obs} bulk samples using {len(common)} shared genes.")

    sc_sub   = adata_sc[:, common]
    bulk_sub = adata_bulk[:, common]
    X_sc     = sc_sub.X.toarray()  if sp.issparse(sc_sub.X)   else sc_sub.X.copy()
    X_bulk   = bulk_sub.X.toarray() if sp.issparse(bulk_sub.X) else bulk_sub.X.copy()

    sc_means = X_sc.mean(0); sc_stds = X_sc.std(0); sc_stds[sc_stds == 0] = 1
    X_bulk_s = (X_bulk - sc_means) / sc_stds

    loadings = sc_sub.varm['PCs']              # (n_genes_sub, n_pcs)
    n_pcs    = min(loadings.shape[1], adata_sc.obsm['X_pca'].shape[1])
    bulk_pca = X_bulk_s @ loadings[:, :n_pcs]
    adata_bulk.obsm['X_pca'] = bulk_pca.astype(np.float32)

    # kNN UMAP regression: predict bulk t-SNE from SC t-SNE
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nn.fit(adata_sc.obsm['X_pca'][:, :n_pcs])
    dists, idxs = nn.kneighbors(bulk_pca)
    weights = 1.0 / (dists + 1e-9)
    weights /= weights.sum(1, keepdims=True)
    sc_tsne  = adata_sc.obsm['X_tsne']
    bulk_tsne = np.einsum('bi,bik->bk',
                          weights,
                          sc_tsne[idxs])   # weighted average of neighbour coords
    adata_bulk.obsm['X_tsne'] = bulk_tsne.astype(np.float32)

    # Cluster composition
    sc_clusters = adata_sc.obs['cluster'].values.astype(str)
    unique_cl   = sorted(np.unique(sc_clusters))
    comp = np.zeros((adata_bulk.n_obs, len(unique_cl)))
    for i, nn_idx in enumerate(idxs):
        nn_cl = sc_clusters[nn_idx]
        for j, cl in enumerate(unique_cl):
            comp[i, j] = (nn_cl == cl).sum() / len(nn_idx)
    adata_bulk.uns['nn_cluster_composition'] = pd.DataFrame(
        comp, index=adata_bulk.obs_names, columns=unique_cl)
    return adata_bulk


def make_plot(adata_sc, adata_bulk, save="demo_figs/bulk_alignment.png"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scipy.sparse as sp

    umap_sc   = adata_sc.obsm['X_tsne']
    umap_bulk = adata_bulk.obsm['X_tsne']
    n_bulk    = adata_bulk.n_obs
    cluster_labels = adata_sc.obs['cluster'].values.astype(str)
    unique_cl = sorted(np.unique(cluster_labels))
    n_cl = len(unique_cl)
    cmap_cl  = plt.cm.get_cmap('tab10', n_cl)
    cl_colors = {c: cmap_cl(i) for i, c in enumerate(unique_cl)}
    bulk_names = adata_bulk.obs['sample'].tolist()
    bulk_cmap  = plt.cm.get_cmap('Set1', n_bulk)

    fig = plt.figure(figsize=(22, 18), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # ── Panel A: SC t-SNE + Bulk overlay ─────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor('white')
    for cl in unique_cl:
        mask = cluster_labels == cl
        ax_a.scatter(umap_sc[mask, 0], umap_sc[mask, 1],
                     c=[cl_colors[cl]], s=5, alpha=0.5, rasterized=True, label=f'C{cl}')
    for i in range(n_bulk):
        ax_a.scatter(umap_bulk[i, 0], umap_bulk[i, 1], s=280, marker='*',
                     color=bulk_cmap(i), edgecolors='#333', linewidth=0.8, zorder=10,
                     label=bulk_names[i])
    ax_a.set_title('Bulk Samples on SC t-SNE\n(★ = bulk sample, dots = single cells)',
                   fontsize=11, fontweight='bold')
    ax_a.set_xlabel('t-SNE 1', fontsize=9); ax_a.set_ylabel('t-SNE 2', fontsize=9)
    handles, labels = [], []
    for cl in unique_cl:
        handles.append(plt.Line2D([0], [0], marker='o', color=cl_colors[cl],
                                  linestyle='none', markersize=5))
        labels.append(f'Cluster {cl}')
    for i in range(n_bulk):
        handles.append(plt.Line2D([0], [0], marker='*', color=bulk_cmap(i),
                                  linestyle='none', markersize=10, markeredgecolor='#333'))
        labels.append(bulk_names[i])
    ax_a.legend(handles, labels, fontsize=5.5, ncol=2, frameon=False, loc='upper right')

    # ── Panel B: Per-bulk cluster composition pie charts ─────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor('white'); ax_b.axis('off')
    ax_b.set_title('Neighbourhood Cluster Composition',
                   fontsize=11, fontweight='bold', y=1.0)
    comp_df = adata_bulk.uns['nn_cluster_composition']
    pie_cols = 4
    pie_rows = int(np.ceil(n_bulk / pie_cols))
    pie_gs   = gridspec.GridSpecFromSubplotSpec(pie_rows, pie_cols,
                                                subplot_spec=gs[0, 1],
                                                hspace=0.7, wspace=0.5)
    for bi in range(n_bulk):
        ri, ci = divmod(bi, pie_cols)
        ax_p = fig.add_subplot(pie_gs[ri, ci])
        ax_p.set_facecolor('white')
        vals   = comp_df.iloc[bi].values
        colors = [cl_colors[c] for c in comp_df.columns]
        ax_p.pie(vals, colors=colors, startangle=90,
                 wedgeprops=dict(linewidth=0.3, edgecolor='white'))
        ax_p.set_title(bulk_names[bi][:12], fontsize=6.5, pad=2)
    patches = [plt.matplotlib.patches.Patch(color=cl_colors[c], label=f'C{c}')
               for c in unique_cl]
    ax_b.legend(handles=patches, loc='lower center', ncol=n_cl,
                fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.05))

    # ── Panel C: Bulk × cluster correlation heatmap ───────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    common = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    X_sc  = adata_sc[:, common].X
    X_sc  = X_sc.toarray() if sp.issparse(X_sc) else X_sc
    X_bk  = adata_bulk[:, common].X
    X_bk  = X_bk.toarray() if sp.issparse(X_bk) else X_bk
    pseudobulk = np.stack([X_sc[adata_sc.obs['cluster'].values == cl].mean(0)
                           for cl in unique_cl])
    corr = np.zeros((n_bulk, n_cl))
    for i in range(n_bulk):
        for j in range(n_cl):
            if pseudobulk[j].std() > 0 and X_bk[i].std() > 0:
                corr[i, j] = np.corrcoef(X_bk[i], pseudobulk[j])[0, 1]
    im = ax_c.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax_c, label='Pearson r', fraction=0.046, pad=0.04)
    ax_c.set_xticks(range(n_cl)); ax_c.set_xticklabels([f'C{c}' for c in unique_cl],
                                                         fontsize=8, rotation=45)
    ax_c.set_yticks(range(n_bulk)); ax_c.set_yticklabels(bulk_names, fontsize=7)
    ax_c.set_title('Bulk ↔ SC Cluster Pearson Correlation',
                   fontsize=11, fontweight='bold')
    ax_c.set_facecolor('white')

    # ── Panel D: Marker gene dot plot ─────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('white')
    gene_var = X_sc.var(0); top_idx = np.argsort(gene_var)[::-1][:20]
    top_genes = [list(common)[i] for i in top_idx]
    dot_means = np.stack([X_sc[adata_sc.obs['cluster'].values == cl][:, top_idx].mean(0)
                          for cl in unique_cl])
    dot_frac  = np.stack([(X_sc[adata_sc.obs['cluster'].values == cl][:, top_idx] > 0).mean(0)
                          for cl in unique_cl])
    mu_min, mu_max = dot_means.min(0), dot_means.max(0)
    mu_rng = mu_max - mu_min; mu_rng[mu_rng == 0] = 1
    dot_norm = (dot_means - mu_min) / mu_rng
    cmap_d = plt.cm.YlOrRd
    from matplotlib.colors import Normalize; from matplotlib import cm
    norm = Normalize(0, 1)
    for gi, gene in enumerate(top_genes):
        for ci, cl in enumerate(unique_cl):
            ax_d.scatter(gi, ci, s=dot_frac[ci, gi] * 250,
                         c=[cmap_d(norm(dot_norm[ci, gi]))],
                         edgecolors='#555', linewidth=0.3, zorder=3)
    ax_d.set_xticks(range(len(top_genes)))
    ax_d.set_xticklabels(top_genes, rotation=55, ha='right', fontsize=6)
    ax_d.set_yticks(range(n_cl))
    ax_d.set_yticklabels([f'Cluster {c}' for c in unique_cl], fontsize=7)
    ax_d.set_title('Top Variable Gene Dot Plot\n(size=% expr., colour=norm. mean)',
                   fontsize=10, fontweight='bold')
    ax_d.grid(True, alpha=0.2, linewidth=0.5)
    sm = cm.ScalarMappable(cmap=cmap_d, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax_d, label='Normalised Expr.', fraction=0.046, pad=0.04)

    fig.suptitle('scCytoTrek — Bulk RNA-seq Alignment\n(SeuratExtend-style, PCA-loading projection)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save) if os.path.dirname(save) else '.', exist_ok=True)
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save}")
    plt.close()


def main():
    os.makedirs("demo_figs", exist_ok=True)
    print("Generating SC reference (1500 cells, 2000 genes, 5 clusters)...")
    adata_sc = make_mock_scrna(n_cells=1500, n_genes=2000, n_clusters=5, random_state=42)

    print("Preprocessing...")
    sc.pp.normalize_total(adata_sc, target_sum=1e4)
    sc.pp.log1p(adata_sc)
    sc.pp.highly_variable_genes(adata_sc, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pp.pca(adata_sc, n_comps=30)

    print("Computing t-SNE on SC reference (numba-free)...")
    adata_sc.obsm['X_tsne'] = _tsne_from_pca(adata_sc.obsm['X_pca'], n_comps=30,
                                              perplexity=30)
    print(f"  SC t-SNE: {adata_sc.obsm['X_tsne'].shape}")

    print("Generating 8 mock bulk samples...")
    adata_bulk = make_mock_bulk_samples(adata_sc, n_bulk=8, random_state=42)

    print("Projecting bulk onto SC space...")
    adata_bulk = project_bulk(adata_sc, adata_bulk, n_neighbors=15)

    print("Generating SeuratExtend-style 4-panel figure...")
    make_plot(adata_sc, adata_bulk, save="demo_figs/bulk_alignment.png")

    print("\nDone! ✓ demo_figs/bulk_alignment.png" if os.path.exists(
        "demo_figs/bulk_alignment.png") else "\nFailed to save figure.")


if __name__ == "__main__":
    main()
