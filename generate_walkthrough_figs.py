"""
Generate all missing walkthrough figures:
  1. demo_figs/bulk_alignment.png         — SeuratExtend-style 4-panel
  2. demo_figs/tipping_genes_barplot.png  — top genes at tipping point
  3. demo_figs/tf_score_ranking.png       — TF activity heatmap + bar rankings

Numba/UMAP is avoided entirely — uses sklearn t-SNE instead.
"""
import os, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
sc.settings.n_jobs = 1
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sccytotrek.datasets.mock_data import make_mock_scrna


# ─── helpers ──────────────────────────────────────────────────────────────────

def tsne2d(X_pca, n_comps=30, perplexity=30):
    from sklearn.manifold import TSNE
    n = X_pca.shape[0]; n_comp = min(n_comps, X_pca.shape[1])
    perp = min(perplexity, max(5, n // 5))
    kw = dict(n_components=2, perplexity=perp, random_state=42)
    try:   tsne = TSNE(max_iter=500, **kw)
    except TypeError: tsne = TSNE(n_iter=500, **kw)
    return tsne.fit_transform(X_pca[:, :n_comp])


def make_sc(n_cells=1200, n_genes=1500, n_clusters=5):
    adata = make_mock_scrna(n_cells=n_cells, n_genes=n_genes,
                            n_clusters=n_clusters, random_state=42)
    # The mock dataset stores cluster as 'cluster' — verify:
    clus_key = 'cluster' if 'cluster' in adata.obs else adata.obs.columns[0]
    if clus_key != 'cluster':
        adata.obs['cluster'] = adata.obs[clus_key].values
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                 n_top_genes=min(500, adata.n_vars))
    sc.pp.pca(adata, n_comps=min(30, adata.n_obs - 1, adata.n_vars - 1))
    adata.obsm['X_tsne'] = tsne2d(adata.obsm['X_pca']).astype(np.float32)
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Bulk Alignment (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def make_bulk_samples(adata_sc, n_bulk=8):
    rng = np.random.default_rng(42)
    clusters = sorted(adata_sc.obs['cluster'].unique())
    X_sc = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else adata_sc.X.copy()
    profiles, names = [], []
    for i in range(n_bulk):
        n_cl   = rng.integers(1, 3)
        chosen = list(rng.choice(clusters, size=n_cl, replace=False))
        w      = rng.dirichlet(np.ones(n_cl))
        expr   = sum(w[j] * X_sc[adata_sc.obs['cluster'].values == cl].mean(0)
                     for j, cl in enumerate(chosen))
        expr   = np.clip(expr + rng.normal(0, 0.3 * (expr.std() or 1), expr.shape), 0, None)
        profiles.append(expr.astype(np.float32))
        names.append(f"Bulk{i+1}_C{'+'.join(str(c) for c in chosen)}")
    obs = pd.DataFrame({'sample': names}, index=names)
    return ad.AnnData(X=np.vstack(profiles), obs=obs,
                      var=pd.DataFrame(index=adata_sc.var_names))


def project_bulk(adata_sc, adata_bulk, n_neighbors=15):
    from sklearn.neighbors import NearestNeighbors
    common   = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    X_sc_c   = adata_sc[:, common].X
    X_sc_c   = X_sc_c.toarray() if sp.issparse(X_sc_c) else X_sc_c
    X_bk_c   = adata_bulk[:, common].X
    X_bk_c   = X_bk_c.toarray() if sp.issparse(X_bk_c) else X_bk_c
    mu, std  = X_sc_c.mean(0), X_sc_c.std(0); std[std == 0] = 1
    X_bk_s   = (X_bk_c - mu) / std
    # PCA loadings aligned to common genes
    sc_genes  = list(adata_sc[:, common].var_names)
    if 'PCs' in adata_sc[:, common].varm:
        L = adata_sc[:, common].varm['PCs']
    else:
        L = adata_sc.varm['PCs'][[list(adata_sc.var_names).index(g) for g in sc_genes], :]
    n_pcs    = min(L.shape[1], adata_sc.obsm['X_pca'].shape[1])
    bulk_pca = X_bk_s @ L[:, :n_pcs]
    adata_bulk.obsm['X_pca'] = bulk_pca.astype(np.float32)
    # kNN interpolation into t-SNE space
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(adata_sc.obsm['X_pca'][:, :n_pcs])
    dists, idxs = nn.kneighbors(bulk_pca)
    w = 1 / (dists + 1e-9); w /= w.sum(1, keepdims=True)
    sc_tsne  = adata_sc.obsm['X_tsne']
    adata_bulk.obsm['X_tsne'] = (w[:, :, None] * sc_tsne[idxs]).sum(1).astype(np.float32)
    # Cluster composition
    sc_cl    = adata_sc.obs['cluster'].values.astype(str)
    uq_cl    = sorted(np.unique(sc_cl))
    comp     = np.array([[(sc_cl[idx] == cl).mean() for cl in uq_cl] for idx in idxs])
    adata_bulk.uns['nn_cluster_composition'] = pd.DataFrame(
        comp, index=adata_bulk.obs_names, columns=uq_cl)
    return adata_bulk, uq_cl


def fig_bulk_alignment(adata_sc, adata_bulk, unique_cl, save):
    umap_sc  = adata_sc.obsm['X_tsne']
    umap_bk  = adata_bulk.obsm['X_tsne']
    n_bulk   = adata_bulk.n_obs
    n_cl     = len(unique_cl)
    cmap_cl  = plt.cm.get_cmap('tab10', n_cl)
    cl_col   = {c: cmap_cl(i) for i, c in enumerate(unique_cl)}
    bulk_col = plt.cm.get_cmap('Set1', n_bulk)
    bulk_nm  = adata_bulk.obs['sample'].tolist()
    comp_df  = adata_bulk.uns['nn_cluster_composition']

    fig = plt.figure(figsize=(22, 18), facecolor='white')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)

    # A — SC t-SNE + bulk stars
    ax_a = fig.add_subplot(gs[0, 0]); ax_a.set_facecolor('white')
    for cl in unique_cl:
        m = adata_sc.obs['cluster'].values.astype(str) == cl
        ax_a.scatter(umap_sc[m, 0], umap_sc[m, 1], c=[cl_col[cl]],
                     s=5, alpha=0.5, rasterized=True, label=f'C{cl}')
    for i in range(n_bulk):
        ax_a.scatter(umap_bk[i, 0], umap_bk[i, 1], s=280, marker='*',
                     color=bulk_col(i), edgecolors='#333', linewidth=0.8, zorder=10,
                     label=bulk_nm[i])
    ax_a.set_title('Bulk on SC t-SNE (★ = bulk, dots = cells)',
                   fontsize=11, fontweight='bold')
    ax_a.set_xlabel('t-SNE 1'); ax_a.set_ylabel('t-SNE 2')
    h, lb = ax_a.get_legend_handles_labels()
    ax_a.legend(h, lb, fontsize=5.5, ncol=2, frameon=False, loc='upper right')

    # B — Pie charts
    ax_b = fig.add_subplot(gs[0, 1]); ax_b.set_facecolor('white'); ax_b.axis('off')
    ax_b.set_title('Cluster Composition per Bulk Sample', fontsize=11, fontweight='bold')
    pie_gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 1],
                                              hspace=0.7, wspace=0.5)
    for bi in range(n_bulk):
        ri, ci = divmod(bi, 4)
        ax_p = fig.add_subplot(pie_gs[ri, ci]); ax_p.set_facecolor('white')
        vals = comp_df.iloc[bi].values
        ax_p.pie(vals, colors=[cl_col[c] for c in comp_df.columns],
                 startangle=90, wedgeprops=dict(linewidth=0.3, edgecolor='white'))
        ax_p.set_title(bulk_nm[bi][:12], fontsize=6.5, pad=2)
    ax_b.legend(
        handles=[plt.matplotlib.patches.Patch(color=cl_col[c], label=f'C{c}')
                 for c in unique_cl],
        loc='lower center', ncol=n_cl, fontsize=6, frameon=False,
        bbox_to_anchor=(0.5, -0.08))

    # C — Correlation heatmap
    ax_c = fig.add_subplot(gs[1, 0]); ax_c.set_facecolor('white')
    common_g = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    X_sc2    = adata_sc[:, common_g].X
    X_sc2    = X_sc2.toarray() if sp.issparse(X_sc2) else X_sc2
    X_bk2    = adata_bulk[:, common_g].X
    X_bk2    = X_bk2.toarray() if sp.issparse(X_bk2) else X_bk2
    pseudobulk = np.stack([X_sc2[adata_sc.obs['cluster'].values.astype(str) == cl].mean(0)
                            for cl in unique_cl])
    corr = np.zeros((n_bulk, n_cl))
    for i in range(n_bulk):
        for j in range(n_cl):
            if pseudobulk[j].std() > 0 and X_bk2[i].std() > 0:
                corr[i, j] = np.corrcoef(X_bk2[i], pseudobulk[j])[0, 1]
    im = ax_c.imshow(corr, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax_c, label='Pearson r', fraction=0.046, pad=0.04)
    ax_c.set_xticks(range(n_cl)); ax_c.set_xticklabels([f'C{c}' for c in unique_cl],
                                                         fontsize=8, rotation=45)
    ax_c.set_yticks(range(n_bulk)); ax_c.set_yticklabels(bulk_nm, fontsize=7)
    ax_c.set_title('Bulk ↔ SC Cluster Pearson Correlation', fontsize=11, fontweight='bold')

    # D — Dot plot
    ax_d = fig.add_subplot(gs[1, 1]); ax_d.set_facecolor('white')
    top_idx   = np.argsort(X_sc2.var(0))[::-1][:20]
    top_genes = [list(common_g)[i] for i in top_idx]
    cl_vals   = adata_sc.obs['cluster'].values.astype(str)
    dot_mean  = np.stack([X_sc2[cl_vals == cl][:, top_idx].mean(0) for cl in unique_cl])
    dot_frac  = np.stack([(X_sc2[cl_vals == cl][:, top_idx] > 0).mean(0) for cl in unique_cl])
    d_rng     = dot_mean.max(0) - dot_mean.min(0); d_rng[d_rng == 0] = 1
    dot_norm  = (dot_mean - dot_mean.min(0)) / d_rng
    cmap_d    = plt.cm.YlOrRd
    from matplotlib.colors import Normalize; from matplotlib import cm as mpl_cm
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
    ax_d.set_title('Top Variable Gene Dot Plot\n(size = % expr., colour = norm. mean)',
                   fontsize=10, fontweight='bold')
    ax_d.grid(True, alpha=0.2)
    sm = mpl_cm.ScalarMappable(cmap=cmap_d, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax_d, label='Norm. Expr.', fraction=0.046, pad=0.04)

    fig.suptitle('scCytoTrek — Bulk RNA-seq Alignment (SeuratExtend-style)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Tipping Genes Barplot
# ─────────────────────────────────────────────────────────────────────────────

def fig_tipping_genes(adata_sc, save):
    from sccytotrek.trajectory.tipping_point import compute_sandpile_entropy

    # Inject pseudotime as linear ordering (no Diffusion needed)
    rng = np.random.default_rng(77)
    adata_sc.obs['pseudo_t'] = np.sort(rng.random(adata_sc.n_obs))

    result = compute_sandpile_entropy(
        adata_sc, pseudotime_key='pseudo_t', n_bins=30,
        correlation_threshold=0.3
    )
    tg_df  = result['tipping_genes']
    tp_bin = result['tipping_point_bin']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')

    # Panel A — entropy along pseudotime
    ax = axes[0]; ax.set_facecolor('#f8f9fa')
    ents = result['entropy']
    bins = np.arange(len(ents))
    ax.fill_between(bins, ents, alpha=0.35, color='#4e79a7')
    ax.plot(bins, ents, color='#4e79a7', linewidth=2)
    ax.axvline(tp_bin, color='#e15759', linewidth=2, linestyle='--',
               label=f'Tipping pt (bin {tp_bin})')
    ax.set_xlabel('Pseudotime Bin', fontsize=10)
    ax.set_ylabel('Network Entropy', fontsize=10)
    ax.set_title('Sandpile Network Entropy Along Pseudotime', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)

    # Panel B — top genes barplot
    ax2 = axes[1]; ax2.set_facecolor('white')
    top_n = min(20, len(tg_df))
    df    = tg_df.head(top_n).sort_values('degree_weight')
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n))
    bars = ax2.barh(df['gene'], df['degree_weight'], color=colors,
                    edgecolor='grey', linewidth=0.4)
    ax2.set_xlabel('Network Degree (Hub Weight)', fontsize=10)
    ax2.set_title(f'Top {top_n} Tipping Point Driving Genes\n(at pseudotime bin {tp_bin})',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_facecolor('white')

    fig.suptitle('scCytoTrek — Tipping Point Analysis (Sandpile Model)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: TF Score Ranking
# ─────────────────────────────────────────────────────────────────────────────

def fig_tf_ranking(adata_sc, save):
    """Mock TF network + run_tf_enrichment → heatmap + bar ranking figure."""
    from sccytotrek.grn.tf_enrichment import run_tf_enrichment

    rng   = np.random.default_rng(42)
    genes = list(adata_sc.var_names)
    n_tfs = 30

    # Simulate a small TF–target network using real gene names
    tf_names   = rng.choice(genes, size=n_tfs, replace=False).tolist()
    target_pool = [g for g in genes if g not in tf_names]
    rows = []
    for tf in tf_names:
        n_targets = rng.integers(5, 25)
        targets   = rng.choice(target_pool, size=n_targets, replace=False)
        weights   = rng.choice([-1.0, 1.0], size=n_targets)
        for tg, w in zip(targets, weights):
            rows.append({'tf': tf, 'target': tg, 'weight': w})
    tf_net = pd.DataFrame(rows)

    adata_sc = run_tf_enrichment(
        adata_sc, tf_network=tf_net,
        source_col='tf', target_col='target', weight_col='weight',
        min_expr_fraction=0.01
    )
    tf_df = adata_sc.obsm['X_tf_activity']
    if isinstance(tf_df, pd.DataFrame):
        tf_mat = tf_df.values
        tf_cols = tf_df.columns.tolist()
    else:
        tf_mat = np.array(tf_df)
        tf_cols = [f'TF{i}' for i in range(tf_mat.shape[1])]

    cluster_vals = adata_sc.obs['cluster'].values.astype(str)
    unique_cl    = sorted(np.unique(cluster_vals))

    # Mean TF score per cluster
    mean_scores = np.stack([tf_mat[cluster_vals == cl].mean(0) for cl in unique_cl])  # (n_cl, n_tfs)

    # Rank TFs by variance across clusters (most discriminative)
    tf_var   = mean_scores.var(0)
    top_idx  = np.argsort(tf_var)[::-1][:20]
    top_tfs  = [tf_cols[i] for i in top_idx]
    heat_mat = mean_scores[:, top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')

    # Panel A — TF activity heatmap
    ax1 = axes[0]; ax1.set_facecolor('white')
    mn, mx = heat_mat.min(), heat_mat.max()
    if mn == mx: mx = mn + 1e-6
    norm_heat = (heat_mat - mn) / (mx - mn)
    im = ax1.imshow(norm_heat, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, ax=ax1, label='Norm. TF Activity', fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(top_tfs)))
    ax1.set_xticklabels(top_tfs, rotation=55, ha='right', fontsize=7)
    ax1.set_yticks(range(len(unique_cl)))
    ax1.set_yticklabels([f'Cluster {c}' for c in unique_cl], fontsize=8)
    ax1.set_title('TF Activity Heatmap\n(Top 20 discriminative TFs × Cluster)',
                  fontsize=11, fontweight='bold')

    # Panel B — Mean TF activity ranking (global)
    ax2 = axes[1]; ax2.set_facecolor('#f8f9fa')
    global_mean = tf_mat.mean(0)
    rank_idx    = np.argsort(global_mean)[::-1][:20]
    rank_tfs    = [tf_cols[i] for i in rank_idx]
    rank_scores = global_mean[rank_idx]
    colors      = plt.cm.viridis(np.linspace(0.85, 0.2, len(rank_tfs)))
    ax2.barh(rank_tfs[::-1], rank_scores[::-1], color=colors[::-1],
             edgecolor='grey', linewidth=0.4)
    ax2.set_xlabel('Mean TF Activity Score', fontsize=10)
    ax2.set_title('Top 20 TFs by Mean Activity\n(global ranking)',
                  fontsize=11, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    fig.suptitle('scCytoTrek — TF Enrichment Score Ranking',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('demo_figs', exist_ok=True)

    print("Building SC reference...")
    adata_sc = make_sc(n_cells=1200, n_genes=1500, n_clusters=5)
    print(f"  {adata_sc.shape}, clusters: {sorted(adata_sc.obs['cluster'].unique())}")

    print("\n[1/3] Bulk alignment figure...")
    adata_bulk = make_bulk_samples(adata_sc, n_bulk=8)
    adata_bulk, uq_cl = project_bulk(adata_sc, adata_bulk)
    fig_bulk_alignment(adata_sc, adata_bulk, uq_cl, 'demo_figs/bulk_alignment.png')

    print("\n[2/3] Tipping genes figure...")
    fig_tipping_genes(adata_sc.copy(), 'demo_figs/tipping_genes_barplot.png')

    print("\n[3/3] TF score ranking figure...")
    fig_tf_ranking(adata_sc.copy(), 'demo_figs/tf_score_ranking.png')

    print("\nAll done:")
    for f in ['demo_figs/bulk_alignment.png',
              'demo_figs/tipping_genes_barplot.png',
              'demo_figs/tf_score_ranking.png']:
        ok = "✓" if os.path.exists(f) else "✗"
        print(f"  {ok} {f}")
