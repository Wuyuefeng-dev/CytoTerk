"""
Generate all walkthrough figures with SeuratExtend-style aesthetics:
  1. demo_figs/bulk_alignment.png
  2. demo_figs/tipping_genes_barplot.png
  3. demo_figs/tf_score_ranking.png
"""
import os, warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
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
from matplotlib.colors import Normalize
from matplotlib import cm as mpl_cm

from sccytotrek.datasets.mock_data import make_mock_scrna
from sccytotrek.plotting.style import (
    apply_seurat_theme, seurat_figure,
    SEURAT_DISCRETE, SEURAT_CORR_CMAP,
    SEURAT_DOTPLOT_CMAP, SEURAT_ENTROPY_CMAP,
    SEURAT_FEATURE_CMAP,
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
    BULK_SIZE, FIG_BG,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def tsne2d(X_pca, perplexity=30):
    from sklearn.manifold import TSNE
    n = X_pca.shape[0]; n_comp = min(30, X_pca.shape[1])
    perp = min(perplexity, max(5, n // 5))
    kw = dict(n_components=2, perplexity=perp, random_state=42)
    try:   tsne = TSNE(max_iter=500, **kw)
    except TypeError: tsne = TSNE(n_iter=500, **kw)
    return tsne.fit_transform(X_pca[:, :n_comp])


def make_sc(n_cells=1200, n_genes=1500, n_clusters=5):
    adata = make_mock_scrna(n_cells=n_cells, n_genes=n_genes,
                            n_clusters=n_clusters, random_state=42)
    if 'cluster' not in adata.obs:
        adata.obs['cluster'] = adata.obs.iloc[:, 0].values
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5,
                                 n_top_genes=min(500, adata.n_vars))
    sc.pp.pca(adata, n_comps=min(30, adata.n_obs - 1, adata.n_vars - 1))
    adata.obsm['X_tsne'] = tsne2d(adata.obsm['X_pca']).astype(np.float32)
    return adata


def _cl_colors(unique_cl):
    """Map cluster IDs to SeuratExtend NPG colors."""
    n = len(unique_cl)
    colors = SEURAT_DISCRETE * ((n // len(SEURAT_DISCRETE)) + 1)
    return {c: colors[i] for i, c in enumerate(sorted(unique_cl))}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Bulk Alignment (4-panel, SeuratExtend style)
# ─────────────────────────────────────────────────────────────────────────────

def make_bulk_samples(adata_sc, n_bulk=8):
    rng = np.random.default_rng(42)
    clusters = sorted(adata_sc.obs['cluster'].unique())
    X_sc = adata_sc.X.toarray() if sp.issparse(adata_sc.X) else adata_sc.X.copy()
    profiles, names = [], []
    for i in range(n_bulk):
        chosen = list(rng.choice(clusters, size=rng.integers(1, 3), replace=False))
        w = rng.dirichlet(np.ones(len(chosen)))
        expr = sum(w[j] * X_sc[adata_sc.obs['cluster'].values == cl].mean(0)
                   for j, cl in enumerate(chosen))
        std = expr.std() or 1.0
        expr = np.clip(expr + rng.normal(0, 0.3 * std, expr.shape), 0, None)
        profiles.append(expr.astype(np.float32))
        names.append(f"Bulk{i+1}_{'&'.join(str(c) for c in chosen)}")
    obs = pd.DataFrame({'sample': names}, index=names)
    return ad.AnnData(X=np.vstack(profiles), obs=obs,
                      var=pd.DataFrame(index=adata_sc.var_names))


def project_bulk(adata_sc, adata_bulk, n_neighbors=15):
    from sklearn.neighbors import NearestNeighbors
    common = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    X_sc_c = adata_sc[:, common].X
    X_sc_c = X_sc_c.toarray() if sp.issparse(X_sc_c) else X_sc_c
    X_bk_c = adata_bulk[:, common].X
    X_bk_c = X_bk_c.toarray() if sp.issparse(X_bk_c) else X_bk_c
    mu, std = X_sc_c.mean(0), X_sc_c.std(0); std[std == 0] = 1
    X_bk_s = (X_bk_c - mu) / std
    sc_sub = adata_sc[:, common]
    L = sc_sub.varm['PCs'] if 'PCs' in sc_sub.varm else adata_sc.varm['PCs'][
        [list(adata_sc.var_names).index(g) for g in common], :]
    n_pcs = min(L.shape[1], adata_sc.obsm['X_pca'].shape[1])
    bulk_pca = X_bk_s @ L[:, :n_pcs]
    adata_bulk.obsm['X_pca'] = bulk_pca.astype(np.float32)
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(adata_sc.obsm['X_pca'][:, :n_pcs])
    dists, idxs = nn.kneighbors(bulk_pca)
    w = 1 / (dists + 1e-9); w /= w.sum(1, keepdims=True)
    adata_bulk.obsm['X_tsne'] = (w[:, :, None] * adata_sc.obsm['X_tsne'][idxs]).sum(1).astype(np.float32)
    sc_cl = adata_sc.obs['cluster'].values.astype(str)
    uq_cl = sorted(np.unique(sc_cl))
    comp = np.array([[(sc_cl[idx] == cl).mean() for cl in uq_cl] for idx in idxs])
    adata_bulk.uns['nn_cluster_composition'] = pd.DataFrame(comp, index=adata_bulk.obs_names, columns=uq_cl)
    return adata_bulk, uq_cl


def fig_bulk_alignment(adata_sc, adata_bulk, unique_cl, save):
    umap_sc  = adata_sc.obsm['X_tsne']
    umap_bk  = adata_bulk.obsm['X_tsne']
    n_bulk   = adata_bulk.n_obs
    n_cl     = len(unique_cl)
    cl_col   = _cl_colors(unique_cl)
    comp_df  = adata_bulk.uns['nn_cluster_composition']
    bulk_nm  = adata_bulk.obs['sample'].tolist()
    # Bulk dot colours: also SeuratExtend (offset into palette)
    bulk_colors = [(SEURAT_DISCRETE * 3)[i % (len(SEURAT_DISCRETE) * 3)] for i in range(n_cl, n_cl + n_bulk)]

    fig = plt.figure(figsize=(22, 18), facecolor=FIG_BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # A — SC embedding + bulk ★
    ax_a = fig.add_subplot(gs[0, 0])
    apply_seurat_theme(ax_a, spines="none")
    cl_vals = adata_sc.obs['cluster'].values.astype(str)
    for cl in sorted(unique_cl):
        m = cl_vals == cl
        ax_a.scatter(umap_sc[m, 0], umap_sc[m, 1],
                     c=[cl_col[cl]], s=5, alpha=0.65,
                     rasterized=True, label=f'Cluster {cl}', edgecolors='none')
    for i in range(n_bulk):
        ax_a.scatter(umap_bk[i, 0], umap_bk[i, 1],
                     s=BULK_SIZE, marker='*', color=bulk_colors[i],
                     edgecolors='#444444', linewidth=0.9, zorder=10,
                     label=bulk_nm[i])
    ax_a.set_title('Bulk Samples on SC Reference', fontsize=TITLE_SIZE, fontweight='bold', pad=10)
    ax_a.set_xlabel('t-SNE 1', fontsize=LABEL_SIZE); ax_a.set_ylabel('t-SNE 2', fontsize=LABEL_SIZE)
    h, lb = ax_a.get_legend_handles_labels()
    ax_a.legend(h, lb, fontsize=6, ncol=2, frameon=False, loc='upper right',
                markerscale=2, handletextpad=0.4)

    # B — Pie charts panel
    ax_b = fig.add_subplot(gs[0, 1])
    apply_seurat_theme(ax_b, spines='none'); ax_b.axis('off')
    ax_b.set_title('Neighbourhood Cluster Composition', fontsize=TITLE_SIZE, fontweight='bold', y=1.0)
    pie_gs = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0, 1], hspace=0.8, wspace=0.5)
    for bi in range(n_bulk):
        ri, ci = divmod(bi, 4)
        ax_p = fig.add_subplot(pie_gs[ri, ci])
        ax_p.set_facecolor(FIG_BG)
        vals = comp_df.iloc[bi].values
        wedge_colors = [cl_col[c] for c in comp_df.columns]
        ax_p.pie(vals, colors=wedge_colors, startangle=90,
                 wedgeprops=dict(linewidth=0.4, edgecolor='white'))
        ax_p.set_title(bulk_nm[bi][:13], fontsize=6.5, pad=2)
    import matplotlib.patches as mpatches
    leg_patches = [mpatches.Patch(color=cl_col[c], label=f'C{c}') for c in unique_cl]
    ax_b.legend(handles=leg_patches, loc='lower center', ncol=n_cl,
                fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

    # C — Pearson correlation heatmap
    ax_c = fig.add_subplot(gs[1, 0]); apply_seurat_theme(ax_c, spines='all')
    common_g = np.intersect1d(adata_sc.var_names, adata_bulk.var_names)
    X_sc2 = adata_sc[:, common_g].X; X_sc2 = X_sc2.toarray() if sp.issparse(X_sc2) else X_sc2
    X_bk2 = adata_bulk[:, common_g].X; X_bk2 = X_bk2.toarray() if sp.issparse(X_bk2) else X_bk2
    pseudobulk = np.stack([X_sc2[cl_vals == cl].mean(0) for cl in unique_cl])
    corr = np.zeros((n_bulk, n_cl))
    for i in range(n_bulk):
        for j in range(n_cl):
            if pseudobulk[j].std() > 0 and X_bk2[i].std() > 0:
                corr[i, j] = np.corrcoef(X_bk2[i], pseudobulk[j])[0, 1]
    im = ax_c.imshow(corr, cmap=SEURAT_CORR_CMAP, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax_c, label='Pearson r', fraction=0.046, pad=0.04)
    ax_c.set_xticks(range(n_cl)); ax_c.set_xticklabels([f'C{c}' for c in unique_cl],
                                                         fontsize=TICK_SIZE, rotation=45, ha='right')
    ax_c.set_yticks(range(n_bulk)); ax_c.set_yticklabels(bulk_nm, fontsize=TICK_SIZE)
    ax_c.set_title('Bulk ↔ SC Cluster Correlation', fontsize=TITLE_SIZE, fontweight='bold')

    # D — Dot plot
    ax_d = fig.add_subplot(gs[1, 1]); apply_seurat_theme(ax_d, grid=True)
    top_idx   = np.argsort(X_sc2.var(0))[::-1][:20]
    top_genes = [list(common_g)[i] for i in top_idx]
    dot_mean  = np.stack([X_sc2[cl_vals == cl][:, top_idx].mean(0) for cl in unique_cl])
    dot_frac  = np.stack([(X_sc2[cl_vals == cl][:, top_idx] > 0).mean(0) for cl in unique_cl])
    d_rng = dot_mean.max(0) - dot_mean.min(0); d_rng[d_rng == 0] = 1
    dot_norm = (dot_mean - dot_mean.min(0)) / d_rng
    dp_cmap = plt.cm.get_cmap(SEURAT_DOTPLOT_CMAP)
    norm = Normalize(0, 1)
    for gi, gene in enumerate(top_genes):
        for ci_, cl in enumerate(unique_cl):
            ax_d.scatter(gi, ci_, s=dot_frac[ci_, gi] * 280,
                         c=[dp_cmap(norm(dot_norm[ci_, gi]))],
                         edgecolors='#888888', linewidth=0.25, zorder=3)
    ax_d.set_xticks(range(len(top_genes)))
    ax_d.set_xticklabels(top_genes, rotation=55, ha='right', fontsize=6)
    ax_d.set_yticks(range(n_cl))
    ax_d.set_yticklabels([f'Cluster {c}' for c in unique_cl], fontsize=TICK_SIZE)
    ax_d.set_title('Marker Dot Plot  (size = % expr · colour = norm. mean)',
                   fontsize=TITLE_SIZE - 1, fontweight='bold')
    sm = mpl_cm.ScalarMappable(cmap=dp_cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax_d, label='Normalised Expression', fraction=0.046, pad=0.04)

    fig.suptitle('Bulk RNA-seq Alignment to Single-Cell Reference',
                 fontsize=TITLE_SIZE + 1, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=FIG_BG); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Tipping Genes (entropy curve + gene barplot)
# ─────────────────────────────────────────────────────────────────────────────

def fig_tipping_genes(adata_sc, save):
    from sccytotrek.trajectory.tipping_point import compute_sandpile_entropy

    rng = np.random.default_rng(77)
    adata_sc.obs['pseudo_t'] = np.sort(rng.random(adata_sc.n_obs))

    result = compute_sandpile_entropy(
        adata_sc, pseudotime_key='pseudo_t', n_bins=30, correlation_threshold=0.3)
    tg_df  = result['tipping_genes']
    tp_bin = result['tipping_point_bin']
    ents   = result['entropy']

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), facecolor=FIG_BG)

    # Entropy curve
    ax = axes[0]; apply_seurat_theme(ax, grid=True)
    bins = np.arange(len(ents))
    ax.fill_between(bins, ents, alpha=0.25, color=SEURAT_DISCRETE[2])
    ax.plot(bins, ents, color=SEURAT_DISCRETE[2], linewidth=2.0)
    ax.axvline(tp_bin, color=SEURAT_DISCRETE[1], linewidth=2.0, linestyle='--',
               label=f'Tipping point (bin {tp_bin}, entropy={ents[tp_bin]:.2f})')
    ax.set_xlabel('Pseudotime Bin', fontsize=LABEL_SIZE)
    ax.set_ylabel('Network Entropy (bits)', fontsize=LABEL_SIZE)
    ax.set_title('Sandpile Network Entropy Along Pseudotime',
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=TICK_SIZE, frameon=False)

    # Gene barplot
    ax2 = axes[1]; apply_seurat_theme(ax2, grid=True, spines='bl')
    top_n = min(20, len(tg_df))
    df    = tg_df.head(top_n).sort_values('degree_weight')
    # gradient from low (light teal) to high (dark navy) using two NPG colors
    n_colors = len(df)
    bar_colors = plt.cm.get_cmap('YlOrRd', n_colors)(np.linspace(0.2, 0.9, n_colors))
    ax2.barh(df['gene'], df['degree_weight'], color=bar_colors,
             edgecolor='none', height=0.75)
    ax2.set_xlabel('Network Hub Degree', fontsize=LABEL_SIZE)
    ax2.set_title(f'Top {top_n} Genes Driving Tipping Point (bin {tp_bin})',
                  fontsize=TITLE_SIZE, fontweight='bold')

    fig.suptitle('Tipping Point Analysis — Sandpile Model',
                 fontsize=TITLE_SIZE + 1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=FIG_BG); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: TF Score Ranking (heatmap + bar)
# ─────────────────────────────────────────────────────────────────────────────

def fig_tf_ranking(adata_sc, save):
    from sccytotrek.grn.tf_enrichment import run_tf_enrichment

    rng = np.random.default_rng(42)
    genes = list(adata_sc.var_names)
    n_tfs = 30
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
    adata_sc = run_tf_enrichment(adata_sc, tf_network=tf_net,
                                 source_col='tf', target_col='target',
                                 weight_col='weight', min_expr_fraction=0.01)
    tf_df   = adata_sc.obsm['X_tf_activity']
    tf_mat  = tf_df.values if isinstance(tf_df, pd.DataFrame) else np.array(tf_df)
    tf_cols = list(tf_df.columns) if isinstance(tf_df, pd.DataFrame) else [f'TF{i}' for i in range(tf_mat.shape[1])]

    cl_vals  = adata_sc.obs['cluster'].values.astype(str)
    unique_cl = sorted(np.unique(cl_vals))
    n_cl      = len(unique_cl)

    mean_scores = np.stack([tf_mat[cl_vals == cl].mean(0) for cl in unique_cl])
    tf_var      = mean_scores.var(0)
    top_idx     = np.argsort(tf_var)[::-1][:20]
    top_tfs     = [tf_cols[i] for i in top_idx]
    heat_mat    = mean_scores[:, top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), facecolor=FIG_BG)

    # Heatmap
    ax1 = axes[0]; apply_seurat_theme(ax1, spines='all')
    mn, mx = heat_mat.min(), heat_mat.max()
    norm_heat = (heat_mat - mn) / (mx - mn + 1e-9)
    im = ax1.imshow(norm_heat, cmap=SEURAT_FEATURE_CMAP, aspect='auto',
                    interpolation='nearest')
    plt.colorbar(im, ax=ax1, label='Normalised TF Activity', fraction=0.046, pad=0.04)
    ax1.set_xticks(range(len(top_tfs)))
    ax1.set_xticklabels(top_tfs, rotation=55, ha='right', fontsize=TICK_SIZE - 1)
    ax1.set_yticks(range(n_cl))
    ax1.set_yticklabels([f'Cluster {c}' for c in unique_cl], fontsize=TICK_SIZE)
    # Colour-coded y-tick labels by cluster
    cl_col = _cl_colors(unique_cl)
    for tick, cl in zip(ax1.get_yticklabels(), unique_cl):
        tick.set_color(cl_col[cl])
        tick.set_fontweight('bold')
    ax1.set_title('TF Activity Heatmap\n(Top 20 discriminative TFs × Cluster)',
                  fontsize=TITLE_SIZE, fontweight='bold')

    # Bar ranking
    ax2 = axes[1]; apply_seurat_theme(ax2, grid=True, spines='bl')
    global_mean = tf_mat.mean(0)
    rank_idx    = np.argsort(global_mean)[::-1][:20]
    rank_tfs    = [tf_cols[i] for i in rank_idx]
    rank_scores = global_mean[rank_idx]
    n_bars      = len(rank_tfs)
    # SeuratExtend NPG gradient per rank
    bar_colors  = [SEURAT_DISCRETE[i % len(SEURAT_DISCRETE)] for i in range(n_bars)]
    ax2.barh(list(reversed(rank_tfs)), list(reversed(rank_scores)),
             color=list(reversed(bar_colors)), edgecolor='none', height=0.72)
    ax2.set_xlabel('Mean TF Activity Score', fontsize=LABEL_SIZE)
    ax2.set_title('Top 20 TFs — Global Activity Ranking',
                  fontsize=TITLE_SIZE, fontweight='bold')

    fig.suptitle('TF Enrichment Score ranking  ·  scCytoTrek GRN',
                 fontsize=TITLE_SIZE + 1, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight', facecolor=FIG_BG); plt.close()
    print(f"  ✓ {save}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('demo_figs', exist_ok=True)

    print("Building SC reference (1200 cells × 1500 genes)...")
    adata_sc = make_sc(n_cells=1200, n_genes=1500, n_clusters=5)

    print("\n[1/3] Bulk alignment...")
    adata_bulk = make_bulk_samples(adata_sc, n_bulk=8)
    adata_bulk, uq_cl = project_bulk(adata_sc, adata_bulk)
    fig_bulk_alignment(adata_sc, adata_bulk, uq_cl, 'demo_figs/bulk_alignment.png')

    print("\n[2/3] Tipping genes...")
    fig_tipping_genes(adata_sc.copy(), 'demo_figs/tipping_genes_barplot.png')

    print("\n[3/3] TF score ranking...")
    fig_tf_ranking(adata_sc.copy(), 'demo_figs/tf_score_ranking.png')

    print("\nDone:")
    for f in ['demo_figs/bulk_alignment.png',
              'demo_figs/tipping_genes_barplot.png',
              'demo_figs/tf_score_ranking.png']:
        print(f"  {'✓' if os.path.exists(f) else '✗'} {f}")
