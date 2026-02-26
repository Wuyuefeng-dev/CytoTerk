"""
Pseudotime gene ordering + heatmap.

Functions
---------
find_pseudotime_genes
    Identify genes whose expression correlates with pseudotime using
    Spearman rank correlation. Returns a ranked DataFrame.

plot_pseudotime_heatmap
    Smoothed gene-expression heatmap ordered by pseudotime (cells binned
    along X-axis, top-correlated genes along Y-axis). SeuratExtend style.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse as sp


def find_pseudotime_genes(
    adata,
    pseudotime_key: str = "dpt_pseudotime",
    n_top: int = 50,
    use_highly_variable: bool = True,
    min_expr_fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Identify genes whose expression changes significantly along pseudotime.

    Uses Spearman rank correlation between each gene's expression and the
    pseudotime values, then returns the top `n_top` genes by |correlation|.

    Parameters
    ----------
    adata            : AnnData (log-normalised recommended).
    pseudotime_key   : obs column containing pseudotime values.
    n_top            : Number of top correlated genes to return.
    use_highly_variable : If True and adata.var['highly_variable'] exists,
                       restrict to HVGs for speed.
    min_expr_fraction : Drop genes expressed in fewer than this fraction of
                       cells (avoiding sparse noise genes).

    Returns
    -------
    pd.DataFrame with columns: gene, spearman_r, pval_approx, direction
    """
    from scipy.stats import spearmanr

    pt = adata.obs[pseudotime_key].values.astype(float)
    # Remove cells with NaN pseudotime
    valid = ~np.isnan(pt)
    pt    = pt[valid]

    X = adata.X[valid, :]
    if sp.issparse(X):
        X = X.toarray()

    genes = np.array(adata.var_names)

    if use_highly_variable and "highly_variable" in adata.var.columns:
        hv_mask = adata.var["highly_variable"].values
    else:
        hv_mask = np.ones(len(genes), dtype=bool)

    # Expression fraction filter
    expr_frac = (X > 0).mean(0)
    keep_mask = hv_mask & (expr_frac >= min_expr_fraction)
    X_sub  = X[:, keep_mask]
    g_sub  = genes[keep_mask]

    print(f"  Computing Spearman r for {X_sub.shape[1]} genes vs pseudotime "
          f"({valid.sum()} cells)...")

    rs, pvals = [], []
    for i in range(X_sub.shape[1]):
        r, p = spearmanr(pt, X_sub[:, i])
        rs.append(r)
        pvals.append(p)

    rs    = np.array(rs)
    pvals = np.array(pvals)

    df = pd.DataFrame({
        "gene":        g_sub,
        "spearman_r":  rs,
        "pval_approx": pvals,
        "direction":   ["up" if r > 0 else "down" for r in rs],
    })
    df["abs_r"] = df["spearman_r"].abs()
    df = df.sort_values("abs_r", ascending=False).head(n_top).reset_index(drop=True)
    df = df.drop(columns="abs_r")

    n_up   = (df["direction"] == "up").sum()
    n_down = (df["direction"] == "down").sum()
    print(f"  Top {len(df)} pseudotime genes: {n_up} up-regulated, "
          f"{n_down} down-regulated along pseudotime.")
    return df


def plot_pseudotime_heatmap(
    adata,
    gene_df: pd.DataFrame | None = None,
    pseudotime_key: str = "dpt_pseudotime",
    n_top: int = 40,
    n_bins: int = 50,
    smooth_window: int = 5,
    cluster_genes: bool = True,
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Plot a smoothed gene-expression heatmap ordered by pseudotime, in
    SeuratExtend style.

    Cells are binned along the pseudotime axis; expression is averaged and
    optionally smoothed within each bin. Genes are optionally hierarchically
    clustered for cleaner visual grouping.

    Parameters
    ----------
    adata          : AnnData (log-normalised).
    gene_df        : Output of find_pseudotime_genes. If None, it is
                     computed automatically with n_top genes.
    pseudotime_key : obs column with pseudotime.
    n_top          : Number of genes if gene_df is None.
    n_bins         : Number of pseudotime bins.
    smooth_window  : Rolling-mean window (bins) for smoothing.
    cluster_genes  : If True, hierarchically cluster gene rows.
    save / show    : Save path and display flag.
    """
    import matplotlib
    matplotlib.use("Agg") if save else None
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize

    try:
        from sccytotrek.plotting.style import (
            apply_seurat_theme, FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
            SEURAT_DISCRETE, SEURAT_FEATURE_CMAP,
        )
    except ImportError:
        # Fallback if style module not available
        FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE = "white", 13, 9, 8
        SEURAT_DISCRETE = ["#4DBBD5", "#E64B35", "#00A087", "#3C5488"]
        SEURAT_FEATURE_CMAP = "RdYlBu_r"
        def apply_seurat_theme(ax, **kw): return ax

    # 1. Pseudotime gene table
    if gene_df is None:
        gene_df = find_pseudotime_genes(
            adata, pseudotime_key=pseudotime_key, n_top=n_top)

    genes = gene_df["gene"].tolist()

    # 2. Order cells by pseudotime
    pt     = adata.obs[pseudotime_key].values.astype(float)
    valid  = ~np.isnan(pt)
    pt_v   = pt[valid]
    order  = np.argsort(pt_v)
    pt_ord = pt_v[order]

    X_full = adata[:, genes].X
    if sp.issparse(X_full):
        X_full = X_full.toarray()
    X_ord = X_full[np.where(valid)[0][order], :]  # (n_valid_cells, n_genes)

    # 3. Bin and average
    bin_edges = np.linspace(pt_ord.min(), pt_ord.max(), n_bins + 1)
    bin_means = np.zeros((n_bins, len(genes)))
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask   = (pt_ord >= lo) & (pt_ord <= hi)
        if mask.sum() > 0:
            bin_means[b, :] = X_ord[mask, :].mean(0)

    # 4. Smooth along pseudotime axis (rolling mean)
    if smooth_window > 1:
        df_tmp = pd.DataFrame(bin_means)
        bin_means = df_tmp.rolling(smooth_window, min_periods=1,
                                   center=True).mean().values

    # 5. Z-score across bins per gene (rows = genes, cols = bins)
    mat = bin_means.T  # (n_genes, n_bins)
    row_mean = mat.mean(1, keepdims=True)
    row_std  = mat.std(1, keepdims=True)
    row_std[row_std == 0] = 1
    mat_z    = (mat - row_mean) / row_std

    # 6. Optionally cluster gene rows
    if cluster_genes and len(genes) > 2:
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
            from scipy.spatial.distance import pdist
            Z = linkage(pdist(mat_z, "correlation"), method="average")
            row_order = leaves_list(Z)
            mat_z = mat_z[row_order, :]
            genes  = [genes[i] for i in row_order]
            gene_df = gene_df.iloc[row_order].reset_index(drop=True)
        except Exception:
            pass

    # 7. Cluster colour bar (obs column)
    has_cluster = any(k in adata.obs.columns for k in
                      ["cluster", "leiden", "louvain", "cell_type"])
    cl_key = next((k for k in ["cluster", "leiden", "louvain", "cell_type"]
                   if k in adata.obs.columns), None)

    # ── Figure layout ─────────────────────────────────────────────────────
    n_genes = len(genes)
    fig_h   = max(8, n_genes * 0.28 + 3)
    fig     = plt.figure(figsize=(16, fig_h), facecolor=FIG_BG)

    # GridSpec: top bar (pseudotime), main heatmap, right bar (direction)
    gs = gridspec.GridSpec(
        3, 2,
        height_ratios=[0.12, 1, 0.08],
        width_ratios=[1, 0.04],
        hspace=0.04, wspace=0.04,
        figure=fig
    )

    # Top colour bar — cluster composition per bin
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.set_facecolor(FIG_BG)
    ax_top.axis("off")

    if cl_key:
        cl_vals = adata.obs[cl_key].values.astype(str)[np.where(valid)[0][order]]
        unique_cl = sorted(np.unique(cl_vals))
        cl_colors = {
            c: SEURAT_DISCRETE[i % len(SEURAT_DISCRETE)]
            for i, c in enumerate(unique_cl)
        }
        # Build bin-level cluster composition  (n_bins, n_clusters)
        bar_mat = np.zeros((1, n_bins, len(unique_cl)))
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask   = (pt_ord >= lo) & (pt_ord <= hi)
            for j, cl in enumerate(unique_cl):
                bar_mat[0, b, j] = (cl_vals[mask] == cl).mean() if mask.sum() else 0

        left = np.zeros(n_bins)
        for j, cl in enumerate(unique_cl):
            vals = bar_mat[0, :, j]
            ax_top.barh([0] * n_bins, vals, left=left, height=1,
                         color=cl_colors[cl], label=cl, edgecolor="none")
            left += vals
        ax_top.set_xlim(0, 1); ax_top.set_ylim(-0.5, 0.5)
        ax_top.set_title("Pseudotime-Ordered Gene Expression Heatmap",
                          fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=6)
        # Legend for cluster bar
        import matplotlib.patches as mpatches
        hs = [mpatches.Patch(color=cl_colors[c], label=c) for c in unique_cl]
        ax_top.legend(handles=hs, loc="upper right", ncol=min(6, len(unique_cl)),
                      fontsize=TICK_SIZE - 1, frameon=False, bbox_to_anchor=(1, 2.2))
    else:
        ax_top.set_title("Pseudotime-Ordered Gene Expression Heatmap",
                          fontsize=TITLE_SIZE, fontweight="bold", loc="left", pad=6)

    # Main heatmap
    ax_hm = fig.add_subplot(gs[1, 0])
    apply_seurat_theme(ax_hm, spines="none")
    vmax  = min(3, np.nanpercentile(np.abs(mat_z), 98))
    im    = ax_hm.imshow(mat_z, aspect="auto", cmap=SEURAT_FEATURE_CMAP,
                          vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax_hm.set_yticks(range(n_genes))
    ax_hm.set_yticklabels(genes, fontsize=max(5, TICK_SIZE - 1))
    # Colour gene labels by direction
    for tick, gene_name in zip(ax_hm.get_yticklabels(), genes):
        row = gene_df[gene_df["gene"] == gene_name]
        if not row.empty:
            tick.set_color("#E64B35" if row.iloc[0]["direction"] == "up" else "#4DBBD5")

    ax_hm.set_xticks([])
    ax_hm.set_xlabel(f"Pseudotime →  ({n_bins} bins, window={smooth_window})",
                      fontsize=LABEL_SIZE)

    # Pseudotime gradient bar (bottom)
    ax_pt = fig.add_subplot(gs[2, 0])
    apply_seurat_theme(ax_pt, spines="none")
    pt_gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax_pt.imshow(pt_gradient, aspect="auto", cmap="viridis")
    ax_pt.set_xticks([]); ax_pt.set_yticks([])
    ax_pt.set_xlabel("", fontsize=LABEL_SIZE)
    ax_pt.text(0, 1.1, "Early", transform=ax_pt.transAxes,
               fontsize=TICK_SIZE, ha="left", va="bottom")
    ax_pt.text(1, 1.1, "Late", transform=ax_pt.transAxes,
               fontsize=TICK_SIZE, ha="right", va="bottom")

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cb, label="Z-score", extend="both")
    ax_cb.yaxis.label.set_fontsize(TICK_SIZE)
    ax_cb.tick_params(labelsize=TICK_SIZE - 1)

    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
        print(f"  Saved pseudotime heatmap: {save}")
    if show:
        plt.show()
    plt.close()
