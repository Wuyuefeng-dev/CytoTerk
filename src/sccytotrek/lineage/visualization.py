import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def plot_lineage_umap(
    adata: ad.AnnData, 
    barcode_key: str = "barcode", 
    status_key: str = "barcode_imputed_status",
    palette: str = "tab20",
    show: bool = True,
    save: str = None
):
    """
    Visualize imputed lineage tracing over the RNA UMAP embedding.
    Highlights cells that have been imputed versus original.
    """
    if "X_umap" not in adata.obsm:
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata)
        
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Plot all clones
    sc.pl.umap(adata, color=barcode_key, ax=axes[0], show=False, palette=palette, legend_loc='on data')
    axes[0].set_title(f"Clonal Assignment ({barcode_key})")
    
    # 2. Plot imputation status
    if status_key in adata.obs:
        sc.pl.umap(adata, color=status_key, ax=axes[1], show=False, palette={"original": "lightgrey", "imputed": "red"})
        axes[1].set_title("Imputation Status")
    else:
        axes[1].axis('off')
        
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    plt.close()

def plot_clone_size_distribution(
    adata: ad.AnnData, 
    barcode_key: str = "barcode", 
    status_key: str = "barcode_imputed_status",
    show: bool = True,
    save: str = None
):
    """
    Plots the distribution of clone sizes before and after imputation.
    """
    if status_key not in adata.obs:
        raise ValueError(f"Status key '{status_key}' not found. Was imputation run?")
        
    counts = adata.obs.groupby([barcode_key, status_key], observed=False).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind='bar', stacked=True, color={'original': '#1f77b4', 'imputed': '#ff7f0e'}, ax=ax)
    
    ax.set_title("Clone Sizes: Original vs Imputed")
    ax.set_xlabel("Clone Barcode")
    ax.set_ylabel("Number of Cells")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Only show top N clones if there are too many
    if len(counts) > 30:
        ax.set_xticklabels([])
        ax.set_xlabel("Clone Barcode (Top 30+)")
        
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        
    if show:
        plt.show()
    plt.close()


def plot_clonal_streamgraph(
    adata: ad.AnnData,
    pseudotime_key: str = "dpt_pseudotime",
    barcode_key: str = "barcode",
    n_bins: int = 40,
    top_n_clones: int = 10,
    smooth_sigma: float = 1.5,
    show: bool = True,
    save: str = None,
):
    """
    Visualize clonal lineage dynamics along a pseudotime axis as a stacked streamgraph.

    Each stream represents one clone. The y-axis is the proportion of cells belonging
    to that clone within each pseudotime bin. Cells with missing barcodes are shown as
    a separate 'Unknown' stream.

    Parameters
    ----------
    adata : AnnData
        Must have `pseudotime_key` in obs and `barcode_key` in obs.
    pseudotime_key : str
        Column in `adata.obs` holding pseudotime values.
    barcode_key : str
        Column in `adata.obs` holding clone barcode labels.
    n_bins : int
        Number of pseudotime bins.
    top_n_clones : int
        Number of most abundant clones to display individually; rest grouped into 'Other'.
    smooth_sigma : float
        Gaussian smoothing sigma for stream curves.
    show : bool
        Display interactively.
    save : str or None
        File path to save the figure.
    """
    from scipy.ndimage import gaussian_filter1d

    if pseudotime_key not in adata.obs:
        if "X_umap" in adata.obsm:
            print(f"'{pseudotime_key}' not found. Deriving rough pseudotime from UMAP PC1...")
            from sklearn.decomposition import PCA as _PCA
            _pt = _PCA(n_components=1).fit_transform(adata.obsm["X_umap"]).flatten()
            _pt = (_pt - _pt.min()) / (_pt.max() - _pt.min() + 1e-9)
            adata.obs["_auto_pseudotime"] = _pt
            pseudotime_key = "_auto_pseudotime"
        else:
            raise ValueError(f"Pseudotime key '{pseudotime_key}' not found.")

    pseudotime = adata.obs[pseudotime_key].values.astype(float)
    barcodes   = adata.obs[barcode_key].fillna("Unknown").replace("NA", "Unknown").values.astype(str)

    # Bin pseudotime
    bins     = np.linspace(pseudotime.min(), pseudotime.max(), n_bins + 1)
    bin_idx  = np.clip(np.digitize(pseudotime, bins) - 1, 0, n_bins - 1)
    bin_mids = 0.5 * (bins[:-1] + bins[1:])

    # Determine top clones
    all_counts   = pd.Series(barcodes).value_counts()
    top_clones   = list(all_counts.head(top_n_clones).index)
    mapped       = np.where(pd.Series(barcodes).isin(top_clones), barcodes, "Other")
    clone_labels = top_clones[:]
    if "Other" in mapped:
        clone_labels.append("Other")
    if "Unknown" in mapped:
        clone_labels.append("Unknown")

    # Count matrix (clones × bins)
    counts = np.zeros((len(clone_labels), n_bins))
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        for ci, cl in enumerate(clone_labels):
            counts[ci, b] = (mapped[mask] == cl).sum()

    bin_totals = counts.sum(axis=0)
    bin_totals[bin_totals == 0] = 1
    props        = counts / bin_totals[np.newaxis, :]
    props_smooth = np.array([gaussian_filter1d(props[i], sigma=smooth_sigma) for i in range(len(clone_labels))])

    # Colours
    cmap   = plt.cm.get_cmap("tab20", len(clone_labels))
    colors = [cmap(i) for i in range(len(clone_labels))]
    if "Other" in clone_labels:
        colors[clone_labels.index("Other")] = "#999999"
    if "Unknown" in clone_labels:
        colors[clone_labels.index("Unknown")] = "#cccccc"

    # ── Figure (white background) ───────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor='white')

    ax_stream = fig.add_subplot(2, 1, 1)
    ax_stream.set_facecolor('white')

    # Symmetric streamgraph baseline
    cum      = props_smooth.sum(axis=0)
    baseline = -cum / 2
    bottom   = baseline.copy()

    for i, (cl, col) in enumerate(zip(clone_labels, colors)):
        top = bottom + props_smooth[i]
        ax_stream.fill_between(bin_mids, bottom, top, color=col, alpha=0.85, label=cl, linewidth=0)
        ax_stream.plot(bin_mids, top, color='white', linewidth=0.5, alpha=0.4)
        bottom = top

    ax_stream.set_xlim(bin_mids[0], bin_mids[-1])
    ax_stream.set_title("Clonal Dynamics Along Pseudotime",
                        color='#111111', fontsize=14, fontweight='bold', pad=12)
    ax_stream.set_xlabel(f"Pseudotime  ({pseudotime_key})", color='#333333', fontsize=11)
    ax_stream.set_ylabel("Clone Proportion", color='#333333', fontsize=11)
    ax_stream.tick_params(colors='#333333')
    ax_stream.set_yticks([])
    for sp in ax_stream.spines.values():
        sp.set_color('#cccccc')

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(clone_labels))]
    ax_stream.legend(handles, clone_labels, loc='upper right', framealpha=0.8,
                     labelcolor='#111111', fontsize=8, ncol=2, borderpad=0.5)

    # ── Lower panels ────────────────────────────────────────────────────────────
    ax_pt  = fig.add_subplot(2, 3, 4)
    ax_cl  = fig.add_subplot(2, 3, 5)
    ax_pie = fig.add_subplot(2, 3, 6)
    for ax in (ax_pt, ax_cl, ax_pie):
        ax.set_facecolor('white')

    if "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]

        sc1 = ax_pt.scatter(umap[:, 0], umap[:, 1], c=pseudotime, cmap='viridis',
                            s=5, alpha=0.7, rasterized=True)
        fig.colorbar(sc1, ax=ax_pt, label='Pseudotime', fraction=0.046, pad=0.04)
        ax_pt.set_title('Pseudotime', fontsize=10, fontweight='bold')
        ax_pt.set_xlabel('UMAP 1', fontsize=9); ax_pt.set_ylabel('UMAP 2', fontsize=9)

        clone_cmap  = {cl: colors[i] for i, cl in enumerate(clone_labels)}
        cell_colors = [clone_cmap.get(m, '#999999') for m in mapped]
        ax_cl.scatter(umap[:, 0], umap[:, 1], c=cell_colors, s=5, alpha=0.7, rasterized=True)
        ax_cl.set_title('Clonal Identity', fontsize=10, fontweight='bold')
        ax_cl.set_xlabel('UMAP 1', fontsize=9); ax_cl.set_ylabel('UMAP 2', fontsize=9)
    else:
        for ax in (ax_pt, ax_cl):
            ax.text(0.5, 0.5, 'No UMAP data', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    pie_sizes = counts.sum(axis=1)
    wedges, texts, autotexts = ax_pie.pie(
        pie_sizes, labels=None, colors=colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75,
        wedgeprops=dict(linewidth=0.5, edgecolor='white')
    )
    for at in autotexts:
        at.set_color('#111111')
        at.set_fontsize(7)
    ax_pie.set_title('Clone Composition', fontsize=10, fontweight='bold')

    fig.suptitle('scCytoTrek — Clonal Lineage Tracing × Pseudotime',
                 fontsize=14, fontweight='bold', y=1.01, color='#111111')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=180, bbox_inches='tight', facecolor='white')
        print(f"Saved clonal streamgraph to {save}")
    if show:
        plt.show()
    plt.close()
