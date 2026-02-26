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
    show_barcode_timeline: bool = True,
    show: bool = True,
    save: str = None,
):
    """
    Clonal lineage dynamics along pseudotime — streamgraph + barcode timeline.

    Top panel   : Symmetric streamgraph showing each clone's proportion along
                  pseudotime. Smoothed with Gaussian kernel.
    Middle panel: Barcode event timeline — a rug/scatter plot where each clone
                  gets a horizontal lane showing the pseudotime span of its
                  cells (first → last appearance), plus the distribution of
                  individual cell positions as tick marks.
    Bottom row  : UMAP (pseudotime + clone identity) and clone-composition pie.

    Parameters
    ----------
    show_barcode_timeline : bool
        Whether to include the barcode event timeline panel.
    """
    from scipy.ndimage import gaussian_filter1d

    try:
        from sccytotrek.plotting.style import (
            apply_seurat_theme, SEURAT_DISCRETE,
            FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
        )
        _npg = SEURAT_DISCRETE
    except ImportError:
        FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE = "white", 13, 9, 8
        _npg = ["#4DBBD5","#E64B35","#00A087","#3C5488","#F39B7F",
                "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]
        def apply_seurat_theme(ax, **kw): return ax

    if pseudotime_key not in adata.obs:
        if "X_umap" in adata.obsm:
            print(f"'{pseudotime_key}' not found. Deriving from UMAP PC1...")
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

    # Top clones
    all_counts = pd.Series(barcodes).value_counts()
    top_clones = list(all_counts.head(top_n_clones).index)
    mapped     = np.where(pd.Series(barcodes).isin(top_clones), barcodes, "Other")
    clone_labels = top_clones[:]
    if "Other" in mapped:  clone_labels.append("Other")
    if "Unknown" in mapped: clone_labels.append("Unknown")

    # Proportion matrix (clones × bins)
    counts = np.zeros((len(clone_labels), n_bins))
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any(): continue
        for ci, cl in enumerate(clone_labels):
            counts[ci, b] = (mapped[mask] == cl).sum()

    bin_totals = counts.sum(axis=0); bin_totals[bin_totals == 0] = 1
    props        = counts / bin_totals[np.newaxis, :]
    props_smooth = np.array([gaussian_filter1d(props[i], sigma=smooth_sigma)
                              for i in range(len(clone_labels))])

    # SeuratExtend colours
    colors = []
    for i, cl in enumerate(clone_labels):
        if cl == "Other":    colors.append("#AAAAAA")
        elif cl == "Unknown": colors.append("#DDDDDD")
        else: colors.append(_npg[i % len(_npg)])

    # ── Layout ────────────────────────────────────────────────────────────────
    n_rows   = 4 if show_barcode_timeline else 3
    heights  = [3, 1.5, 1.5, 1.5] if show_barcode_timeline else [3, 1.5, 1.5]
    fig, axes_all = plt.subplots(
        n_rows, 1, figsize=(18, sum(heights) + 1),
        facecolor=FIG_BG,
        gridspec_kw={"height_ratios": heights, "hspace": 0.45}
    )
    ax_stream = axes_all[0]
    ax_tl     = axes_all[1] if show_barcode_timeline else None
    ax_bot1   = axes_all[2 if show_barcode_timeline else 1]
    ax_bot2   = axes_all[3 if show_barcode_timeline else 2]

    # ── Streamgraph ───────────────────────────────────────────────────────────
    apply_seurat_theme(ax_stream, spines="bl")
    cum      = props_smooth.sum(axis=0)
    baseline = -cum / 2
    bottom   = baseline.copy()
    for i, (cl, col) in enumerate(zip(clone_labels, colors)):
        top = bottom + props_smooth[i]
        ax_stream.fill_between(bin_mids, bottom, top, color=col,
                               alpha=0.88, label=cl, linewidth=0)
        ax_stream.plot(bin_mids, top, color="white", linewidth=0.4, alpha=0.5)
        bottom = top
    ax_stream.set_xlim(bin_mids[0], bin_mids[-1])
    ax_stream.set_title("Clonal Dynamics Along Pseudotime",
                         fontsize=TITLE_SIZE, fontweight="bold")
    ax_stream.set_ylabel("Clone Proportion", fontsize=LABEL_SIZE)
    ax_stream.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i])
               for i in range(len(clone_labels))]
    ax_stream.legend(handles, clone_labels, loc="upper right",
                     fontsize=TICK_SIZE, ncol=3, frameon=False,
                     handlelength=1.2, handletextpad=0.5)

    # ── Barcode event timeline ─────────────────────────────────────────────────
    if show_barcode_timeline and ax_tl is not None:
        apply_seurat_theme(ax_tl, spines="bl", grid=True)
        for yi, (cl, col) in enumerate(zip(clone_labels, colors)):
            if cl in ("Other", "Unknown"): continue
            mask_cl  = barcodes == cl
            pt_cl    = pseudotime[mask_cl]
            if len(pt_cl) == 0: continue
            pt_min, pt_max = pt_cl.min(), pt_cl.max()
            # Horizontal span line
            ax_tl.plot([pt_min, pt_max], [yi, yi],
                       color=col, linewidth=2.5, solid_capstyle="round", alpha=0.8)
            # Individual cell rug marks
            ax_tl.scatter(pt_cl, np.full_like(pt_cl, yi),
                          color=col, s=8, alpha=0.5, zorder=4, edgecolors="none")
            # Start / end dots
            ax_tl.scatter([pt_min, pt_max], [yi, yi],
                          color=col, s=40, zorder=5,
                          edgecolors="white", linewidth=0.8)
            ax_tl.text(pt_max + (bins[-1] - bins[0]) * 0.01, yi,
                       cl, fontsize=TICK_SIZE - 1, va="center", color=col,
                       fontweight="bold")
        ax_tl.set_xlim(bins[0], bins[-1] * 1.12)
        ax_tl.set_yticks(range(len([c for c in clone_labels
                                    if c not in ("Other", "Unknown")])))
        ax_tl.set_yticklabels([c for c in clone_labels
                                if c not in ("Other", "Unknown")],
                               fontsize=TICK_SIZE, color="#333333")
        ax_tl.set_xlabel(f"Pseudotime  ({pseudotime_key})", fontsize=LABEL_SIZE)
        ax_tl.set_title("Barcode Event Timeline  "
                          "(each lane = clone span; rug marks = individual cells)",
                          fontsize=TITLE_SIZE - 1, fontweight="bold")

    # ── Bottom: pseudotime UMAP + clone UMAP ─────────────────────────────────
    for ax in (ax_bot1, ax_bot2):
        apply_seurat_theme(ax, spines="none")

    if "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]
        sc1  = ax_bot1.scatter(umap[:, 0], umap[:, 1], c=pseudotime,
                                cmap="viridis", s=5, alpha=0.7, rasterized=True,
                                edgecolors="none")
        plt.colorbar(sc1, ax=ax_bot1, label="Pseudotime", fraction=0.046, pad=0.04)
        ax_bot1.set_title("Pseudotime on UMAP", fontsize=LABEL_SIZE + 1,
                           fontweight="bold")
        ax_bot1.set_xlabel("UMAP 1", fontsize=LABEL_SIZE)
        ax_bot1.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)

        clone_cmap  = {cl: colors[i] for i, cl in enumerate(clone_labels)}
        cell_colors = [clone_cmap.get(m, "#AAAAAA") for m in mapped]
        ax_bot2.scatter(umap[:, 0], umap[:, 1], c=cell_colors,
                         s=5, alpha=0.7, rasterized=True, edgecolors="none")
        ax_bot2.set_title("Clonal Identity on UMAP", fontsize=LABEL_SIZE + 1,
                           fontweight="bold")
        ax_bot2.set_xlabel("UMAP 1", fontsize=LABEL_SIZE)
        ax_bot2.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)
    else:
        for ax in (ax_bot1, ax_bot2):
            ax.text(0.5, 0.5, "No UMAP data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=LABEL_SIZE)
            ax.axis("off")

    fig.suptitle("scCytoTrek — Clonal Lineage Tracing × Pseudotime",
                 fontsize=TITLE_SIZE + 1, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=180, bbox_inches="tight", facecolor=FIG_BG)
        print(f"Saved clonal streamgraph to {save}")
    if show:
        plt.show()
    plt.close()

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
