"""
Cell-cell interaction UMAP arc visualisation.

plot_cell2cell_umap
    Draws Bézier arc lines between cluster centroids on the UMAP embedding.
    Line colour encodes interaction type (ligand-receptor pair family) and
    line thickness encodes interaction strength (score or -log10 p-value).
    Arrow direction shows sender → receiver.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def plot_cell2cell_umap(
    adata,
    lr_results: pd.DataFrame,
    groupby: str = "leiden_0.5",
    score_col: str | None = None,
    pval_col: str | None = None,
    top_n: int = 15,
    line_width_scale: float = 4.0,
    alpha: float = 0.80,
    arc_height: float = 0.35,
    embedding_key: str = "X_umap",
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    Draw Bézier arc lines between cluster centroids on a UMAP embedding.

    Line width  ∝ interaction strength (score or -log10 p-value).
    Line colour = unique colour per sender cluster (SeuratExtend NPG palette).
    Arrow head  = receiver end (directed arc).

    Parameters
    ----------
    adata           : AnnData with obsm[embedding_key] and obs[groupby].
    lr_results      : DataFrame from run_cellphonedb_scoring.
                      Must contain 'source', 'target', and either
                      score_col or pval_col column.
    groupby         : adata.obs key used as sender/receiver labels.
    score_col       : Column in lr_results for interaction score.
    pval_col        : Column for p-value (will be -log10 transformed).
    top_n           : Show only the top-N strongest interactions.
    line_width_scale: Scale factor for line widths.
    alpha           : Transparency of arc lines.
    arc_height      : Bézier curve bow height (fraction of distance).
    embedding_key   : obsm key for the 2-D embedding.
    save / show     : Save path and display flag.
    """
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.patheffects as pe

    try:
        from sccytotrek.plotting.style import (
            apply_seurat_theme, SEURAT_DISCRETE,
            FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE,
        )
    except ImportError:
        FIG_BG, TITLE_SIZE, LABEL_SIZE, TICK_SIZE = "white", 13, 9, 8
        SEURAT_DISCRETE = ["#4DBBD5","#E64B35","#00A087","#3C5488","#F39B7F",
                           "#8491B4","#91D1C2","#DC0000","#7E6148","#B09C85"]
        def apply_seurat_theme(ax, **kw): return ax

    # ── Determine interaction strength ──────────────────────────────────────
    df = lr_results.copy()

    if score_col and score_col in df.columns:
        df["_strength"] = df[score_col].astype(float)
    elif pval_col and pval_col in df.columns:
        pv = df[pval_col].astype(float).clip(1e-300, 1)
        df["_strength"] = -np.log10(pv)
    else:
        # Try to auto-detect
        for cand in ["score", "interaction_score", "mean", "pvalue", "p_value", "pval"]:
            if cand in df.columns:
                if "pval" in cand or cand.startswith("p"):
                    pv = df[cand].astype(float).clip(1e-300, 1)
                    df["_strength"] = -np.log10(pv)
                else:
                    df["_strength"] = df[cand].astype(float)
                break
        else:
            df["_strength"] = 1.0  # uniform if nothing found

    # Source / target columns
    for src_col in ["source", "sender", "source_cluster", "cell_type_1"]:
        if src_col in df.columns:
            df = df.rename(columns={src_col: "_source"})
            break
    for tgt_col in ["target", "receiver", "target_cluster", "cell_type_2"]:
        if tgt_col in df.columns:
            df = df.rename(columns={tgt_col: "_target"})
            break
    if "_source" not in df.columns or "_target" not in df.columns:
        print("lr_results missing 'source'/'target' columns — cannot plot arcs.")
        return

    # Keep top-N by strength
    df = df.nlargest(top_n, "_strength").reset_index(drop=True)

    # ── Compute cluster centroids ───────────────────────────────────────────
    emb = adata.obsm[embedding_key]
    cl_vals = adata.obs[groupby].astype(str).values
    unique_cl = sorted(np.unique(cl_vals))
    centroids = {cl: emb[cl_vals == cl].mean(0) for cl in unique_cl}

    # ── Assign colours by source cluster ───────────────────────────────────
    cl_color = {
        cl: SEURAT_DISCRETE[i % len(SEURAT_DISCRETE)]
        for i, cl in enumerate(unique_cl)
    }

    # ── Normalise line widths ───────────────────────────────────────────────
    s_vals = df["_strength"].values.astype(float)
    s_min, s_max = s_vals.min(), s_vals.max()
    if s_max > s_min:
        lw_norm = (s_vals - s_min) / (s_max - s_min)
    else:
        lw_norm = np.ones_like(s_vals)
    lw_scaled = 0.5 + lw_norm * line_width_scale   # range [0.5, 0.5+scale]

    # ── Draw ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 9), facecolor=FIG_BG)
    apply_seurat_theme(ax, spines="none")

    # BG scatter — all cells
    for cl in unique_cl:
        m = cl_vals == cl
        ax.scatter(emb[m, 0], emb[m, 1],
                   c=[cl_color[cl]], s=5, alpha=0.45,
                   rasterized=True, edgecolors="none", label=cl)

    # Cluster centroid labels
    for cl, (cx, cy) in centroids.items():
        ax.text(cx, cy, cl, fontsize=TICK_SIZE + 1, fontweight="bold",
                ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7,
                          ec=cl_color[cl], lw=1))

    # Bézier arc for each interaction
    for idx, row in df.iterrows():
        src, tgt = str(row["_source"]), str(row["_target"])
        if src not in centroids or tgt not in centroids:
            continue
        p0 = np.array(centroids[src])
        p2 = np.array(centroids[tgt])
        if np.allclose(p0, p2):
            continue

        # Bézier control point: perpendicular offset
        mid    = (p0 + p2) / 2
        perp   = np.array([-(p2 - p0)[1], (p2 - p0)[0]])
        norm   = np.linalg.norm(perp)
        if norm > 0:
            perp = perp / norm
        p1 = mid + perp * arc_height * np.linalg.norm(p2 - p0)

        # Bézier verts
        n_pts   = 80
        t       = np.linspace(0, 1, n_pts)
        bx      = (1 - t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
        by      = (1 - t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]

        color   = cl_color.get(src, "#888888")
        lw      = float(lw_scaled[idx])

        ax.plot(bx, by, color=color, linewidth=lw,
                alpha=alpha, solid_capstyle="round", zorder=3,
                path_effects=[pe.Stroke(linewidth=lw + 0.4, foreground="white", alpha=0.3),
                               pe.Normal()])

        # Arrowhead at target end
        dx = bx[-1] - bx[-5]; dy = by[-1] - by[-5]
        ax.annotate("", xy=(p2[0], p2[1]),
                    xytext=(bx[-5], by[-5]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw * 0.6, mutation_scale=10 + lw * 2),
                    zorder=4)

    ax.set_title("Cell-Cell Interaction Map\n(arc width = interaction strength, "
                 "colour = sender cluster)",
                 fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=LABEL_SIZE)
    ax.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)

    # Cluster colour legend
    handles = [mpatches.Patch(color=cl_color[cl], label=f"Cluster {cl}")
               for cl in unique_cl]
    ax.legend(handles=handles, fontsize=TICK_SIZE, frameon=False,
              loc="lower right", ncol=2)

    # Interaction strength scale bar (inset axes)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax, width="25%", height="4%", loc="upper left")
    import matplotlib.cm as mpl_cm
    from matplotlib.colors import Normalize as MNorm
    sm = mpl_cm.ScalarMappable(
        cmap=plt.cm.Greens,
        norm=MNorm(vmin=s_min, vmax=s_max))
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_ins, orientation="horizontal")
    cb.set_label("Interaction Strength", fontsize=TICK_SIZE - 1)
    cb.ax.tick_params(labelsize=TICK_SIZE - 2)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
        print(f"  Saved CCI UMAP: {save}")
    if show:
        plt.show()
    plt.close()
