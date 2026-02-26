"""
Identify cell types based on canonical markers (e.g. malignant cells).
"""

from anndata import AnnData
import scanpy as sc
import numpy as np


def score_cell_types(adata: AnnData, marker_dict: dict, groupby: str = 'leiden_0.5') -> AnnData:
    """
    Score and assign putative cell types using a dictionary of marker genes.

    Parameters
    ----------
    adata : AnnData
    marker_dict : dict — {cell_type: [gene1, gene2, ...]}
    groupby : str — cluster key in adata.obs

    Returns
    -------
    AnnData with 'cell_type_prediction' added to adata.obs.
    """
    print("Scoring cell types based on marker signatures...")
    score_names = []
    for celltype, genes in marker_dict.items():
        valid_genes = [g for g in genes if g in adata.var_names]
        if valid_genes:
            score_name = f'score_{celltype}'
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name)
            score_names.append(score_name)

    if not score_names:
        print("No valid marker genes found.")
        return adata

    cluster_mapping = {}
    for cluster in adata.obs[groupby].unique():
        cluster_cells = adata.obs[adata.obs[groupby] == cluster]
        means = {score: cluster_cells[score].mean() for score in score_names}
        best_match = max(means, key=means.get).replace('score_', '')
        cluster_mapping[cluster] = "Unknown" if all(m < 0 for m in means.values()) else best_match

    adata.obs['cell_type_prediction'] = adata.obs[groupby].map(cluster_mapping)
    print(f"Cell types in `adata.obs['cell_type_prediction']`: {set(cluster_mapping.values())}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# CellTypist integration
# ─────────────────────────────────────────────────────────────────────────────

def run_celltypist(
    adata: AnnData,
    model: str = "Immune_All_Low.pkl",
    majority_voting: bool = True,
    groupby: str | None = None,
    over_clustering: str | None = None,
    min_prob: float = 0.0,
    result_key: str = "celltypist",
    fallback_marker_dict: dict | None = None,
) -> AnnData:
    """
    Classify cell types using the CellTypist API.

    Requires: ``pip install celltypist``

    If CellTypist is not installed and ``fallback_marker_dict`` is provided,
    falls back gracefully to :func:`score_cell_types`.

    Parameters
    ----------
    adata          : AnnData — raw or log-normalised counts. CellTypist expects
                     log1p-normalised counts at 10,000 counts/cell; the function
                     auto-normalises if total_counts > 10,001 or not log-normalised.
    model          : CellTypist model name **or** absolute path to a local ``.pkl``.
                     Built-in models include:
                     - ``"Immune_All_Low.pkl"``   — pan-immune, high-granularity
                     - ``"Immune_All_High.pkl"``  — pan-immune, low-granularity
                     - ``"Human_Lung_Atlas.pkl"`` — lung-specific
                     Call ``celltypist.models.get_all_models()`` for the full list.
    majority_voting : bool — if True, run majority voting over clusters to
                     smooth per-cell predictions (recommended for scRNA data).
    groupby        : If majority_voting=True, the obs column to use as over-
                     clustering input (e.g. ``"leiden_0.5"``). Auto-detected if None.
    over_clustering : Explicit over-clustering key; overrides ``groupby``.
    min_prob       : Zero-out predictions below this probability threshold.
    result_key     : Prefix for obs columns added to adata:
                     - ``{result_key}_predicted_labels``
                     - ``{result_key}_majority_voting`` (if majority_voting)
                     - ``{result_key}_conf_score``
    fallback_marker_dict : dict — fallback marker dict for :func:`score_cell_types`
                     when celltypist is unavailable.

    Returns
    -------
    AnnData with CellTypist predictions in adata.obs.
    """
    try:
        import celltypist
        from celltypist import models as ct_models
    except ImportError:
        print("[CellTypist] `celltypist` not installed.  "
              "Install with: pip install celltypist")
        if fallback_marker_dict:
            print("[CellTypist] Falling back to score_cell_types().")
            return score_cell_types(adata, fallback_marker_dict,
                                    groupby=groupby or "leiden_0.5")
        return adata

    # ── Pre-processing check ─────────────────────────────────────────────────
    import scipy.sparse as sp
    X_check = adata.X[:20, :].toarray() if sp.issparse(adata.X) else adata.X[:20, :]
    is_log   = X_check.max() < 20          # heuristic: raw counts typically have large max
    if not is_log:
        print("[CellTypist] Normalising to 10,000 counts/cell and log1p-transforming...")
        adata_ct = adata.copy()
        sc.pp.normalize_total(adata_ct, target_sum=1e4)
        sc.pp.log1p(adata_ct)
    else:
        adata_ct = adata

    # ── Load model ───────────────────────────────────────────────────────────
    import os
    if os.path.isfile(model):
        m = celltypist.Model.load(model)
        print(f"[CellTypist] Loaded local model: {model}")
    else:
        print(f"[CellTypist] Downloading model: {model}")
        ct_models.download_models(model=model, force_update=False)
        m = ct_models.Model.load(model)

    # ── Determine over_clustering key ────────────────────────────────────────
    if majority_voting:
        oc_key = over_clustering or groupby
        if oc_key is None:
            # Auto-detect a cluster column
            for cand in ["leiden_0.5", "leiden", "louvain", "cluster"]:
                if cand in adata.obs.columns:
                    oc_key = cand
                    break
        if oc_key and oc_key not in adata.obs.columns:
            print(f"[CellTypist] over_clustering key '{oc_key}' not found; "
                  "running without majority voting.")
            majority_voting = False
            oc_key = None
    else:
        oc_key = None

    # ── Run prediction ───────────────────────────────────────────────────────
    print(f"[CellTypist] Running prediction with model '{m.description.get('Name', model)}'...")
    predictions = celltypist.annotate(
        adata_ct,
        model=m,
        majority_voting=majority_voting,
        over_clustering=oc_key,
    )

    # ── Write results back to adata ──────────────────────────────────────────
    pred_adata = predictions.to_adata()

    # Copy key prediction columns
    for col in ["predicted_labels", "majority_voting", "conf_score"]:
        src = col
        dst = f"{result_key}_{col}"
        if src in pred_adata.obs.columns:
            adata.obs[dst] = pred_adata.obs[src].values

    # Apply min_prob filter
    if min_prob > 0 and f"{result_key}_conf_score" in adata.obs.columns:
        low = adata.obs[f"{result_key}_conf_score"] < min_prob
        label_col = (f"{result_key}_majority_voting"
                     if majority_voting else f"{result_key}_predicted_labels")
        if label_col in adata.obs.columns:
            adata.obs.loc[low, label_col] = "Unassigned"
        print(f"[CellTypist] {low.sum()} cells set to 'Unassigned' (conf < {min_prob}).")

    final_key = (f"{result_key}_majority_voting"
                 if majority_voting else f"{result_key}_predicted_labels")
    n_types = adata.obs[final_key].nunique() if final_key in adata.obs else "?"
    print(f"[CellTypist] Done. {n_types} cell types assigned in "
          f"adata.obs['{final_key}'].")
    return adata


def plot_celltypist_umap(
    adata: AnnData,
    result_key: str = "celltypist",
    embedding_key: str = "X_umap",
    save: str | None = None,
    show: bool = True,
) -> None:
    """
    SeuratExtend-style 3-panel CellTypist result visualisation.

    Panels
    ------
    A : UMAP coloured by predicted cell type (majority_voting or predicted_labels)
    B : UMAP coloured by confidence score (continuous, YlOrRd)
    C : Violin plot of confidence scores per predicted cell type

    Parameters
    ----------
    adata       : AnnData with CellTypist results in obs.
    result_key  : Prefix used in run_celltypist().
    embedding_key : obsm key for 2-D embedding.
    """
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

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

    # Pick the best label column
    label_col = next(
        (f"{result_key}_{s}" for s in ["majority_voting", "predicted_labels"]
         if f"{result_key}_{s}" in adata.obs.columns),
        None
    )
    conf_col = f"{result_key}_conf_score"

    if label_col is None:
        print(f"No CellTypist columns with prefix '{result_key}' found in adata.obs.")
        return

    emb      = adata.obsm[embedding_key]
    labels   = adata.obs[label_col].astype(str).values
    unique_l = sorted(set(labels))
    n_l      = len(unique_l)
    l_colors = {l: SEURAT_DISCRETE[i % len(SEURAT_DISCRETE)]
                for i, l in enumerate(unique_l)}

    fig = plt.figure(figsize=(22, 7.5), facecolor=FIG_BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # A — Cell type UMAP
    ax_a = fig.add_subplot(gs[0, 0]); apply_seurat_theme(ax_a, spines="none")
    for lb in unique_l:
        m = labels == lb
        ax_a.scatter(emb[m, 0], emb[m, 1], c=[l_colors[lb]], s=4, alpha=0.6,
                     rasterized=True, edgecolors="none", label=lb)
    ax_a.set_title("CellTypist Predicted Types", fontsize=TITLE_SIZE, fontweight="bold")
    ax_a.set_xlabel("UMAP 1", fontsize=LABEL_SIZE); ax_a.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)
    ax_a.legend(fontsize=TICK_SIZE - 1, ncol=2, frameon=False,
                loc="lower right", markerscale=3, handletextpad=0.3)

    # B — Confidence score UMAP
    ax_b = fig.add_subplot(gs[0, 1]); apply_seurat_theme(ax_b, spines="none")
    if conf_col in adata.obs.columns:
        conf = adata.obs[conf_col].values.astype(float)
        sc_b = ax_b.scatter(emb[:, 0], emb[:, 1], c=conf, cmap=SEURAT_EXPR_CMAP,
                             s=4, alpha=0.6, rasterized=True, edgecolors="none",
                             vmin=0, vmax=1)
        plt.colorbar(sc_b, ax=ax_b, label="Confidence Score", fraction=0.046, pad=0.04)
    else:
        ax_b.text(0.5, 0.5, "No confidence scores", ha="center", va="center",
                  transform=ax_b.transAxes)
    ax_b.set_title("Prediction Confidence", fontsize=TITLE_SIZE, fontweight="bold")
    ax_b.set_xlabel("UMAP 1", fontsize=LABEL_SIZE); ax_b.set_ylabel("UMAP 2", fontsize=LABEL_SIZE)

    # C — Confidence violin per cell type
    ax_c = fig.add_subplot(gs[0, 2]); apply_seurat_theme(ax_c, grid=True)
    if conf_col in adata.obs.columns:
        import pandas as pd
        plot_df = adata.obs[[label_col, conf_col]].copy()
        plot_df.columns = ["cell_type", "conf"]
        # Order by median confidence
        med_order = (plot_df.groupby("cell_type")["conf"]
                     .median().sort_values(ascending=False).index.tolist())
        positions = range(len(med_order))
        parts = ax_c.violinplot(
            [plot_df.loc[plot_df["cell_type"] == ct, "conf"].values
             for ct in med_order],
            positions=list(positions), showmedians=True, widths=0.7
        )
        for i, (body, ct) in enumerate(zip(parts["bodies"], med_order)):
            body.set_facecolor(l_colors.get(ct, "#AAAAAA"))
            body.set_alpha(0.75)
        parts["cmedians"].set_color("#333333")
        parts["cbars"].set_color("#AAAAAA")
        parts["cmins"].set_color("#AAAAAA")
        parts["cmaxes"].set_color("#AAAAAA")
        ax_c.set_xticks(list(positions))
        ax_c.set_xticklabels(med_order, rotation=45, ha="right",
                              fontsize=TICK_SIZE - 1)
        ax_c.set_ylabel("Confidence Score", fontsize=LABEL_SIZE)
        ax_c.set_ylim(-0.05, 1.1)
    else:
        ax_c.text(0.5, 0.5, "No confidence data", ha="center", va="center",
                  transform=ax_c.transAxes)
    ax_c.set_title("Confidence per Cell Type", fontsize=TITLE_SIZE, fontweight="bold")

    fig.suptitle("CellTypist Cell Type Prediction  ·  scCytoTrek",
                 fontsize=TITLE_SIZE + 1, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches="tight", facecolor=FIG_BG)
        print(f"  Saved: {save}")
    if show:
        plt.show()
    plt.close()
