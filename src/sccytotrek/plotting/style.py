# sccytotrek plotting style module — SeuratExtend-inspired

# ── Discrete palette (Nature / SeuratExtend signature colours) ────────────────
# Matches the NPG (Nature Publishing Group) palette used by SeuratExtend
SEURAT_DISCRETE = [
    "#4DBBD5",   # teal
    "#E64B35",   # red-orange
    "#00A087",   # green-teal
    "#3C5488",   # navy blue
    "#F39B7F",   # salmon
    "#8491B4",   # slate-lavender
    "#91D1C2",   # mint
    "#DC0000",   # crimson
    "#7E6148",   # umber
    "#B09C85",   # warm beige
    "#FFDC91",   # straw
    "#A9D18E",   # moss green
]

# ── Continuous / feature expression palettes ─────────────────────────────────
SEURAT_FEATURE_CMAP   = "RdYlBu_r"   # heatmap / scalar features
SEURAT_EXPR_CMAP      = "YlOrRd"     # UMP feature expression
SEURAT_CORR_CMAP      = "RdBu_r"     # correlation heatmaps (-1..+1)
SEURAT_ENTROPY_CMAP   = "magma"      # entropy / enrichment
SEURAT_DOTPLOT_CMAP   = "Blues"      # dot plot expression colour

# ── Typography ────────────────────────────────────────────────────────────────
FONT_FAMILY   = "sans-serif"
TITLE_SIZE    = 13
SUBTITLE_SIZE = 10
LABEL_SIZE    = 9
TICK_SIZE     = 8

# ── Layout ────────────────────────────────────────────────────────────────────
FIG_BG        = "white"
AX_BG         = "white"
SPINE_COLOR   = "#CCCCCC"
GRID_COLOR    = "#EEEEEE"
GRID_ALPHA    = 0.7
POINT_SIZE    = 6.0           # default scatter point size (SC cells)
BULK_MARKER   = "*"
BULK_SIZE     = 280
LINE_WIDTH    = 0.5

def apply_seurat_theme(ax, grid=False, spines="bl"):
    """
    Apply SeuratExtend-style aesthetics to a matplotlib Axes.

    Parameters
    ----------
    ax     : matplotlib.axes.Axes
    grid   : bool — whether to show a subtle background grid
    spines : str  — which spines to keep: 'bl' = bottom+left (classic),
                    'none' = all hidden, 'all' = keep all four
    """
    ax.set_facecolor(AX_BG)
    ax.patch.set_alpha(1.0)

    # Spine handling
    if spines == "none":
        for s in ax.spines.values():
            s.set_visible(False)
    elif spines == "bl":
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for s in ["bottom", "left"]:
            ax.spines[s].set_color(SPINE_COLOR)
            ax.spines[s].set_linewidth(0.8)
    else:
        for s in ax.spines.values():
            s.set_color(SPINE_COLOR)
            s.set_linewidth(0.8)

    # Grid
    if grid:
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA,
                linewidth=0.5, linestyle="-")
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    # Tick params
    ax.tick_params(axis="both", labelsize=TICK_SIZE,
                   length=3, width=0.6, color=SPINE_COLOR)

    return ax


def seurat_figure(nrows=1, ncols=1, figsize=None, title=None):
    """
    Create a figure pre-configured with SeuratExtend aesthetics.
    Returns (fig, axes).  axes is a flat numpy array if multiple subplots.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family":      FONT_FAMILY,
        "font.size":        LABEL_SIZE,
        "axes.titlesize":   TITLE_SIZE,
        "axes.labelsize":   LABEL_SIZE,
        "xtick.labelsize":  TICK_SIZE,
        "ytick.labelsize":  TICK_SIZE,
        "axes.facecolor":   AX_BG,
        "figure.facecolor": FIG_BG,
        "axes.linewidth":   0.8,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "legend.frameon":   False,
        "legend.fontsize":  TICK_SIZE,
    })

    if figsize is None:
        figsize = (7 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=FIG_BG, squeeze=False)
    for ax in axes.ravel():
        apply_seurat_theme(ax)

    if title:
        fig.suptitle(title, fontsize=TITLE_SIZE + 1,
                     fontweight="bold", y=1.02)

    if nrows == 1 and ncols == 1:
        return fig, axes[0, 0]
    return fig, axes


def discrete_colors(n, palette=None):
    """Return a list of n SeuratExtend discrete colors."""
    import matplotlib.pyplot as plt
    if palette is not None:
        return plt.cm.get_cmap(palette, n)
    colors = SEURAT_DISCRETE * ((n // len(SEURAT_DISCRETE)) + 1)
    return colors[:n]
