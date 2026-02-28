from .base import dim_plot, feature_plot
from .volcano import plot_volcano

from .style import (
    apply_seurat_theme,
    seurat_figure,
    discrete_colors,
    SEURAT_DISCRETE,
    SEURAT_FEATURE_CMAP,
    SEURAT_EXPR_CMAP,
    SEURAT_CORR_CMAP,
    SEURAT_ENTROPY_CMAP,
    SEURAT_DOTPLOT_CMAP,
)

__all__ = [
    "dim_plot",
    "feature_plot",
    "plot_volcano",
    "apply_seurat_theme",
    "seurat_figure",
    "discrete_colors",
    "SEURAT_DISCRETE",
    "SEURAT_FEATURE_CMAP",
    "SEURAT_EXPR_CMAP",
    "SEURAT_CORR_CMAP",
    "SEURAT_ENTROPY_CMAP",
    "SEURAT_DOTPLOT_CMAP",
]
