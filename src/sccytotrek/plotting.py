"""
Enhanced visualizations inspired by SeuratExtend.
"""

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import numpy as np
from typing import Optional, Union, List

# SeuratExtend inspired default palettes
# Combining modern aesthetic palettes from popular toolkits
SEURAT_EXTEND_PALETTES = {
    "default": [
        "#000080", # Navy Blue
        "#FFD700", # Egg Yellow (or Gold)
        "#8B0000", # Dark Red
        "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", 
        "#A65628", "#F781BF", "#999999", "#66C2A5", "#FC8D62", "#8DA0CB", 
        "#E78AC3", "#A6D854", "#B3B3B3"
    ],
    "nature": [
        "#DC0000", "#3C5488", "#00A087", "#4DBBD5", "#E64B35", "#F39B7F", 
        "#8491B4", "#91D1C2", "#7E6148", "#B09C85"
    ]
}

def set_style():
    """
    Apply SeuratExtend-like styles to matplotlib global parameters.
    """
    sns.set_style("ticks")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.frameon"] = False
    
def dim_plot(
    adata: AnnData, 
    color: Union[str, List[str]], 
    basis: str = "umap",
    palette: str = "default",
    title: Optional[str] = None,
    show: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """
    A customized wrapper around sc.pl.embedding to simulate SeuratExtend DimPlot.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    color : str or list of str
        Keys for annotations of observations/cells or variables/genes.
    basis : str, optional
        String indicating the basis to use., by default "umap"
    palette : str, optional
        Color palette to use, by default "default"
    
    Returns
    -------
    Axes or None
        Returns axes, list of axes, or None if `show` is True.
    """
    set_style()
    
    # Select palette
    selected_palette = SEURAT_EXTEND_PALETTES.get(palette, SEURAT_EXTEND_PALETTES["default"])
    
    # Handle single color vs multiple
    if isinstance(color, str):
        colors = [color]
    else:
        colors = color
        
    for c in colors:
        if c in adata.obs.columns and adata.obs[c].dtype.name in ['category', 'object']:
            # Ensure custom palette length matches categories
            n_cats = len(adata.obs[c].value_counts())
            adata.uns[f"{c}_colors"] = selected_palette[:n_cats]
            
    # Draw logic wrapping scanpy
    ax = sc.pl.embedding(
        adata, 
        basis=basis, 
        color=color, 
        frameon=False, 
        title=title if title else color,
        show=show,
        save=save,
        **kwargs
    )
    
    return ax
    
def feature_plot(
    adata: AnnData, 
    features: Union[str, List[str]], 
    basis: str = "umap",
    cmap: str = "viridis",
    show: bool = True,
    save: Optional[str] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """
    A customized wrapper for plotting feature expression (similar to Seurat's FeaturePlot).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    features : str or list of str
        Features to plot.
    """
    set_style()
    
    ax = sc.pl.embedding(
        adata,
        basis=basis,
        color=features,
        color_map=cmap,
        frameon=False,
        show=show,
        save=save,
        **kwargs
    )
    
    return ax
