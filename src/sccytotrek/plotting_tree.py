"""
Visualization of hierarchical lineage trees.
"""

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Optional

def plot_hierarchical_tree(
    G: nx.DiGraph, 
    title: str = "Hierarchical Lineage Tree",
    figsize: tuple = (12, 8),
    node_size: int = 50,
    font_size: int = 8,
    with_labels: bool = False,
    show: bool = True
) -> plt.Axes:
    """
    Plot a directed hierarchical tree graph (e.g., from Polylox or DARLIN).
    
    Parameters
    ----------
    G : nx.DiGraph
        Directed graph representing the lineage tree.
    title : str
        Plot title.
    figsize : tuple
        Figsize for matplotlib.
    node_size : int
        Size of the nodes.
    font_size : int
        Font size for labels (if with_labels=True).
    with_labels : bool
        Whether to draw node labels. For many cells, False is recommended.
    show : bool
        Whether to show the plot immediately.
        
    Returns
    -------
    plt.Axes
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    
    # Check if we have pygraphviz or pydot installed for true hierarchical layouts.
    # We will use Kamada-Kawai or Spring layout as fallback if standard tree layouts aren't available.
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog='dot')
    except ImportError:
        # Fallback to standard networkx hierarchical-like layout (shell or spring)
        # For a true tree with a single root, we can mock a hierarchical layout
        print("Note: PyGraphviz not found. Using spring_layout. For true hierarchical trees, 'pip install pygraphviz'.")
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        
    # Draw logic
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, node_color='skyblue', alpha=0.8, edgecolors='black')
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5, arrows=True, arrowsize=10, node_size=node_size)
    
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
        
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    if show:
        plt.show()
        
    return ax
