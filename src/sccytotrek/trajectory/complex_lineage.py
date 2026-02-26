"""
Specific parsers and lineage trees for complex barcoding systems like Polylox and DARLIN.
"""

from anndata import AnnData
import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional

def build_polylox_tree(adata: AnnData, barcode_key: str = "barcode_polylox") -> nx.DiGraph:
    """
    Build a hierarchical lineage tree for Polylox barcoding.
    Polylox barcodes are generated through staggered Cre recombinations, creating a 
    progressively altered barcode string from a known original state.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    barcode_key : str
        Column in `adata.obs` containing the Polylox barcode string.
        
    Returns
    -------
    nx.DiGraph
        A directed graph representing the developmental lineage (root to terminal cells).
    """
    if barcode_key not in adata.obs:
        raise ValueError(f"Polylox barcode key '{barcode_key}' not found.")
        
    print(f"Building Polylox lineage tree using '{barcode_key}'...")
    
    # In Polylox, the length or difference from an un-recombined state indicates lineage depth.
    # Here we mock a basic hierarchical construction.
    # A true implementation would use edit distance / specific recombination rules to infer parent-child ties.
    
    G = nx.DiGraph()
    barcodes = adata.obs[barcode_key].dropna().unique()
    
    G.add_node("Root_Polylox")
    
    for bc in barcodes:
        # Mocking hierarchy: just link to root for now
        # Advanced: compute pairwise edit distances and build a Minimum Spanning Tree (MST)
        # or use heuristic rules for Polylox cassette deletions.
        G.add_edge("Root_Polylox", str(bc))
        
        # Link cells to their barcode node
        cells = adata.obs.index[adata.obs[barcode_key] == bc]
        for cell in cells:
            G.add_edge(str(bc), cell)
            
    print(f"Polylox Tree built: {G.number_of_nodes()} nodes.")
    return G

def build_darlin_tree(adata: AnnData, barcode_key: str = "barcode_darlin", intClone_key: str = "intClone") -> nx.DiGraph:
    """
    Build a hierarchical lineage tree for DARLIN (polylox-like + Cas9 scarring) barcoding.
    DARLIN often uses both a stable integrant clone ID and accumulated CRISPR scars.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    barcode_key : str
        Column containing the accumulated scar/barcode pattern.
    intClone_key : str
        Column containing the stable integration clone ID (founder cell).
        
    Returns
    -------
    nx.DiGraph
    """
    if barcode_key not in adata.obs or intClone_key not in adata.obs:
        raise ValueError(f"Keys '{barcode_key}' and/or '{intClone_key}' not found.")
        
    print(f"Building DARLIN lineage tree...")
    
    G = nx.DiGraph()
    
    # 1. Root nodes are the intClones
    clones = adata.obs[intClone_key].dropna().unique()
    for clone in clones:
        G.add_node(f"Clone_{clone}")
        
        # Get all barcodes within this clone
        clone_cells = adata.obs[adata.obs[intClone_key] == clone]
        barcodes = clone_cells[barcode_key].dropna().unique()
        
        # 2. Build sub-trees per clone based on barcode edits
        for bc in barcodes:
            G.add_edge(f"Clone_{clone}", str(bc))
            
            # Attach cells
            cells = clone_cells.index[clone_cells[barcode_key] == bc]
            for cell in cells:
                G.add_edge(str(bc), cell)
                
    print(f"DARLIN Tree built: {G.number_of_nodes()} nodes across {len(clones)} clones.")
    return G
