"""
Lineage and clonal tracking operations based on cell barcodes.
"""

from anndata import AnnData
import networkx as nx
import numpy as np

def build_lineage_graph(adata: AnnData, barcode_key: str = "barcode") -> nx.Graph:
    """
    Link cells with the same ancestor based on a shared barcode.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    barcode_key : str
        Column in `adata.obs` containing the clonal barcode.
        
    Returns
    -------
    nx.Graph
        A NetworkX graph where nodes are cells and edges connect cells sharing a barcode.
    """
    if barcode_key not in adata.obs:
        raise ValueError(f"Barcode key '{barcode_key}' not found in adata.obs.")
        
    print(f"Building lineage graph linking cells by '{barcode_key}'...")
    
    G = nx.Graph()
    G.add_nodes_from(adata.obs_names)
    
    # Group cells by barcode
    barcode_groups = adata.obs.groupby(barcode_key).groups
    
    edges = []
    for barcode, indices in barcode_groups.items():
        # Exclude cells without barcodes or with 'NA'
        if pd.isna(barcode) or str(barcode).strip() == "":
            continue
            
        cells = list(indices)
        if len(cells) > 1:
            # Create a clique among cells sharing the same barcode
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    edges.append((cells[i], cells[j]))
                    
    G.add_edges_from(edges)
    
    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} lineage links.")
    
    # Optionally store the graph or connectivity matrix in adata
    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=adata.obs_names)
    adata.obsp['lineage_connectivities'] = adj_matrix
    
    return G

def impute_lineage_barcodes(adata: AnnData, barcode_key: str = 'barcode', n_neighbors: int = 15) -> AnnData:
    """
    Infer missing lineage barcodes based on RNA transcriptomic kNN proximity.
    Assumes standard scanpy neighbors have been computed if `distances` exist.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    barcode_key : str
        Column in `adata.obs` containing the clonal barcode (some may be missing/NA).
    n_neighbors : int
        Number of neighbors to consider for imputation if building a new graph.
        
    Returns
    -------
    AnnData
        The AnnData object with a new column `{barcode_key}_imputed` containing 
        the fused clonal predictions.
    """
    import pandas as pd
    import scanpy as sc
    
    if barcode_key not in adata.obs:
        raise ValueError(f"Barcode key '{barcode_key}' missing from adata.obs.")
        
    print(f"Imputing missing lineage '{barcode_key}' based on RNA topology...")
    
    original_barcodes = adata.obs[barcode_key].copy()
    
    # Identify which cells are missing barcodes (NaN, None, empty string)
    is_missing = original_barcodes.isna() | (original_barcodes.astype(str).str.strip() == "") | (original_barcodes.astype(str).str.lower() == "nan")
    missing_idx = np.where(is_missing)[0]
    present_idx = np.where(~is_missing)[0]
    
    print(f"Found {len(present_idx)} cells with barcodes and {len(missing_idx)} cells missing barcodes.")
    
    if len(missing_idx) == 0:
        print("No missing barcodes. Skipping imputation.")
        adata.obs[f"{barcode_key}_imputed"] = original_barcodes
        return adata
        
    if len(present_idx) == 0:
        print("No valid barcodes exist to propagate!")
        adata.obs[f"{barcode_key}_imputed"] = original_barcodes
        return adata
        
    # Ensure nearest neighbors is computed
    if 'distances' not in adata.obsp:
        print(f"Computing kNN graph (k={n_neighbors})...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors)
        
    distances = adata.obsp['distances']
    
    # Infer missing labels via nearest known neighbor
    imputed_barcodes = original_barcodes.copy().astype(object)
    
    for i in missing_idx:
        # Get neighbors of this cell
        row = distances[i, :]
        if row.nnz == 0:
            continue
            
        # Extract neighbor indices and distances
        neighbors = row.indices
        
        # Filter to neighbors that HAVE a barcode
        labeled_neighbors = [n for n in neighbors if n in present_idx]
        
        if labeled_neighbors:
            # Simple majority voting among labeled neighbors
            neighbor_labels = original_barcodes.iloc[labeled_neighbors]
            majority_label = neighbor_labels.value_counts().index[0]
            imputed_barcodes.iloc[i] = majority_label
            
    adata.obs[f"{barcode_key}_imputed"] = imputed_barcodes
    
    imputed_count = len(missing_idx) - imputed_barcodes.isna().sum()
    print(f"Successfully imputed barcodes for {imputed_count} cells.")
    
    return adata
