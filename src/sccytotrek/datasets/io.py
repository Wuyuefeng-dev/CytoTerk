"""
Data input/output functions for handling CellRanger, Seurat, and Scanpy formatted data.
"""

import scanpy as sc
from anndata import AnnData
import os

def read_10x(path: str, is_h5: bool = False) -> AnnData:
    """
    Read output from 10x Genomics CellRanger.
    
    Parameters
    ----------
    path : str
        Path to the 10x output directory (containing matrix.mtx, features.tsv, barcodes.tsv)
        or path to an hdf5 file (filtered_feature_bc_matrix.h5).
    is_h5 : bool
        If True, reads an h5 file instead of an mtx directory.
    """
    print(f"Reading 10x CellRanger data from: {path}")
    if is_h5:
        adata = sc.read_10x_h5(path)
    else:
        adata = sc.read_10x_mtx(path, var_names='gene_symbols', cache=True)
    adata.var_names_make_unique()
    return adata

def read_h5ad(path: str) -> AnnData:
    """
    Read standard Scanpy h5ad formatted data.
    """
    print(f"Reading Scanpy/AnnData file: {path}")
    return sc.read_h5ad(path)

def read_seurat(path: str) -> AnnData:
    """
    Read a Seurat object.
    Currently, this requires the Seurat object to be saved as an .h5Seurat file 
    using SeuratDisk in R, or a loom file.
    """
    import warnings
    print(f"Reading Seurat data from: {path}")
    
    if path.endswith(".h5seurat") or path.endswith(".h5Seurat"):
        raise NotImplementedError(
            "Direct reading of .h5Seurat requires specific python wrappers or conversion via tools like sceasy. "
            "Please export your Seurat object to .h5ad using SeuratDisk in R: "
            "SaveH5Seurat(seurat_obj, filename='obj.h5Seurat'); Convert('obj.h5Seurat', dest='h5ad')"
        )
    elif path.endswith(".rds"):
        raise ValueError("Cannot read .rds files directly in Python. Please export from Seurat to .h5ad format first.")
        
    warnings.warn("Assuming generic format. For best results, save Seurat objects as .h5ad in R.")
    return sc.read(path)
