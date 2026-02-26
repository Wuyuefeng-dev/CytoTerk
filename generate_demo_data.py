import os
import scanpy as sc
from sccytotrek.datasets import make_mock_scrna, make_mock_multiome
import mudata as md

def main():
    print("Generating scCytoTrek Demo Datasets...")
    os.makedirs("demo_data", exist_ok=True)
    
    # 1. Generate standard scRNA-seq demo (2000 cells, 2500 genes)
    print("Generating scRNA-seq data: 2000 cells x 2500 genes")
    adata = make_mock_scrna(n_cells=2000, n_genes=2500)
    adata.write_h5ad("demo_data/sccytotrek_demo_scrna.h5ad")
    print("Saved -> demo_data/sccytotrek_demo_scrna.h5ad")
    
    # 2. Generate Multiome demo (2000 cells, 2500 genes, 2000 peaks)
    print("Generating scMultiome data: 2000 cells x 2500 genes x 2000 peaks")
    mdata = make_mock_multiome(n_cells=2000, n_genes=2500, n_peaks=2000)
    mdata.write("demo_data/sccytotrek_demo_multiome.h5mu")
    print("Saved -> demo_data/sccytotrek_demo_multiome.h5mu")

if __name__ == "__main__":
    main()
