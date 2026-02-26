"""
Multi-Omics Integration Benchmark Demo
=======================================
Tests 5 integration methods on 3 paired multi-omics datasets:
  1. RNA + ATAC (scMultiome)
  2. RNA + DNA Methylation
  3. RNA + Surface Protein (CITE-seq)

Outputs:
  demo_figs/multiome_rna_atac_integration.png
  demo_figs/multiome_rna_methylation_integration.png
  demo_figs/multiome_rna_protein_integration.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import scanpy as sc
sc.settings.n_jobs = 1
import numpy as np

from sccytotrek.datasets.multiome_mock import (
    make_mock_multiome_rna_atac,
    make_mock_multiome_rna_methylation,
    make_mock_multiome_rna_protein,
)
from sccytotrek.multiome.integration_v2 import (
    run_integration_benchmark,
    plot_integration_results,
)


def run_one(adata_rna, adata_mod2, mod2_type, title, outfile):
    print(f"\n{'='*60}")
    print(f"Dataset: {title}")
    print(f"  RNA: {adata_rna.shape}, {mod2_type}: {adata_mod2.shape}")
    results = run_integration_benchmark(
        adata_rna, adata_mod2,
        mod2_type=mod2_type,
        batch_key="batch",
        cluster_key="cluster",
    )
    print(f"\n  Integration Quality Summary for {title}:")
    for method in ["WNN", "CCA", "ConcatPCA", "Procrustes", "SNF"]:
        m = results[method].get("metrics", {})
        print(f"    {method:12s} | Silhouette={m.get('silhouette_cluster','n/a'):>8} "
              f"| LISI={m.get('batch_lisi','n/a'):>8}")
    plot_integration_results(results, title_prefix=title, save=outfile, show=False)
    return results


def main():
    os.makedirs("demo_figs", exist_ok=True)

    # Dataset 1: RNA + ATAC
    print("Generating RNA + ATAC dataset...")
    rna, atac = make_mock_multiome_rna_atac(n_cells=800, n_genes=1500, n_peaks=3000,
                                             n_clusters=5, n_batches=2, random_state=42)
    run_one(rna, atac, "atac", "RNA + ATAC",
            "demo_figs/multiome_rna_atac_integration.png")

    # Dataset 2: RNA + Methylation
    print("\nGenerating RNA + Methylation dataset...")
    rna2, methyl = make_mock_multiome_rna_methylation(n_cells=800, n_genes=1500,
                                                       n_cpg_sites=2000, n_clusters=5,
                                                       n_batches=2, random_state=43)
    run_one(rna2, methyl, "methylation", "RNA + Methylation",
            "demo_figs/multiome_rna_methylation_integration.png")

    # Dataset 3: RNA + Protein
    print("\nGenerating RNA + Surface Protein dataset...")
    rna3, protein = make_mock_multiome_rna_protein(n_cells=800, n_genes=1500,
                                                    n_proteins=80, n_clusters=5,
                                                    n_batches=2, random_state=44)
    run_one(rna3, protein, "protein", "RNA + Protein (CITE-seq)",
            "demo_figs/multiome_rna_protein_integration.png")

    print("\nAll done! Figures saved:")
    for f in ["demo_figs/multiome_rna_atac_integration.png",
              "demo_figs/multiome_rna_methylation_integration.png",
              "demo_figs/multiome_rna_protein_integration.png"]:
        print(f"  {'✓' if os.path.exists(f) else '✗'} {f}")


if __name__ == "__main__":
    main()
