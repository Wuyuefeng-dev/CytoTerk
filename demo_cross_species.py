"""
Demo: Cross-Species Alignment (Human ↔ Mouse ↔ Rat)
====================================================
Demonstrates:
  1. Mock ortholog table (Human/Mouse/Rat gene conventions)
  2. Venn diagram showing shared vs species-unique genes (5-panel figure)
  3. Joint UMAP after CCA cross-species alignment (3-panel figure)

Outputs:
  demo_figs/cross_species_venn.png
  demo_figs/cross_species_umap.png
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from sccytotrek.integration.species import (
    make_ortholog_table,
    convert_orthologs,
    analyze_cross_species_overlap,
    plot_cross_species_venn,
    run_cross_species_alignment,
)


def make_species_adata(gene_universe, n_cells, n_clusters, species_name, random_state):
    """Simulate scRNA-seq data for one species given a gene universe list."""
    rng = np.random.default_rng(random_state)
    n_genes = len(gene_universe)
    cluster_ids = np.repeat(np.arange(n_clusters), n_cells // n_clusters)
    cluster_ids = np.append(cluster_ids, np.zeros(n_cells - len(cluster_ids), dtype=int))
    rng.shuffle(cluster_ids)

    X = np.zeros((n_cells, n_genes))
    for c in range(n_clusters):
        mask = cluster_ids == c
        center = rng.uniform(0, 3, n_genes)
        center[c * (n_genes // n_clusters):(c + 1) * (n_genes // n_clusters)] += 5
        X[mask] = rng.negative_binomial(1, 0.4, (mask.sum(), n_genes)) + center
    X = np.clip(X, 0, None).astype(np.float32)

    obs = pd.DataFrame({"cluster": cluster_ids.astype(str),
                        "species": species_name},
                       index=[f"{species_name}_cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_universe)
    return ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)


def main():
    os.makedirs("demo_figs", exist_ok=True)

    # ── 1. Generate ortholog table ─────────────────────────────────────────────
    print("Generating mock ortholog table (Human / Mouse / Rat)...")
    ortho_table = make_ortholog_table(
        n_orthologs=1200,
        n_human_only=250,
        n_mouse_only=200,
        n_rat_only=150,
        include_rat=True,
        random_state=42
    )
    print(f"  Ortholog table: {ortho_table.shape[0]} rows, "
          f"columns: {list(ortho_table.columns)}")

    # ── 2. Derive per-species gene universes directly from ortholog table ────
    # Human genes (reference): use ortholog GENE* names (UPPERCASE)
    human_shared = sorted(set(ortho_table['human'].dropna().tolist()))
    mouse_shared = sorted(set(ortho_table['mouse'].dropna().tolist()))
    rat_shared   = sorted(set(ortho_table['rat'].dropna().tolist()))

    # Add species-unique (not in ortholog table) genes for realism
    human_extra = [f"HEXTRA{i:04d}" for i in range(100)]
    mouse_extra = [f"Mextra{i:04d}" for i in range(80)]
    rat_extra   = [f"Rextra{i:04d}" for i in range(60)]
    human_genes = human_shared + human_extra
    mouse_genes = mouse_shared + mouse_extra
    rat_genes   = rat_shared   + rat_extra

    # ── 3. Gene overlap analysis ──────────────────────────────────────────────
    print("\nAnalyzing cross-species gene overlaps...")
    gene_sets = {
        "Human": set(human_genes),
        "Mouse": set(mouse_genes),
        "Rat":   set(rat_genes),
    }
    overlap_result = analyze_cross_species_overlap(gene_sets)

    for sp, n in overlap_result["sizes"].items():
        print(f"  {sp}: {n} genes")
    for (A, B), pw in overlap_result["pairwise"].items():
        print(f"  {A} ∩ {B}: {pw['n_shared']} shared  (Jaccard={pw['jaccard']:.3f})")
    print(f"  Shared across all 3: {len(overlap_result['shared_all'])}")

    # ── 4. Venn diagram ────────────────────────────────────────────────────────
    print("\nPlotting cross-species Venn diagram...")
    plot_cross_species_venn(
        gene_sets,
        overlap_result=overlap_result,
        ortholog_table=ortho_table,
        save="demo_figs/cross_species_venn.png",
        show=False
    )

    # ── 5. Simulate scRNA-seq for each species ────────────────────────────────
    print("\nSimulating species-specific scRNA-seq datasets...")
    adata_human = make_species_adata(human_genes, n_cells=600, n_clusters=5,
                                     species_name="Human", random_state=10)
    adata_mouse = make_species_adata(mouse_genes, n_cells=500, n_clusters=4,
                                     species_name="Mouse", random_state=20)
    adata_rat   = make_species_adata(rat_genes,   n_cells=400, n_clusters=4,
                                     species_name="Rat",   random_state=30)
    print(f"  Human: {adata_human.shape}")
    print(f"  Mouse: {adata_mouse.shape}")
    print(f"  Rat:   {adata_rat.shape}")

    # ── 6. CCA alignment + joint UMAP ─────────────────────────────────────────
    print("\nRunning cross-species CCA alignment and joint UMAP...")
    adatas = {"Human": adata_human, "Mouse": adata_mouse, "Rat": adata_rat}
    adata_joint = run_cross_species_alignment(
        adatas,
        ortholog_table=ortho_table,
        reference_species="human",
        n_pcs=25,
        n_cca=15,
        save="demo_figs/cross_species_umap.png",
        show=False
    )
    print(f"\n  Joint AnnData: {adata_joint.shape}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\nOutputs:")
    for f in ["demo_figs/cross_species_venn.png", "demo_figs/cross_species_umap.png"]:
        ok = "✓" if os.path.exists(f) else "✗"
        print(f"  {ok} {f}")


if __name__ == "__main__":
    main()
