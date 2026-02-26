"""
Mock generators for three multi-omics dataset types:
  1. RNA + ATAC (scMultiome)
  2. RNA + DNA Methylation  
  3. RNA + Surface Protein (CITE-seq style)

All datasets include batch effects to make integration non-trivial.
"""

import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp


def _make_base_rna(n_cells, n_genes, n_clusters, random_state):
    """Shared RNA expression generator used by all three functions."""
    rng = np.random.default_rng(random_state)
    cluster_ids = np.repeat(np.arange(n_clusters), n_cells // n_clusters)
    cluster_ids = np.append(cluster_ids, np.zeros(n_cells - len(cluster_ids), dtype=int))
    rng.shuffle(cluster_ids)

    X_rna = np.zeros((n_cells, n_genes))
    for c in range(n_clusters):
        mask = cluster_ids == c
        center = rng.uniform(0, 5, n_genes)
        center[c * (n_genes // n_clusters):(c + 1) * (n_genes // n_clusters)] += 6
        X_rna[mask] = rng.negative_binomial(1, 0.3, (mask.sum(), n_genes)) + center
    X_rna = np.clip(X_rna, 0, None)
    return X_rna, cluster_ids


def make_mock_multiome_rna_atac(
    n_cells: int = 1000,
    n_genes: int = 2000,
    n_peaks: int = 5000,
    n_clusters: int = 5,
    n_batches: int = 2,
    random_state: int = 42,
):
    """
    Generate a paired RNA + ATAC scMultiome dataset with batch effects.

    Returns
    -------
    adata_rna : AnnData — normalized log1p RNA expression
    adata_atac : AnnData — binary ATAC peak accessibility
    """
    X_rna, cluster_ids = _make_base_rna(n_cells, n_genes, n_clusters, random_state)
    rng = np.random.default_rng(random_state + 1)

    # Batch effects
    batch = np.repeat(np.arange(n_batches), n_cells // n_batches)
    batch = np.append(batch, np.zeros(n_cells - len(batch), dtype=int))

    for b in range(n_batches):
        mask = batch == b
        X_rna[mask] += rng.normal(b * 3, 0.5, (mask.sum(), n_genes))

    # ATAC: sparse binary matrix correlated with RNA clusters
    X_atac = np.zeros((n_cells, n_peaks))
    for c in range(n_clusters):
        mask = cluster_ids == c
        peak_probs = np.ones(n_peaks) * 0.05
        peak_probs[c * (n_peaks // n_clusters):(c + 1) * (n_peaks // n_clusters)] = 0.6
        X_atac[mask] = rng.binomial(1, peak_probs, (mask.sum(), n_peaks))
    # Add batch effect to ATAC
    for b in range(n_batches):
        mask = batch == b
        X_atac[mask] += rng.binomial(1, 0.03 * b, (mask.sum(), n_peaks))
    X_atac = np.clip(X_atac, 0, 1)

    obs = pd.DataFrame({
        "cluster": cluster_ids.astype(str),
        "batch": batch.astype(str),
    }, index=[f"cell_{i}" for i in range(n_cells)])

    adata_rna = ad.AnnData(
        X=sp.csr_matrix(X_rna.astype(np.float32)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    )
    adata_atac = ad.AnnData(
        X=sp.csr_matrix(X_atac.astype(np.float32)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"peak_{i}" for i in range(n_peaks)])
    )
    return adata_rna, adata_atac


def make_mock_multiome_rna_methylation(
    n_cells: int = 1000,
    n_genes: int = 2000,
    n_cpg_sites: int = 3000,
    n_clusters: int = 5,
    n_batches: int = 2,
    random_state: int = 42,
):
    """
    Generate RNA + DNA Methylation beta-value dataset with batch effects.

    Returns
    -------
    adata_rna : AnnData — RNA expression
    adata_methyl : AnnData — CpG methylation beta values [0, 1]
    """
    X_rna, cluster_ids = _make_base_rna(n_cells, n_genes, n_clusters, random_state)
    rng = np.random.default_rng(random_state + 2)

    batch = np.repeat(np.arange(n_batches), n_cells // n_batches)
    batch = np.append(batch, np.zeros(n_cells - len(batch), dtype=int))
    for b in range(n_batches):
        mask = batch == b
        X_rna[mask] += rng.normal(b * 2.5, 0.4, (mask.sum(), n_genes))

    # Methylation: inversely correlated with expression (active genes = low methylation)
    X_methyl = np.zeros((n_cells, n_cpg_sites))
    for c in range(n_clusters):
        mask = cluster_ids == c
        base_meth = rng.beta(5, 1, n_cpg_sites)   # high methylation background
        base_meth[c * (n_cpg_sites // n_clusters):(c + 1) * (n_cpg_sites // n_clusters)] = \
            rng.beta(1, 5, n_cpg_sites // n_clusters)  # low methylation at active loci
        noise = rng.normal(0, 0.05, (mask.sum(), n_cpg_sites))
        X_methyl[mask] = np.clip(base_meth + noise, 0, 1)
    for b in range(n_batches):
        mask = batch == b
        X_methyl[mask] += rng.normal(b * 0.05, 0.02, (mask.sum(), n_cpg_sites))
    X_methyl = np.clip(X_methyl, 0, 1)

    obs = pd.DataFrame({
        "cluster": cluster_ids.astype(str),
        "batch": batch.astype(str),
    }, index=[f"cell_{i}" for i in range(n_cells)])

    adata_rna = ad.AnnData(
        X=sp.csr_matrix(X_rna.astype(np.float32)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    )
    adata_methyl = ad.AnnData(
        X=X_methyl.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"CpG_{i}" for i in range(n_cpg_sites)])
    )
    return adata_rna, adata_methyl


def make_mock_multiome_rna_protein(
    n_cells: int = 1000,
    n_genes: int = 2000,
    n_proteins: int = 100,
    n_clusters: int = 5,
    n_batches: int = 2,
    random_state: int = 42,
):
    """
    Generate RNA + Surface Protein (CITE-seq style) dataset with batch effects.

    Returns
    -------
    adata_rna : AnnData — RNA expression
    adata_protein : AnnData — antibody-derived tag (ADT) counts
    """
    X_rna, cluster_ids = _make_base_rna(n_cells, n_genes, n_clusters, random_state)
    rng = np.random.default_rng(random_state + 3)

    batch = np.repeat(np.arange(n_batches), n_cells // n_batches)
    batch = np.append(batch, np.zeros(n_cells - len(batch), dtype=int))
    for b in range(n_batches):
        mask = batch == b
        X_rna[mask] += rng.normal(b * 2, 0.4, (mask.sum(), n_genes))

    # Protein: sparse, cluster-specific surface markers
    X_protein = np.zeros((n_cells, n_proteins))
    proteins_per_cluster = max(1, n_proteins // n_clusters)
    for c in range(n_clusters):
        mask = cluster_ids == c
        high_idx = np.arange(c * proteins_per_cluster,
                             min((c + 1) * proteins_per_cluster, n_proteins))
        X_protein[mask] = rng.negative_binomial(1, 0.5, (mask.sum(), n_proteins)) * 10
        X_protein[np.ix_(np.where(mask)[0], high_idx)] += \
            rng.negative_binomial(5, 0.3, (mask.sum(), len(high_idx))) * 50
    for b in range(n_batches):
        mask = batch == b
        X_protein[mask] *= (1 + b * 0.3)   # multiplicative batch effect
    X_protein = np.clip(X_protein, 0, None)

    obs = pd.DataFrame({
        "cluster": cluster_ids.astype(str),
        "batch": batch.astype(str),
    }, index=[f"cell_{i}" for i in range(n_cells)])

    adata_rna = ad.AnnData(
        X=sp.csr_matrix(X_rna.astype(np.float32)),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    )
    adata_protein = ad.AnnData(
        X=X_protein.astype(np.float32),
        obs=obs.copy(),
        var=pd.DataFrame(index=[f"protein_{i}" for i in range(n_proteins)])
    )
    return adata_rna, adata_protein
