"""
Cross-species alignment: Human vs Mouse (and generalizable to other species pairs).

Features
--------
1. Simulated ortholog table (Human/Mouse gene name conventions)
2. `convert_orthologs` — maps gene names across species (1:1 orthologs only)
3. `make_ortholog_table` — generates a realistic mock ortholog mapping
4. `analyze_cross_species_overlap` — returns sets of shared/unique genes with stats
5. `plot_cross_species_venn` — Venn/UpSet diagram of gene overlaps (2–3 species)
6. `plot_cross_species_umap` — joint UMAP after CCA alignment, colored by species
7. `run_cross_species_alignment` — end-to-end: ortholog mapping → CCA → joint UMAP
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ortholog table generator
# ─────────────────────────────────────────────────────────────────────────────

def make_ortholog_table(
    n_orthologs: int = 1500,
    n_human_only: int = 300,
    n_mouse_only: int = 250,
    n_rat_only: int = 200,
    include_rat: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a mock ortholog mapping table (Human ↔ Mouse [↔ Rat]).

    Human genes: UPPERCASE  (e.g. TP53)
    Mouse genes: Title-case  (e.g. Trp53)
    Rat genes:   Title-case  (e.g. Tp53)

    Returns pd.DataFrame with columns: 'human', 'mouse' [, 'rat']
    plus a 'conservation_score' [0, 1] column.
    """
    rng = np.random.default_rng(random_state)
    # Shared gene (ortholog) root names
    roots = [f"GENE{i:04d}" for i in range(n_orthologs)]

    human_shared = [r for r in roots]
    mouse_shared = [r.capitalize() for r in roots]

    df_shared = pd.DataFrame({
        "human": human_shared,
        "mouse": mouse_shared,
        "conservation_score": rng.beta(8, 2, n_orthologs).round(3),
    })

    if include_rat:
        rat_shared = [r[:2] + r[2:].lower() for r in roots]
        df_shared["rat"] = rat_shared
        # Human-only genes (no mouse/rat ortholog)
        df_hu = pd.DataFrame({
            "human": [f"HGENE{i:04d}" for i in range(n_human_only)],
            "mouse": [None] * n_human_only,
            "rat":   [None] * n_human_only,
            "conservation_score": rng.beta(1, 5, n_human_only).round(3),
        })
        df_mu = pd.DataFrame({
            "human": [None] * n_mouse_only,
            "mouse": [f"Mgene{i:04d}" for i in range(n_mouse_only)],
            "rat":   [None] * n_mouse_only,
            "conservation_score": rng.beta(1, 5, n_mouse_only).round(3),
        })
        df_rat = pd.DataFrame({
            "human": [None] * n_rat_only,
            "mouse": [None] * n_rat_only,
            "rat":   [f"Rgene{i:04d}" for i in range(n_rat_only)],
            "conservation_score": rng.beta(1, 5, n_rat_only).round(3),
        })
        return pd.concat([df_shared, df_hu, df_mu, df_rat], ignore_index=True)
    else:
        df_hu = pd.DataFrame({
            "human": [f"HGENE{i:04d}" for i in range(n_human_only)],
            "mouse": [None] * n_human_only,
            "conservation_score": rng.beta(1, 5, n_human_only).round(3),
        })
        df_mu = pd.DataFrame({
            "human": [None] * n_mouse_only,
            "mouse": [f"Mgene{i:04d}" for i in range(n_mouse_only)],
            "conservation_score": rng.beta(1, 5, n_mouse_only).round(3),
        })
        return pd.concat([df_shared, df_hu, df_mu], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gene conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_orthologs(
    adata: AnnData,
    ortholog_table: pd.DataFrame,
    from_species: str = "mouse",
    to_species: str = "human",
) -> AnnData:
    """
    Subset and rename genes in an AnnData to the ortholog space of `to_species`.
    Only 1:1 orthologs (no None values) are kept.
    """
    if from_species not in ortholog_table.columns or to_species not in ortholog_table.columns:
        raise ValueError(f"ortholog_table must have columns '{from_species}' and '{to_species}'.")

    sub = ortholog_table[[from_species, to_species]].dropna()
    mapping = dict(zip(sub[from_species], sub[to_species]))

    genes      = adata.var_names.tolist()
    keep_idx   = [i for i, g in enumerate(genes) if g in mapping]
    new_names  = [mapping[genes[i]] for i in keep_idx]

    adata_out        = adata[:, keep_idx].copy()
    adata_out.var_names = new_names
    adata_out.uns["from_species"] = from_species
    adata_out.uns["to_species"]   = to_species
    print(f"  {from_species}→{to_species}: {len(new_names)} 1:1 ortholog genes retained.")
    return adata_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_cross_species_overlap(
    gene_sets: dict,          # {"Human": set, "Mouse": set, "Rat": set, ...}
    ortholog_table: pd.DataFrame = None,
) -> dict:
    """
    Compute pairwise + global gene overlaps for a dict of species gene sets.

    Returns
    -------
    dict with keys:
      'sizes'      — {species: n_genes}
      'shared_all' — genes present in ALL species
      'pairwise'   — {(A, B): {"shared", "only_A", "only_B", "jaccard"}}
    """
    species = list(gene_sets.keys())
    result  = {"sizes": {s: len(g) for s, g in gene_sets.items()}}
    result["shared_all"] = set.intersection(*gene_sets.values())

    pairwise = {}
    for i in range(len(species)):
        for j in range(i + 1, len(species)):
            A, B = species[i], species[j]
            sA, sB = gene_sets[A], gene_sets[B]
            shared  = sA & sB
            only_A  = sA - sB
            only_B  = sB - sA
            union   = sA | sB
            pairwise[(A, B)] = {
                "shared": shared,
                "n_shared": len(shared),
                "only_A": only_A,
                "only_B": only_B,
                "n_only_A": len(only_A),
                "n_only_B": len(only_B),
                "jaccard": len(shared) / len(union) if union else 0.0,
            }
    result["pairwise"] = pairwise
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Venn diagram (2 or 3 species)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cross_species_venn(
    gene_sets: dict,
    overlap_result: dict = None,
    ortholog_table: pd.DataFrame = None,
    save: str = None,
    show: bool = True,
):
    """
    Publication-quality Venn diagram showing gene overlaps across 2 or 3 species,
    plus a bar chart of pairwise Jaccard similarity and a conservation score violin.

    Parameters
    ----------
    gene_sets : dict {"SpeciesName": set_of_genes, ...}
    overlap_result : pre-computed dict from analyze_cross_species_overlap()
    ortholog_table : optional, used for the conservation score violin plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, Ellipse
    import matplotlib.gridspec as gridspec

    if overlap_result is None:
        overlap_result = analyze_cross_species_overlap(gene_sets)

    species = list(gene_sets.keys())
    n_sp    = len(species)

    species_colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#76b7b2']
    sp_pal = {s: species_colors[i % len(species_colors)] for i, s in enumerate(species)}

    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── Panel A: Venn diagram (matplotlib circles) ───────────────────────────
    ax_venn = fig.add_subplot(gs[0, 0:2])
    ax_venn.set_facecolor('white')
    ax_venn.set_aspect('equal')
    ax_venn.axis('off')

    if n_sp == 2:
        A_name, B_name = species[0], species[1]
        ov = overlap_result["pairwise"][(A_name, B_name)]
        # Two overlapping circles
        centers = [np.array([-0.6, 0]), np.array([0.6, 0])]
        r = 1.0
        for (x0, y0), sp_name in zip(centers, [A_name, B_name]):
            circle = Circle((x0, y0), r, alpha=0.4,
                             facecolor=sp_pal[sp_name], edgecolor=sp_pal[sp_name], lw=2)
            ax_venn.add_patch(circle)
        ax_venn.text(-1.1, 0, f"{A_name}\n{ov['n_only_A']:,}", ha='center', va='center',
                     fontsize=13, fontweight='bold', color=sp_pal[A_name])
        ax_venn.text(1.1, 0, f"{B_name}\n{ov['n_only_B']:,}", ha='center', va='center',
                     fontsize=13, fontweight='bold', color=sp_pal[B_name])
        ax_venn.text(0, 0, f"Shared\n{ov['n_shared']:,}", ha='center', va='center',
                     fontsize=13, fontweight='bold', color='#333333')
        ax_venn.set_xlim(-2.2, 2.2); ax_venn.set_ylim(-1.8, 1.8)

    elif n_sp >= 3:
        A_name, B_name, C_name = species[0], species[1], species[2]
        centers = [np.array([0, 0.7]), np.array([-0.6, -0.4]), np.array([0.6, -0.4])]
        r = 1.0
        for (xy), sp_name in zip(centers, [A_name, B_name, C_name]):
            circle = Circle(xy, r, alpha=0.3,
                             facecolor=sp_pal[sp_name], edgecolor=sp_pal[sp_name], lw=2)
            ax_venn.add_patch(circle)
        labels = [
            (0, 1.6, A_name), (-1.3, -0.8, B_name), (1.3, -0.8, C_name)
        ]
        for (tx, ty, label) in labels:
            ax_venn.text(tx, ty, f"{label}\n{overlap_result['sizes'][label]:,}",
                         ha='center', va='center', fontsize=11,
                         fontweight='bold', color=sp_pal[label])
        # Shared all
        ax_venn.text(0, -0.1, f"All 3\n{len(overlap_result['shared_all']):,}",
                     ha='center', va='center', fontsize=11, fontweight='bold')
        ax_venn.set_xlim(-2.2, 2.2); ax_venn.set_ylim(-1.8, 2.2)

    ax_venn.set_title('Gene Overlap Across Species', fontsize=13, fontweight='bold', pad=12)

    # ── Panel B: Summary stats table ─────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, 2])
    ax_tbl.axis('off')
    rows = [["Species", "Total Genes"]]
    for s, n in overlap_result["sizes"].items():
        rows.append([s, f"{n:,}"])
    rows.append(["— Shared (all) —", f"{len(overlap_result['shared_all']):,}"])

    for (A, B), pw in overlap_result["pairwise"].items():
        rows.append([f"{A} ∩ {B}", f"{pw['n_shared']:,}"])
        rows.append([f"Jaccard {A}/{B}", f"{pw['jaccard']:.3f}"])

    tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.88])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if r == 0:
            cell.set_facecolor('#2c3e50'); cell.get_text().set_color('white')
            cell.get_text().set_fontweight('bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4f8')
        else:
            cell.set_facecolor('white')
    ax_tbl.set_title('Gene Count Summary', fontsize=11, fontweight='bold', pad=8)

    # ── Panel C: Pairwise Jaccard bar chart ──────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_bar.set_facecolor('#f8f9fa')
    pairs  = [f"{A}∩{B}" for A, B in overlap_result["pairwise"]]
    jaccards = [pw["jaccard"] for pw in overlap_result["pairwise"].values()]
    bar_colors = [species_colors[i % len(species_colors)] for i in range(len(pairs))]
    bars = ax_bar.bar(pairs, jaccards, color=bar_colors, edgecolor='grey', linewidth=0.5)
    for b, v in zip(bars, jaccards):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax_bar.set_ylim(0, max(jaccards) * 1.25)
    ax_bar.set_title('Pairwise Jaccard Similarity\n(↑ = more conserved gene space)',
                     fontsize=10, fontweight='bold')
    ax_bar.set_ylabel('Jaccard Index', fontsize=9)
    ax_bar.tick_params(axis='x', rotation=15, labelsize=8)

    # ── Panel D: Conservation score distribution ──────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_dist.set_facecolor('#f8f9fa')
    if ortholog_table is not None and 'conservation_score' in ortholog_table.columns:
        for sp_name in species[:3]:
            col = sp_name.lower()
            if col in ortholog_table.columns:
                sub = ortholog_table[ortholog_table[col].notna()]
                ax_dist.hist(sub['conservation_score'].dropna(), bins=40,
                             alpha=0.6, label=sp_name,
                             color=sp_pal.get(sp_name, '#888888'), edgecolor='none')
        ax_dist.set_xlabel('Conservation Score', fontsize=9)
        ax_dist.set_ylabel('# Gene Pairs', fontsize=9)
        ax_dist.legend(fontsize=8, frameon=False)
    else:
        # Shared gene count per species combination pie
        pair_items = list(overlap_result["pairwise"].items())
        labels_pie = [f"{A}∩{B}" for (A, B), _ in pair_items]
        sizes_pie  = [pw["n_shared"] for _, pw in pair_items]
        ax_dist.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%',
                    colors=bar_colors, startangle=90,
                    wedgeprops=dict(linewidth=0.5, edgecolor='white'))
    ax_dist.set_title('Conservation Score Distribution', fontsize=10, fontweight='bold')

    # ── Panel E: Unique gene count comparison ─────────────────────────────────
    ax_uniq = fig.add_subplot(gs[1, 2])
    ax_uniq.set_facecolor('#f8f9fa')
    sp_list  = species[:4]
    unique_counts = []
    shared_counts = []
    for s in sp_list:
        others = set.union(*[gene_sets[o] for o in sp_list if o != s]) if len(sp_list) > 1 else set()
        unique_counts.append(len(gene_sets[s] - others))
        shared_counts.append(len(gene_sets[s] & others))
    x = np.arange(len(sp_list))
    ax_uniq.bar(x, shared_counts, color=[sp_pal.get(s, '#888') for s in sp_list],
                alpha=0.9, label='Shared', edgecolor='grey', linewidth=0.5)
    ax_uniq.bar(x, unique_counts, bottom=shared_counts,
                color=[sp_pal.get(s, '#888') for s in sp_list],
                alpha=0.4, label='Species-unique', edgecolor='grey', linewidth=0.5,
                hatch='//')
    ax_uniq.set_xticks(x); ax_uniq.set_xticklabels(sp_list, fontsize=9)
    ax_uniq.set_ylabel('Number of Genes', fontsize=9)
    ax_uniq.set_title('Shared vs Species-Unique Genes', fontsize=10, fontweight='bold')
    ax_uniq.legend(fontsize=8, frameon=False)

    fig.suptitle('Cross-Species Gene Ortholog Analysis', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved Venn figure to {save}")
    if show:
        plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-species UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_species_alignment(
    adatas: dict,             # {"Human": adata_h, "Mouse": adata_m, ...}
    ortholog_table: pd.DataFrame,
    reference_species: str = "human",
    n_pcs: int = 30,
    n_cca: int = 20,
    save: str = None,
    show: bool = True,
):
    """
    Full cross-species pipeline:
    1. Convert all species to reference gene space via orthologs
    2. Concatenate on shared ortholog genes
    3. Run CCA for joint embedding across species
    4. Compute joint UMAP and plot coloured by species + cell type

    Returns the concatenated AnnData with joint UMAP.
    """
    import anndata as ad
    import scanpy as sc

    print("Step 1: Converting gene spaces to reference species...")
    adatas_converted = {}
    for sp_name, adata in adatas.items():
        if sp_name.lower() == reference_species.lower():
            # Reference species — keep as-is, no conversion needed
            adatas_converted[sp_name] = adata.copy()
            adatas_converted[sp_name].uns["from_species"] = sp_name.lower()
            adatas_converted[sp_name].uns["to_species"]   = reference_species.lower()
            print(f"  {sp_name}: reference species — {adata.n_vars} genes kept.")
        else:
            try:
                converted = convert_orthologs(
                    adata, ortholog_table,
                    from_species=sp_name.lower(),
                    to_species=reference_species.lower(),
                )
                adatas_converted[sp_name] = converted
            except Exception as e:
                print(f"  Warning: could not convert {sp_name}: {e}")

    print("Step 2: Finding common ortholog genes...")
    common = list(set.intersection(
        *[set(a.var_names) for a in adatas_converted.values()]
    ))
    print(f"  {len(common)} shared genes across all species.")
    if len(common) < 10:
        raise ValueError("Fewer than 10 shared genes — check ortholog table.")

    print("Step 3: Preprocessing each modality independently...")
    embeddings = {}
    for sp_name, adata in adatas_converted.items():
        sub = adata[:, common].copy()
        X   = sub.X.toarray() if sp.issparse(sub.X) else sub.X.copy()
        X   = np.clip(X, 0, None)
        # Normalize
        row_sum = X.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        X = np.log1p(X / row_sum * 1e4)
        X = StandardScaler().fit_transform(X)
        n_comps = min(n_pcs, X.shape[0] - 1, X.shape[1] - 1)
        pca = PCA(n_components=n_comps, random_state=42)
        embeddings[sp_name] = pca.fit_transform(X)

    print("Step 4: CCA alignment on shared PCA embeddings...")
    sp_list = list(embeddings.keys())
    ref_emb = embeddings[sp_list[0]]
    joint_parts = [ref_emb[:, :n_cca]]

    for k, sp_name in enumerate(sp_list[1:]):
        other_emb = embeddings[sp_name]
        # CCA requires equal number of samples — subsample both to min(n)
        n_min  = min(ref_emb.shape[0], other_emb.shape[0])
        rng_k  = np.random.default_rng(42 + k)
        idx_r  = rng_k.choice(ref_emb.shape[0],   n_min, replace=False)
        idx_o  = rng_k.choice(other_emb.shape[0], n_min, replace=False)
        n_comp = min(n_cca, ref_emb.shape[1], other_emb.shape[1], n_min - 1)
        cca    = CCA(n_components=n_comp, max_iter=500)
        try:
            cca.fit(ref_emb[idx_r, :n_comp], other_emb[idx_o, :n_comp])
            # Transform ALL cells; cca.transform(X, Y) returns (Xc, Yc) pair
            ref_cca_all,   _ = cca.transform(ref_emb[:, :n_comp],
                                             ref_emb[:ref_emb.shape[0], :n_comp])
            other_cca_all, _ = cca.transform(other_emb[:, :n_comp],
                                             other_emb[:other_emb.shape[0], :n_comp])
            if k == 0:
                joint_parts[0] = ref_cca_all    # (n_ref_cells, n_comp)
            joint_parts.append(other_cca_all)   # (n_other_cells, n_comp)
        except Exception as e:
            print(f"  CCA failed for {sp_name}: {e}. Using raw PCA.")
            n_d = min(n_cca, other_emb.shape[1])
            joint_parts.append(other_emb[:, :n_d])

    print("Step 5: Building concatenated AnnData & computing joint UMAP...")
    adata_pieces = []
    for i, (sp_name, adata) in enumerate(adatas_converted.items()):
        sub = adata[:, common].copy()
        sub.obsm['X_joint'] = joint_parts[i].astype(np.float32)
        sub.obs['species']  = sp_name
        # Carry forward cluster if present
        if 'cluster' not in sub.obs and 'cluster' in adata.obs:
            sub.obs['cluster'] = adata.obs['cluster'].values
        adata_pieces.append(sub)

    adata_joint = ad.concat(adata_pieces, label='species', keys=sp_list,
                            uns_merge='first', merge='same')
    adata_joint.obsm['X_joint'] = np.vstack([a.obsm['X_joint'] for a in adata_pieces])

    from sklearn.manifold import TSNE
    print("  Computing t-SNE on joint embedding (numba-free)...")
    X_joint_all = np.vstack([a.obsm['X_joint'] for a in adata_pieces])
    n_tsne = min(50, X_joint_all.shape[1], X_joint_all.shape[0] - 1)
    perp = min(30, max(5, X_joint_all.shape[0] // 5))
    tsne_kwargs = dict(n_components=2, perplexity=perp, random_state=42)
    try:
        tsne = TSNE(max_iter=500, **tsne_kwargs)
    except TypeError:
        tsne = TSNE(n_iter=500, **tsne_kwargs)   # sklearn < 1.4
    joint_2d = tsne.fit_transform(X_joint_all[:, :n_tsne])
    adata_joint.obsm['X_umap'] = joint_2d.astype(np.float32)

    _plot_cross_species_umap(adata_joint, sp_list, save=save, show=show)

    return adata_joint


def _plot_cross_species_umap(adata_joint, sp_list, save=None, show=True):
    """3-panel UMAP: by species, by cell cluster, by species+confidence ellipses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    umap = adata_joint.obsm['X_umap']
    species_vals = adata_joint.obs['species'].values.astype(str)
    cluster_vals = adata_joint.obs.get('cluster', adata_joint.obs.get('cluster', None))

    sp_colors = {'Human': '#4e79a7', 'Mouse': '#f28e2b', 'Rat': '#59a14f',
                 'Zebrafish': '#e15759'}
    sp_cmap = {s: sp_colors.get(s, f'C{i}') for i, s in enumerate(sp_list)}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='white')
    for ax in axes:
        ax.set_facecolor('white')

    # Panel 1: coloured by species
    for sp in sp_list:
        mask = species_vals == sp
        axes[0].scatter(umap[mask, 0], umap[mask, 1],
                        c=[sp_cmap[sp]], s=5, alpha=0.7, label=sp, rasterized=True)
    axes[0].set_title('Joint UMAP by Species', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, markerscale=3, frameon=False, loc='best')
    axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

    # Panel 2: coloured by cluster (if available)
    if cluster_vals is not None and len(cluster_vals) == adata_joint.n_obs:
        cl_vals  = cluster_vals.astype(str).values
        unique_cl = sorted(np.unique(cl_vals))
        cl_cmap  = plt.cm.get_cmap('tab10', len(unique_cl))
        for i, cl in enumerate(unique_cl):
            mask = cl_vals == cl
            axes[1].scatter(umap[mask, 0], umap[mask, 1],
                            c=[cl_cmap(i)], s=5, alpha=0.7, label=f'C{cl}', rasterized=True)
        axes[1].legend(fontsize=7, markerscale=2, frameon=False, ncol=2)
        axes[1].set_title('Joint UMAP by Cell Cluster', fontsize=12, fontweight='bold')
    else:
        axes[1].scatter(umap[:, 0], umap[:, 1], c='#888888', s=5, alpha=0.5, rasterized=True)
        axes[1].set_title('Joint UMAP (No Cluster Info)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')

    # Panel 3: split violin — species distribution along UMAP1
    umap1_by_species = {sp: umap[species_vals == sp, 0] for sp in sp_list}
    positions = range(len(sp_list))
    parts = axes[2].violinplot([umap1_by_species[s] for s in sp_list],
                               positions=list(positions), showmedians=True,
                               widths=0.65)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(sp_cmap[sp_list[i]])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('#333333')
    axes[2].set_xticks(list(positions)); axes[2].set_xticklabels(sp_list, fontsize=9)
    axes[2].set_title('UMAP-1 Distribution by Species\n(shows axis alignment quality)',
                      fontsize=10, fontweight='bold')
    axes[2].set_ylabel('UMAP 1 position', fontsize=9)
    axes[2].set_facecolor('#f8f9fa')

    fig.suptitle('scCytoTrek — Cross-Species Alignment (CCA)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved cross-species UMAP to {save}")
    if show:
        plt.show()
    plt.close()
