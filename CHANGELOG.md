# CHANGELOG — scCytoTrek

## 2026-02-26 — Session: SeuratExtend Style + Five New Features

### New Modules
| File | Description |
|------|-------------|
| `src/sccytotrek/plotting/style.py` | SeuratExtend aesthetics: 12-colour NPG `SEURAT_DISCRETE` palette, `apply_seurat_theme(ax)`, `seurat_figure(nrows, ncols)` factory, `SEURAT_FEATURE_CMAP`, `SEURAT_CORR_CMAP`, `SEURAT_DOTPLOT_CMAP` |
| `src/sccytotrek/trajectory/pseudotime_genes.py` | `find_pseudotime_genes()` — Spearman correlation ranking; `plot_pseudotime_heatmap()` — smoothed z-scored clustered heatmap |
| `src/sccytotrek/interaction/umap_arcs.py` | `plot_cell2cell_umap()` — Bézier arc CCI plot on UMAP |

### Enhanced Modules
| File | Change |
|------|--------|
| `src/sccytotrek/lineage/visualization.py` | `plot_clonal_streamgraph()` — added `show_barcode_timeline` parameter; new barcode event timeline panel with horizontal clone spans and cell rug marks |
| `src/sccytotrek/tools/cell_type.py` | Added `run_celltypist()` (auto-normalise, majority voting, model download, confidence filter, ImportError fallback) and `plot_celltypist_umap()` (3-panel SeuratExtend figure) |

### API Exports Updated
| `__init__.py` | New exports |
|---------------|-------------|
| `sccytotrek/trajectory/__init__.py` | `find_pseudotime_genes`, `plot_pseudotime_heatmap` |
| `sccytotrek/interaction/__init__.py` | `plot_cell2cell_umap` |
| `sccytotrek/tools/__init__.py` | `run_celltypist`, `plot_celltypist_umap` |
| `sccytotrek/plotting/__init__.py` | All style constants and helpers |

### Script Updates
| File | Change |
|------|--------|
| `generate_walkthrough_figs.py` | Fully rewritten with SeuratExtend style; fixed `IndexError` in `bulk_colors` |

### Figures Generated (SeuratExtend style)
| Figure | Path |
|--------|------|
| Bulk RNA Alignment | `demo_figs/bulk_alignment.png` |
| Tipping Genes Barplot | `demo_figs/tipping_genes_barplot.png` |
| TF Score Ranking | `demo_figs/tf_score_ranking.png` |

### Documentation
| File | Change |
|------|--------|
| `README.md` | Key-features section rewritten — 11 modules with all new features |
| `walkthrough.md` | Sections 11–15 added: pseudotime heatmap, CCI arc plot, barcode timeline, CellTypist, drug target testing |
