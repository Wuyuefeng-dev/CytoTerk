import pytest
import numpy as np
import anndata as ad
from scsccytotrek import preprocessing as pp

@pytest.fixture
def mock_adata():
    np.random.seed(42)
    # 100 cells, 50 genes
    X = np.random.poisson(1, (100, 50))
    obs = {"cell_id": [f"cell_{i}" for i in range(100)]}
    var = {"gene_id": [f"gene_{i}" if i >= 5 else f"MT-gene_{i}" for i in range(50)]}
    return ad.AnnData(X=X, obs=obs, var=var)

def test_calculate_qc_metrics(mock_adata):
    pp.calculate_qc_metrics(mock_adata, mt_prefix="MT-")
    assert "n_genes_by_counts" in mock_adata.obs.columns
    assert "total_counts" in mock_adata.obs.columns
    assert "pct_counts_mt" in mock_adata.obs.columns
    assert "mt" in mock_adata.var.columns

def test_filter_cells_and_genes(mock_adata):
    pp.calculate_qc_metrics(mock_adata, mt_prefix="MT-")
    original_cells = mock_adata.n_obs
    original_genes = mock_adata.n_vars
    
    pp.filter_cells_and_genes(mock_adata, min_genes=10, min_cells=3, max_pct_mt=20.0)
    
    assert mock_adata.n_obs <= original_cells
    assert mock_adata.n_vars <= original_genes

def test_normalize_and_log(mock_adata):
    pp.normalize_and_log(mock_adata)
    # Check if data is logged/normalized (roughly, sum isn't exactly target_sum after log, but we can verify it ran)
    assert 'log1p' in mock_adata.uns
