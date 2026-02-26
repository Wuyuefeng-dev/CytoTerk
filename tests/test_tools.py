import pytest
import numpy as np
import anndata as ad
from scsccytotrek import tools as tl
from scsccytotrek import tools as tl

@pytest.fixture
def mock_adata():
    np.random.seed(42)
    X = np.random.poisson(2, (100, 50)).astype(np.float32)
    adata = ad.AnnData(X=X)
    return adata

def test_run_pca_and_neighbors(mock_adata):
    tl.run_pca_and_neighbors(mock_adata, n_pcs=10, n_neighbors=5)
    assert "X_pca" in mock_adata.obsm
    assert "neighbors" in mock_adata.uns

def test_run_umap_and_cluster(mock_adata):
    # Need neighbors first
    tl.run_pca_and_neighbors(mock_adata, n_pcs=10, n_neighbors=5)
    tl.run_umap_and_cluster(mock_adata, resolution=0.5)
    
    assert "X_umap" in mock_adata.obsm
    assert "leiden_0.5" in mock_adata.obs.columns
