import pytest
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from scsccytotrek import plotting as pl
from scsccytotrek import tools as tl

@pytest.fixture
def mock_adata():
    np.random.seed(42)
    X = np.random.poisson(2, (50, 20)).astype(np.float32)
    adata = ad.AnnData(X=X)
    tl.run_pca_and_neighbors(adata, n_pcs=10, n_neighbors=5)
    tl.run_umap_and_cluster(adata, resolution=0.5)
    # Add a mock feature
    adata.obs["mock_feature"] = np.random.rand(50)
    return adata

def test_dim_plot(mock_adata):
    ax = pl.dim_plot(mock_adata, color="leiden_0.5", show=False)
    assert ax is not None or type(ax) is list
    plt.close("all")

def test_feature_plot(mock_adata):
    ax = pl.feature_plot(mock_adata, features="mock_feature", show=False)
    assert ax is not None or type(ax) is list
    plt.close("all")
