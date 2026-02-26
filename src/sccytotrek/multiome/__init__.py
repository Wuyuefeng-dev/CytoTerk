from .integration import (
    run_wnn,
    run_cca_integration,
    run_concat_pca_integration,
    run_procrustes_integration,
    run_snf_integration,
    run_joint_harmony
)
from .integration_v2 import (
    run_integration_benchmark,
    plot_integration_results,
    compute_integration_metrics,
    integrate_wnn,
    integrate_cca,
    integrate_concat_pca,
    integrate_procrustes,
    integrate_snf,
)

__all__ = [
    # Legacy
    "run_wnn",
    "run_cca_integration",
    "run_concat_pca_integration",
    "run_procrustes_integration",
    "run_snf_integration",
    "run_joint_harmony",
    # v2
    "run_integration_benchmark",
    "plot_integration_results",
    "compute_integration_metrics",
    "integrate_wnn",
    "integrate_cca",
    "integrate_concat_pca",
    "integrate_procrustes",
    "integrate_snf",
]
