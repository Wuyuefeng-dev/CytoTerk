"""
Survival analysis tools linking cell cluster representation to patient/mouse survival.
"""

from anndata import AnnData
import pandas as pd
import numpy as np

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

def compute_survival_by_cluster(
    adata: AnnData, 
    time_col: str, 
    event_col: str, 
    patient_col: str, 
    cluster_col: str = 'leiden_0.5',
    high_threshold: float = 0.5
) -> dict:
    """
    Compute Kaplan-Meier survival analysis based on the frequency 
    of each cell cluster within patients/mice.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    time_col : str
        Column in adata.obs with survival time.
    event_col : str
        Column in adata.obs with event status (1 = dead, 0 = censored).
    patient_col : str
        Column in adata.obs identifying the patient/mouse.
    cluster_col : str
        Column in adata.obs defining the cell clusters.
    high_threshold : float
        Threshold for dividing patients into high vs low cluster representation.
        Normally, this is a quantile (e.g., 0.5 for median).
        
    Returns
    -------
    dict
        Dictionary containing kmf (KaplanMeierFitter) objects and p-values for each cluster.
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError("Please install `lifelines` to run survival analysis (pip install lifelines).")
        
    print(f"Running survival analysis on {cluster_col} proportions...")
    
    # Needs to be a patient-level dataframe
    patient_data = adata.obs[[patient_col, time_col, event_col]].drop_duplicates(subset=[patient_col]).set_index(patient_col)
    
    # Calculate proportions
    counts = adata.obs.groupby([patient_col, cluster_col]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    
    # Combine
    combined = patient_data.join(proportions)
    
    results = {}
    
    # Drop rows with NaN in time or event
    combined = combined.dropna(subset=[time_col, event_col])
    
    clusters = proportions.columns
    for cluster in clusters:
        if cluster not in combined.columns:
            continue
            
        cluster_prop = combined[cluster]
        split_val = cluster_prop.quantile(high_threshold)
        
        ix_high = cluster_prop >= split_val
        ix_low = cluster_prop < split_val
        
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        
        try:
            kmf_high.fit(combined.loc[ix_high, time_col], event_observed=combined.loc[ix_high, event_col], label=f'{cluster} High')
            kmf_low.fit(combined.loc[ix_low, time_col], event_observed=combined.loc[ix_low, event_col], label=f'{cluster} Low')
            
            # Use logrank test for p-value if desired, though here we just return the fitters
            results[cluster] = {
                'kmf_high': kmf_high,
                'kmf_low': kmf_low,
                'split_val': split_val
            }
        except Exception as e:
            print(f"Could not fit survival for cluster {cluster}: {e}")
            
    print(f"Successfully computed Kaplan-Meier fits for {len(results)} clusters.")
    return results
