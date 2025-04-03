"""
CG02_grid_power_substr_lat_correlation

This script calculates Spearman correlation (r and p-value) between
lateralized MEG source power (per Hz, per grid index) and lateralization volumes
of subcortical structures.

Steps:
    1. Load lateralization volumes (subcortical structures) from CSV.
    2. Iterate over 235 MEG CSV files (one per 0.5Hz, 7011 rows × #subjects).
    3. Match subjects between MEG and lateralization volumes datasets.
    4. Compute Spearman correlation (r, p-value) for each grid index.
    5. Save correlation results per Hz in a CSV file.

Output:
- 235 CSV files: `spearman_src_lat_power_vol_grad(or mag)_1.5.csv`
  (each containing Spearman r and p-values for all grid indices).

written by Tara Ghafari
tara.ghafari@gmail.com
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'meg_source_all_subs_dir': os.path.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer/all_subs'),
        'lateralization_volumes': os.path.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_nooutliers.csv'),
        'output_dir': os.path.join(sub2ctx_dir, 'derivatives/correlations/src_lat_grid_vol_correlation_nooutliers')
    }

    return paths

def load_lateralization_volumes(file_path):
    """Load the lateralization volumes CSV file."""
    lat_vols = pd.read_csv(file_path, index_col=0)
    lat_vols['subject_ID'] = lat_vols['subject_ID'].astype(str)  # Convert subject IDs to string
    lat_vols = lat_vols.set_index('subject_ID')  # Set subject_ID as index

    return lat_vols 

def compute_spearman(lat_src_file, lat_vols):
    print('Computing Spearman correlation between source data and lateralization volumes.')
    lat_src_data = pd.read_csv(lat_src_file, index_col=0)
    common_subjects = lat_src_data.columns.intersection(lat_vols.index)
    
    if len(common_subjects) == 0:
        print(f"No overlapping subjects found for {lat_src_file}")
        return None
    
    lat_src_data = lat_src_data.reset_index(drop=True)  # Ensure the DataFrame index is reset so we don't retain old subject indices
    lat_src_data = lat_src_data[common_subjects].T  # Align subjects
    lat_vols = lat_vols.loc[common_subjects]  # Align subjects
        
    # Define subcortical structures
    subcortical_structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

    # Create empty arrays to store correlation results
    r_values = np.zeros((lat_src_data.shape[1], len(subcortical_structures)))
    p_values = np.zeros((lat_src_data.shape[1], len(subcortical_structures)))

    # Compute Spearman correlation per grid index and per subcortical structure
    for grid_idx in range(lat_src_data.shape[1]):  # Iterate over 7011 grid indices
        for struct_idx, structure in enumerate(subcortical_structures):  # Iterate over subcortical structures
            r, p = spearmanr(lat_src_data.iloc[:, grid_idx], lat_vols[structure])
            r_values[grid_idx, struct_idx] = r
            p_values[grid_idx, struct_idx] = p

    spearman_r = pd.DataFrame(r_values, columns=subcortical_structures)
    spearman_pval = pd.DataFrame(p_values, columns=subcortical_structures)

    return spearman_r, spearman_pval

def process_correlations(platform='bluebear', freqs=np.arange(1.5, 5, 0.5), sensortypes=['grad', 'mag']):
    print('Processing all MEG source frequency files and computing Spearman correlations with lateralised volumes.')
    paths = setup_paths(platform)
    lat_vols = load_lateralization_volumes(paths['lateralization_volumes'])
    
    for sensor in sensortypes:
        for freq in freqs:
            lat_src_file = os.path.join(paths['meg_source_all_subs_dir'],
                                    f'all_subs_lateralised_src_power_{sensor}_{freq}.csv')
            if os.path.exists(lat_src_file):
                spearman_r, spearman_pval = compute_spearman(lat_src_file, lat_vols)

                if spearman_r is not None:
                    output_file = os.path.join(paths['output_dir'],
                                               f'spearman-r_src_lat_power_vol_{sensor}_{freq}.csv')
                    spearman_r.to_csv(output_file, index=False)
                if spearman_pval is not None:
                    output_file = os.path.join(paths['output_dir'],
                                               f'spearman-pval_src_lat_power_vol_{sensor}_{freq}.csv')
                    spearman_pval.to_csv(output_file, index=False)

                    print(f"Saved: {output_file}")
            else:
                print(f"Missing source data file: {lat_src_file}")

if __name__ == "__main__":
    process_correlations()
