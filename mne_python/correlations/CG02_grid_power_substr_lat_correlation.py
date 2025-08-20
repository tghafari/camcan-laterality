"""
==============================================
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

For band calculations we average over correlation values across
band frequencies. 
At no point in this pipeline the DICS or CSDs are calculated for
band, all are per Hz.

Output:
- 235 CSV files: `spearman_src_lat_power_vol_grad(or mag)_1.5.csv`
  (each containing Spearman r and p-values for all grid indices).

written by Tara Ghafari
last updated 27/06/2025
tara.ghafari@gmail.com
==============================================
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        quinna_dir = '/rds/projects/q/quinna-camcan/'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'sub_list': os.path.join(quinna_dir, 'dataman/data_information/last_FINAL_sublist-vol-outliers-removed.csv'),  # we are only using no vol outliers for final analysis 16/08/2025
        'len_subs': os.path.join(quinna_dir, 'dataman/data_information/source_subs'),
        'meg_source_all_subs_dir': os.path.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer/all_subs'),
        'lateralization_volumes': os.path.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_no-vol-outliers.csv'),
        'output_dir': os.path.join(sub2ctx_dir, 'derivatives/correlations'),
    }

    return paths

def load_lateralization_volumes(file_path):
    """Load the lateralization volumes CSV file."""
    lat_vols = pd.read_csv(file_path, index_col=0)
    lat_vols['subject_ID'] = lat_vols['subject_ID'].astype(str)  # Convert subject IDs to string
    lat_vols = lat_vols.set_index('subject_ID')  # Set subject_ID as index

    return lat_vols 

def compute_spearman(paths, lat_src_file, lat_vols, sensor, freq):
    """Compute Spearman correlation between source data and lateralization volumes."""
    final_sub_list = pd.read_csv(paths['sub_list'])['subjectID'].astype(str).tolist()
    lat_src_data = pd.read_csv(lat_src_file, index_col=None)
    lat_src_data.columns = lat_src_data.columns.astype(str)  # Ensure consistent dtype

    # Get valid common subjects across all three: final list, source data, and volume data
    valid_subjects = (
        set(lat_src_data.columns)
        .intersection(lat_vols.index)
        .intersection(final_sub_list)
    )

    if len(valid_subjects) == 0:
        print(f"No overlapping subjects found for {lat_src_file}")
        return None

    valid_subjects = sorted(valid_subjects)  # ensure consistent ordering
    out_filename = os.path.join(paths['len_subs'], f"{sensor}_{freq}.csv")
    pd.DataFrame({'n_subjects': [len(valid_subjects)]}).to_csv(out_filename, index=False)

    lat_src_data = lat_src_data[valid_subjects].T  # shape: subjects × grid indices
    lat_vols = lat_vols.loc[valid_subjects]
        
    # Define subcortical structures
    subcortical_structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

    # Create empty arrays to store correlation results
    r_values = np.zeros((lat_src_data.shape[1], len(subcortical_structures)))
    p_values = np.zeros((lat_src_data.shape[1], len(subcortical_structures)))

    # Compute Spearman correlation per grid index and per subcortical structure
    for grid_idx in range(lat_src_data.shape[1]):  # Iterate over 7011 grid indices
        for struct_idx, structure in enumerate(subcortical_structures):  # Iterate over subcortical structures
            r, p = spearmanr(lat_src_data.iloc[:, grid_idx], lat_vols[structure])
            if p == np.nan or r == np.nan:
                print(f'{grid_idx} in {structure} is NaN')
            r_values[grid_idx, struct_idx] = r
            p_values[grid_idx, struct_idx] = p

    spearman_r = pd.DataFrame(r_values, columns=subcortical_structures)
    spearman_pval = pd.DataFrame(p_values, columns=subcortical_structures)

    return spearman_r, spearman_pval

def process_correlations(platform='mac', sensortypes=['grad', 'mag'], spec=False):
    """Process all MEG source frequency files and compute Spearman correlations with lateralised volumes."""
    paths = setup_paths(platform)
    lat_vols = load_lateralization_volumes(paths['lateralization_volumes'])
    

    for sensor in sensortypes:
        print(f'Processing {sensor}')
        if spec:
            freqs=np.arange(1.5, 60, 0.5)
            # --- Per-frequency correlation ---
            for freq in freqs:
                print(f'Processing spectrum at {freq}')
                lat_src_file = os.path.join(paths['meg_source_all_subs_dir'],
                                        f'all_subs_lateralised_src_power_{sensor}_{freq}.csv')
                if os.path.exists(lat_src_file):
                    spearman_r, spearman_pval = compute_spearman(paths, lat_src_file, lat_vols, sensor, freq)

                    if spearman_r is not None:
                        output_file = os.path.join(paths['output_dir'], 'src_lat_grid_vol_correlation_no-vol-outliers',
                                                f'FINAL_spearman-r_src_lat_power_vol_{sensor}_{freq}.csv')
                        spearman_r.to_csv(output_file, index=False)
                    if spearman_pval is not None:
                        output_file = os.path.join(paths['output_dir'], 'src_lat_grid_vol_correlation_no-vol-outliers',
                                                f'FINAL_spearman-pval_src_lat_power_vol_{sensor}_{freq}.csv')
                        spearman_pval.to_csv(output_file, index=False)

                        print(f"Saved: {output_file}")
                else:
                    print(f"Missing source data file: {lat_src_file}")

        # --- Bandwise correlation ---
        if not spec:
            bands = {
                'Delta': (1.5, 4),  # this only differs from other (1, 4) because there's no source for 1Hz.
                'Theta': (4, 8),
                'Alpha': (8, 14),  
                'Beta': (14, 40)}
            
            print(f'\n  Computing bandwise correlations for {sensor}')
            for band_name, (low, high) in bands.items():
                print(f'    Processing band: {band_name} ({low}-{high} Hz)')

                band_freqs = np.arange(low, high, 0.5)
                band_data = []

                for freq in band_freqs:
                    lat_src_file = os.path.join(paths['meg_source_all_subs_dir'],
                                                f'all_subs_lateralised_src_power_{sensor}_{freq}.csv')
                    if os.path.exists(lat_src_file):
                        df = pd.read_csv(lat_src_file, index_col=None)                  
                        band_data.append(df)
                    else:
                        print(f"      Missing file for {freq} Hz")

                if band_data:
                    data_array = np.stack([df.values for df in band_data], axis=2)

                    # Compute mean across the frequency axis (axis=2), ignoring NaNs
                    avg_array = np.nanmean(data_array, axis=2)

                    # Reconstruct the DataFrame with same subjects and columns
                    avg_band_data = pd.DataFrame(avg_array, columns=band_data[0].columns, index=band_data[0].index)

                    avg_band_file = os.path.join(paths['meg_source_all_subs_dir'],
                                                f'FINAL_all_subs_lateralised_src_power_{sensor}_{band_name}_band_avg.csv')
                    avg_band_data.to_csv(avg_band_file)
                    
                    # Now compute Spearman correlation
                    spearman_r, spearman_pval = compute_spearman(paths, avg_band_file, lat_vols, sensor, band_name)

                    if spearman_r is not None:
                        spearman_r.to_csv(os.path.join(paths['output_dir'], 'src_lat_grid_vol_correlation_no-vol-outliers',
                                                    f'FINAL_spearman-r_src_lat_power_vol_{sensor}_{band_name}.csv'),
                                        index=False)
                    if spearman_pval is not None:
                        spearman_pval.to_csv(os.path.join(paths['output_dir'], 'src_lat_grid_vol_correlation_no-vol-outliers',
                                                        f'FINAL_spearman-pval_src_lat_power_vol_{sensor}_{band_name}.csv'),
                                            index=False)
                else:
                    print(f"    No data found for band {band_name}")

if __name__ == "__main__":
    process_correlations()
