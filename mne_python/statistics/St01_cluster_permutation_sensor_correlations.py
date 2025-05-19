"""
========================================================
St01_cluster_permutation_sensor_correlations

This code:
    1. organises the significant rvalues
    between bands and substrs into one csv with 
    153 rows (#sensor pair) and two columns (pair names and 
    rvalues). for e.g., alpha for thalamus, beta for putamen, 
    and delta/theta for hippocampus
    2. calculates the cluster permutation tests for 
    each of them separately.

Author: Tara Ghafari
tara.ghafari@gmail.com
Date: 16/05/2025
============================================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import ttest_1samp

from mpl_toolkits.mplot3d import Axes3D

def setup_paths(platform='mac'):
    """Set up file paths for different platforms."""
    if platform == 'bluebear':
        quinna_dir = '/rds/projects/q/quinna-camcan/'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
        output_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
        output_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'output_dir': op.join(sub2ctx_dir,'derivatives/correlations/bands_signif_correlations_subtraction_nooutlier-psd'),
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
    }

    return paths


def organise_csvs():
    """this def will organise csv files into one csv per significant substr-band, 
    that contains spearman rvalue for all 153 sensor pairs.
    Only run once."""
    # Set base directory and output directory
    paths = setup_paths(platform='mac')
    if not op.exists(paths['output_dir']):
        os.makedirs(paths['output_dir'])

    substrs_bands = [{'Thal':'Alpha'}, {'Puta':'Beta'}, {'Hipp':'Delta'}]

    for substr_band in substrs_bands:
        for substr, band in substr_band.items():
            # Prepare storage for sensor pair names and alpha r-values
            pair_names = []
            alpha_rvals = []

            # Loop through each sensor pair folder
            for pair_folder in os.listdir(paths['correlation_dir']):
                pair_path = os.path.join(paths['correlation_dir'], pair_folder)
                substr_csv_path = os.path.join(pair_path, substr, f'{substr}_lat_spectra_substr_spearmanr.csv')

                if os.path.isfile(substr_csv_path):
                    try:
                        df = pd.read_csv(substr_csv_path)
                        # Extract the alpha band r-value
                        alpha_r = df.loc[df.iloc[:,0] == band, '0'].values[0]
                        pair_names.append(pair_folder)
                        alpha_rvals.append(alpha_r)
                    except Exception as e:
                        print(f"Error reading {substr_csv_path}: {e}")

            # Create a DataFrame with sensor pair names on the left
            output_df = pd.DataFrame({
                'sensor_pair': pair_names,
                'alpha_rval': alpha_rvals
            })

            # Save the resulting table
            output_csv_path = os.path.join(paths['output_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
            output_df.to_csv(output_csv_path, index=False)

            print(f"Saved alpha correlation values to: {output_csv_path}")


def run_cluster_test_for_correlations(paths, substrs_bands, info):
    """
    Run a one-sample cluster permutation test on sensor correlation values for each subcortical structure and band.

    Parameters:
    ------------
    paths : dict
        Dictionary containing paths to correlation and output directories.
    substrs_bands : list of dict
        List of dictionaries mapping subcortical structures to oscillatory bands.
    info : mne.Info
        MNE Info object with sensor locations (must match sensor count in CSV).
    """
    adjacency, ch_names = find_ch_adjacency(info, ch_type='mag')

    for item in substrs_bands:
        for substr, band in item.items():
            fname = f'{substr}_allpairs_{band}_spearmanr.csv'
            fpath = os.path.join(paths['correlation_dir'], fname)

            # Load correlation values (shape: n_sensors,)
            r_vals = pd.read_csv(fpath, header=None).values.squeeze()
            z_vals = np.arctanh(r_vals)  # Fisher z-transform
            X = z_vals[np.newaxis, :]  # shape (1, n_sensors)

            # Run permutation cluster test
            T_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
                X,
                threshold=None,
                n_permutations=1000,
                tail=0,
                adjacency=adjacency,
                out_type='mask',
                verbose=True
            )

            # Save results
            sig_clusters = np.where(p_values < 0.05)[0]
            out_txt = os.path.join(paths['output_dir'], f'{substr}_{band}_significant_clusters.txt')
            with open(out_txt, 'w') as f:
                for i in sig_clusters:
                    cluster_sensors = np.array(ch_names)[clusters[i]]
                    f.write(f"Cluster {i+1} (p = {p_values[i]:.3f}):\n")
                    f.write(", ".join(cluster_sensors) + "\n\n")
            print(f"Finished {substr}-{band}: {len(sig_clusters)} significant clusters")



def main():
    paths = setup_paths(platform='mac')
    # read one meg file for the info
    meg_fname =  (paths['sample_meg_file'])
    raw = mne.io.read_raw_fif(meg_fname)
    magraw = raw.copy().pick('mag')
    info = magraw.info
    # Prompt user for input
    freq_input = input("Enter frequency (e.g., 5.0 or Alpha): (make sure input a float number or band name)").strip()
    structure = input("Enter subcortical structure (e.g., Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    sensor = input("Enter sensor type (grad or mag): ").strip()
    do_plot_3d_input = input("Plot 3D visualization? (y/n): ").strip().lower()
    do_plot_3d = do_plot_3d_input == 'y'

    corr_file = op.join(paths['correlation_dir'], f'spearman-r_src_lat_power_vol_{sensor}_{freq_input}.csv')
    pval_file = op.join(paths['correlation_dir'], f'spearman-pval_src_lat_power_vol_{sensor}_{freq_input}.csv')
    
    if not op.exists(corr_file) or not op.exists(pval_file):
        print(f"Files not found for frequency {freq_input} and sensor {sensor}.")
        return
    
    df_corr = pd.read_csv(corr_file, index_col=None)
    df_pval = pd.read_csv(pval_file, index_col=None)
    
    if structure not in df_corr.columns:
        print(f"Structure {structure} not found in correlation file.")
        return
    
    correlation_values = df_corr[structure].values
    p_values = df_pval[structure].values
    significant_mask = p_values < 0.05
    
    # Read fsaverage source space
    fname_fsaverage_src = op.join(paths['fs_sub_dir'], "fsaverage", "bem", "fsaverage-vol-5-src.fif")
    src_fs = mne.read_source_spaces(fname_fsaverage_src)
    
    # Compute right hemisphere grid positions and indices
    grid_positions, right_indices = compute_hemispheric_index(src_fs)
    
    # Plot scatter of correlation values
    scatter_output = op.join(paths['output_base'], structure, f'src-correlation_{freq_input}.png')
    os.makedirs(op.dirname(scatter_output), exist_ok=True)
    plot_scatter(grid_positions, correlation_values, significant_mask, scatter_output)
    
    # Create a volume estimate using the correlation data
    stc, vol_mask = create_volume_estimate(correlation_values, significant_mask, src_fs, right_indices)
    
    # Plot volume estimate on MRI and optionally in 3D
    plot_volume_estimate(stc, vol_mask, src_fs, paths, freq_input, sensor, structure, do_plot_3d)

if __name__ == "__main__":
    main()
