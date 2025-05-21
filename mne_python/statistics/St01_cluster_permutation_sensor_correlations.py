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
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir,'derivatives/correlations/bands_signif_correlations_subtraction_nooutlier-psd'),
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout' : op.join(quinna_dir, 'dataman/data_information/sensors_layout_names.csv')
    }

    return paths

def organise_csvs():
    """this def will organise csv files into one csv per significant substr-band, 
    that contains spearman rvalue for all 153 sensor pairs.
    Only run once."""
    # Set base directory and output directory
    paths = setup_paths(platform='mac')
    if not op.exists(paths['signif_correlation_dir']):
        os.makedirs(paths['signif_correlation_dir'])

    substrs_bands = [{'Thal':'Alpha'}, {'Puta':'Beta'}, {'Hipp':'Delta'}]

    for substr_band in substrs_bands:
        for substr, band in substr_band.items():
            # Prepare storage for sensor pair names and alpha r-values
            pair_names = []
            band_rvals = []

            # Loop through each sensor pair folder
            for pair_folder in os.listdir(paths['correlation_dir']):
                pair_path = os.path.join(paths['correlation_dir'], pair_folder)
                substr_csv_path = os.path.join(pair_path, substr, f'{substr}_lat_spectra_substr_spearmanr.csv')

                if os.path.isfile(substr_csv_path):
                    try:
                        df = pd.read_csv(substr_csv_path)
                        # Extract the alpha band r-value
                        band_r = df.loc[df.iloc[:,0] == band, '0'].values[0]
                        pair_names.append(pair_folder)
                        band_rvals.append(band_r)
                    except Exception as e:
                        print(f"Error reading {substr_csv_path}: {e}")

            # Create a DataFrame with sensor pair names on the left
            output_df = pd.DataFrame({
                'sensor_pair': pair_names,
                f'{band}_rval': band_rvals
            })

            # Save the resulting table
            output_csv_path = os.path.join(paths['signif_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
            output_df.to_csv(output_csv_path, index=False)

            print(f"Saved alpha correlation values to: {output_csv_path}")

def read_info(paths, ch_type='mag'):
    """This function inputs the type of channel you want to test significancy for,
    and reads the info object from a sample MEG file for adjacency information"""

    # read one meg file for the info
    raw = mne.io.read_raw_fif(paths['sample_meg_file'], verbose='ERROR')  # double check the verbose parameter
    sensor_layout = pd.read_csv(paths['sensor_layout'])
    right_sensors = sensor_layout['right_sensors'].dropna().tolist()  

    if ch_type == 'mag':
        right_mags = [ch for ch in right_sensors if ch.endswith('1')]
        raw = raw.pick('mag').pick(right_mags)
    elif ch_type == 'grad':
        right_grads = [ch for ch in right_sensors if not ch.endswith('1')]
        raw = raw.pick('grad').pick(right_grads)
    else:
        raise ValueError("ch_type must be 'mag' or 'grad'")
    
    return raw.info


def find_custom_adjacency(info, ch_type):
    """
    Computes adjacency matrix for the channels in the provided info object only.

    Parameters
    ----------
    info : mne.Info
        Info object containing the subset of channels.
    ch_type : str
        'mag' or 'grad'.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix
        Channel adjacency matrix.
    ch_names : list of str
        Channel names in the adjacency matrix.
    """
    full_adj, full_ch_names = find_ch_adjacency(info, ch_type=ch_type)

    # Keep only those adjacency rows/cols that match info['ch_names']
    mask = [ch in info['ch_names'] for ch in full_ch_names]
    adjacency = full_adj[mask][:, mask]
    ch_names = [ch for ch in full_ch_names if ch in info['ch_names']]

    return adjacency, ch_names



def run_cluster_test_for_correlations(paths, substrs_bands, adjacency, ch_names, ch_type):
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
    ch_type : 'mag' or 'grad'
        the type of channels you are testing significancy for. Should correspond to the info
    """

    substrs_bands = [{'Thal':'Alpha'}, {'Puta':'Beta'}, {'Hipp':'Delta'}]
    for item in substrs_bands:
        for substr, band in item.items():
            fname = f'{substr}_allpairs_{band}_spearmanr.csv'
            fpath = os.path.join(paths['signif_correlation_dir'], fname)

            # Load correlation values (shape: n_sensors,)
            df = pd.read_csv(fpath)
            sensor_pairs = df['sensor_pair'].astype(str)  # Ensure strings

            if ch_type == 'mag':
                # Keep only sensor pairs ending in '1'
                mask = sensor_pairs.str.endswith('1')
            elif ch_type == 'grad':
                # Keep only sensor pairs NOT ending in '1'
                mask = ~sensor_pairs.str.endswith('1')
            else:
                raise ValueError("ch_type must be 'mag' or 'grad'.")
            
        # Filter values based on ch_type
        filtered_sensor_pairs = sensor_pairs[mask].tolist()
        r_vals = df.loc[mask, f'{band.lower()}_rval'].values.squeeze()
        ch_names = filtered_sensor_pairs  # for saving cluster names

        # Fisher z-transform and reshape
        z_vals = np.arctanh(r_vals)
        X = z_vals[np.newaxis, :]  # shape: (1, n_sensors)

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
        out_txt = os.path.join(paths['signif_correlation_dir'], f'{substr}_{band}_significant_clusters.txt')
        with open(out_txt, 'w') as f:
            for i in sig_clusters:
                cluster_sensors = np.array(ch_names)[clusters[i]]
                f.write(f"Cluster {i + 1} (p = {p_values[i]:.3f}):\n")
                f.write(", ".join(cluster_sensors) + "\n\n")

        print(f"Finished {substr}-{band}: {len(sig_clusters)} significant clusters")



def main():
    paths = setup_paths(platform='mac')



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
