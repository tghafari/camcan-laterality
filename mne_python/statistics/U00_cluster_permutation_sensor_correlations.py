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
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test
from scipy.stats import ttest_1samp, spearmanr

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
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index'),  # directory of laterlised band power
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_nooutliers.csv'),  # directory of lateralised volume of substrs
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



def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """Merge the dataframes containing spectrum lateralization values and 
    subcortical structure lateralization values together."""

    # Navigate to the sensor_pair folder
    spec_lat_index_fname = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')

    # Load lateralization index for each pair
    spectrum_pair_lat_df = pd.read_csv(spec_lat_index_fname)
    spectrum_pair_lat_df = spectrum_pair_lat_df.rename(columns={'Unnamed: 0':'subject_ID'})
    
    # Merge and match the subject_ID column and remove nans
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['subject_ID'])
    working_df = working_df.dropna()

    # Get the freqs of spectrum from spec_pair_lat
    freqs = spectrum_pair_lat_df.columns.values[1:]  # remove subject_ID column
    freqs = [float(freq) for freq in freqs]  # convert strings to floats
    return working_df, freqs

def calculate_band_power(working_df, freqs, band):
    """Calculate the average power within a specified frequency band."""
    # Round frequencies to one decimal place to match the column names in working_df
    freqs_rounded = [round(f, 1) for f in freqs]

    # Select frequencies that fall within the band range
    band_freqs = [f for f in freqs_rounded if band[0] <= f <= band[1]]
    
    # Ensure the selected frequencies are actually in the DataFrame columns
    band_freqs = [str(f) for f in band_freqs if str(f) in working_df.columns]

    # Check if there are any valid frequencies selected
    if len(band_freqs) == 0:
        raise ValueError(f"No frequencies found in the range {band[0]}-{band[1]} Hz in the data.")
    
    # Calculate the average power across the selected frequencies
    band_power = working_df[band_freqs].mean(axis=1)
    
    return band_power

# Define sensors and band limits
sensor_pairs = paths['sensor_layout'] 
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# Load subcortical LV CSV
substr_lat_df = pd.read_csv(paths['LV_csv'])  # shape: (subjects, 7)

# Create container for each band's power across sensor pairs
band_li_dict = {band: [] for band in bands}
subject_ids = None

# Loop through sensor pairs
for left_sensor, right_sensor in sensor_pairs:
    working_df, freqs = working_df_maker(paths['spectra_dir'], left_sensor, right_sensor, substr_lat_df)

    # Save subject IDs
    if subject_ids is None:
        subject_ids = working_df['subject_ID']

    # Compute band power and store
    for band_name, band_range in bands.items():
        band_power = calculate_band_power(working_df, freqs, band_range)
        band_li_dict[band_name].append(band_power.values)

# Concatenate all sensor pairs for each band: subjects x sensor_pairs
for band_name, band_matrix in band_li_dict.items():
    band_li = np.stack(band_matrix, axis=1)
    df_out = pd.DataFrame(band_li, columns=[f'{i}' for i in range(band_li.shape[1])])
    df_out.insert(0, 'subject_ID', subject_ids.values)
    
    out_path = op.join(paths['LI_dir'], f'{band_name}_lateralised_power_allsens_subtraction_nonoise.csv')
    df_out.to_csv(out_path, index=False)

# --- Raw Spearman correlations ---
# Load LV again (aligned with saved subject_IDs)
lv_df = substr_lat_df.set_index('subject_ID').loc[subject_ids].reset_index()

# For each (band, substructure) pair
band_substr_map = {
    'Alpha': 'Thal',
    'Beta': 'Puta',
    'Delta': 'Hipp'
}

for band, substr in band_substr_map.items():
    band_csv = op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv')
    band_df = pd.read_csv(band_csv)
    lv_vals = lv_df[substr].values

    rval_list = []
    for ch in band_df.columns[1:]:  # skip subject_ID
        li_vals = band_df[ch].values
        rval, _ = spearmanr(lv_vals, li_vals)
        rval_list.append(rval)

    # Save correlation values
    out_df = pd.DataFrame({
        'sensor_pair': band_df.columns[1:],
        f'{band.lower()}_rval': rval_list
    })
    out_df.to_csv(op.join(paths['LI_dir'], f'{substr}_allpairs_{band}_spearmanr.csv'), index=False)

# You can now call `run_cluster_test_for_correlations(...)` to run the permutation tests.

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

def run_cluster_test_from_raw_corr(paths, ch_type='mag'):
    """
    Runs cluster-based permutation tests on correlations between subcortical LVs
    and MEG power lateralization indices for specific bands.

    Parameters:
    ------------
    paths : dict
        Dictionary containing keys: 'LI_dir', 'LV_csv', 'sensor_layout', 'sample_meg_file', 'signif_correlation_dir'
    ch_type : str
        'mag' or 'grad'
    """

    substrs_bands = [{'Thal': 'Alpha'}, {'Puta': 'Beta'}, {'Hipp': 'Delta'}]

    # Load subcortical LVs
    lv_df = pd.read_csv(paths['LV_csv'])  # shape: (n_subjects, 7)

    # Get sensor info and adjacency
    info = read_info(paths, ch_type=ch_type)
    adjacency, ch_names = find_custom_adjacency(info, ch_type=ch_type)

    for item in substrs_bands:
        for substr, band in item.items():

            # Load LI data
            li_fname = os.path.join(paths['LI_dir'], f"{band}_lateralised_power_allsens_subtraction_nonoise.csv")
            li_df = pd.read_csv(li_fname)

            # Filter sensor columns by sensor type (mag or grad)
            if ch_type == 'mag':
                selected_cols = [col for col in li_df.columns if col.endswith('1')]
            else:  # grad
                selected_cols = [col for col in li_df.columns if not col.endswith('1')]

            li_filtered = li_df[selected_cols].to_numpy()
            lv_vector = lv_df[substr].to_numpy()  # (n_subjects,)

            # Sanity check
            assert li_filtered.shape[0] == lv_vector.shape[0], f"Subject mismatch for {substr}-{band}"

            n_subjects, n_sensors = li_filtered.shape

            # --- Compute observed correlations ---
            r_obs = np.array([
                spearmanr(lv_vector, li_filtered[:, ch])[0]
                for ch in range(n_sensors)
            ])

            z_obs = np.arctanh(r_obs)  # Fisher transform

            # --- Permutation Testing ---
            n_permutations = 1000
            z_null = np.zeros((n_permutations, n_sensors))
            rng = np.random.RandomState(42)

            for i in range(n_permutations):
                shuffled_lv = rng.permutation(lv_vector)
                z_null[i] = [
                    np.arctanh(spearmanr(shuffled_lv, li_filtered[:, ch])[0])
                    for ch in range(n_sensors)
                ]

            # --- Run cluster permutation test ---
            T_obs, clusters, p_values, _ = permutation_cluster_test(
                [z_obs] + [z_null[i] for i in range(n_permutations)],
                n_permutations=n_permutations,
                threshold=None,
                tail=0,
                adjacency=adjacency,
                out_type='mask',
                verbose=True
            )

            # --- Save results ---
            sig_clusters = np.where(p_values < 0.05)[0]
            out_txt = os.path.join(paths['signif_correlation_dir'], f"{substr}_{band}_{ch_type}_significant_clusters.txt")

            with open(out_txt, 'w') as f:
                for i in sig_clusters:
                    cluster_sensors = np.array(selected_cols)[clusters[i]]
                    f.write(f"Cluster {i + 1} (p = {p_values[i]:.3f}):\n")
                    f.write(", ".join(cluster_sensors) + "\n\n")

            print(f"Finished {substr}-{band} ({ch_type}): {len(sig_clusters)} significant clusters")



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
