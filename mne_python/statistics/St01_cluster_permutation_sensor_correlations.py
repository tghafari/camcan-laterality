"""
========================================================
St01_cluster_permutation_sensor_correlations

This script performs the following:

1. Organizes significant Spearman correlation (r) values between lateralized MEG band power and lateralized subcortical volumes into a unified CSV per band-structure pair.
2. Calculates cluster-based permutation tests for significance across MEG sensor pairs.

Author: Tara Ghafari
Email: tara.ghafari@gmail.com
Date: 16/05/2025
========================================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.channels import find_ch_adjacency
from mne.stats import permutation_cluster_test
from scipy.stats import spearmanr


def setup_paths(platform='mac'):
    """Set up and return file paths based on the system platform."""
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
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index'),
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_nooutliers.csv'),
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_signif_correlations_subtraction_nooutlier-psd'),
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout': op.join(quinna_dir, 'dataman/data_information/sensors_layout_names.csv'),
        'spectra_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_nooutliers_absolute-thresh')
    }
    return paths


def organise_csvs():
    """Organizes correlation r-values per structure-band into separate CSV files.
    You only need to run this once."""
    paths = setup_paths()
    os.makedirs(paths['signif_correlation_dir'], exist_ok=True)

    substrs_bands = [{'Thal': 'Alpha'}, {'Puta': 'Beta'}, {'Hipp': 'Delta'}]

    for pair in substrs_bands:
        for substr, band in pair.items():
            pair_names, band_rvals = [], []

            for pair_folder in os.listdir(paths['correlation_dir']):
                pair_path = os.path.join(paths['correlation_dir'], pair_folder)
                csv_path = os.path.join(pair_path, substr, f'{substr}_lat_spectra_substr_spearmanr.csv')

                if os.path.isfile(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        band_r = df.loc[df.iloc[:, 0] == band, '0'].values[0]
                        pair_names.append(pair_folder)
                        band_rvals.append(band_r)
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

            output_df = pd.DataFrame({
                'sensor_pair': pair_names,
                f'{band}_rval': band_rvals
            })
            output_csv = os.path.join(paths['signif_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
            output_df.to_csv(output_csv, index=False)
            print(f"Saved r-values to: {output_csv}")


def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """Combines spectra and volume lateralization data for one sensor pair."""
    pair_file = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')
    spectrum_df = pd.read_csv(pair_file).rename(columns={'Unnamed: 0': 'subject_ID'})
    merged_df = spectrum_df.merge(substr_lat_df, on='subject_ID').dropna()
    freqs = [float(f) for f in spectrum_df.columns[1:]]
    return merged_df, freqs


def calculate_band_power(working_df, freqs, band):
    """Computes average power for a given frequency band."""
    band_freqs = [str(round(f, 1)) for f in freqs if band[0] <= f <= band[1] and str(round(f, 1)) in working_df.columns]
    if not band_freqs:
        raise ValueError(f"No frequencies found for band {band}.")
    return working_df[band_freqs].mean(axis=1)


def extract_all_band_power(paths):
    """Extracts band power values from sensor pairs and saves them per band."""
    sensor_pairs = pd.read_csv(paths['sensor_layout'])[['left_sensors', 'right_sensors']].dropna()
    substr_lat_df = pd.read_csv(paths['LV_csv'])
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 13), 'Beta': (13, 30)}

    band_li_dict = {band: [] for band in bands}
    label_dict = {band: [] for band in bands}
    subject_ids = None

    for _, row in sensor_pairs.iterrows():
        left_sensor, right_sensor = row['left_sensors'], row['right_sensors']
        working_df, freqs = working_df_maker(paths['spectra_dir'], left_sensor, right_sensor, substr_lat_df)
        if subject_ids is None:
            subject_ids = working_df['subject_ID']
        for band, frange in bands.items():
            band_power = calculate_band_power(working_df, freqs, frange)
            band_li_dict[band].append(band_power.values)
            label_dict[band].append(f'{left_sensor}_{right_sensor}')

    for band, data in band_li_dict.items():
        matrix = np.stack(data, axis=1)
        df_out = pd.DataFrame(matrix, columns=label_dict[band])
        df_out.insert(0, 'subject_ID', subject_ids.values)
        df_out.to_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv'), index=False)


def save_spearman_correlations(paths):
    """Computes Spearman r-values between MEG band power and subcortical LVs."""
    band_substr_map = {'Alpha': 'Thal', 'Beta': 'Puta', 'Delta': 'Hipp'}
    lv_df = pd.read_csv(paths['LV_csv'])

    for band, substr in band_substr_map.items():
        band_df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv'))
        lv_vals = lv_df.set_index('subject_ID').loc[band_df['subject_ID']][substr].values

        rval_list = [spearmanr(band_df[col], lv_vals)[0] for col in band_df.columns[1:]]
        pd.DataFrame({'sensor_pair': band_df.columns[1:], f'{band.lower()}_rval': rval_list}).to_csv(
            op.join(paths['signif_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv'), index=False
        )


def read_info(paths, ch_type='mag'):
    """Reads sensor info from a MEG file for adjacency computation."""
    raw = mne.io.read_raw_fif(paths['sample_meg_file'], verbose='ERROR')
    layout = pd.read_csv(paths['sensor_layout'])
    right_sensors = layout['right_sensors'].dropna().tolist()

    if ch_type == 'mag':
        channels = [ch for ch in right_sensors if ch.endswith('1')]
        raw.pick('mag').pick(channels)
    else:
        channels = [ch for ch in right_sensors if not ch.endswith('1')]
        raw.pick('grad').pick(channels)

    return raw.info


def find_custom_adjacency(info, ch_type):
    """Returns adjacency matrix for selected sensor type and layout."""
    full_adj, full_names = find_ch_adjacency(info, ch_type=ch_type)
    mask = [name in info['ch_names'] for name in full_names]
    return full_adj[mask][:, mask], [name for name in full_names if name in info['ch_names']]


def run_cluster_test_from_raw_corr(paths, ch_type='mag'):
    """Runs cluster-based permutation tests for correlations between LI and LVs."""
    lv_df = pd.read_csv(paths['LV_csv'])
    substrs_bands = [{'Thal': 'Alpha'}, {'Puta': 'Beta'}, {'Hipp': 'Delta'}]

    info = read_info(paths, ch_type=ch_type)
    adjacency, _ = find_custom_adjacency(info, ch_type)

    for pair in substrs_bands:
        for substr, band in pair.items():
            li_df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv'))
            selected_cols = [c for c in li_df.columns if c.endswith('1')] if ch_type == 'mag' else [c for c in li_df.columns if not c.endswith('1')]
            li_data = li_df[selected_cols].to_numpy()
            lv_vals = lv_df.set_index('subject_ID').loc[li_df['subject_ID']][substr].values

            r_obs = np.array([spearmanr(lv_vals, li_data[:, i])[0] for i in range(li_data.shape[1])])
            z_obs = np.arctanh(r_obs)  # Fisher transform (Fisher's transformation = 0.5 * ln((1 + r) / (1 - r)) , Standard Error = 1 / sqrt(n - 3), z = (z - 0) / (Standard Error))
            # The arctanh function, also known as the inverse hyperbolic tangent, is a statistical transformation often used in conjunction with Fisher information to address the issue of non-normality in correlation coefficients
            
            rng = np.random.RandomState(42)
            z_null = np.array([
                np.arctanh([spearmanr(rng.permutation(lv_vals), li_data[:, i])[0] for i in range(li_data.shape[1])])
                for _ in range(1000)
            ])

            z_data = np.vstack([z_obs] + list(z_null))  # shape: (n_samples, n_sensors)
            assert z_data.shape[1] == adjacency.shape[0], "Mismatch between tests and adjacency size"

            X = [z_data[i, :][np.newaxis, :] for i in range(z_data.shape[0])]  # shape (1, 51) each

            T_obs, clusters, p_vals, _ = permutation_cluster_test(
                X,
                n_permutations=1000,
                tail=0,
                threshold=None,
                adjacency=adjacency,
                out_type='mask',
                verbose=True
            )

            sig_idx = np.where(p_vals < 0.05)[0]
            out_txt = op.join(paths['signif_correlation_dir'], f'{substr}_{band}_{ch_type}_significant_clusters.txt')

            with open(out_txt, 'w') as f:
                for i in sig_idx:
                    sig_sensors = np.array(selected_cols)[clusters[i]]
                    f.write(f"Cluster {i+1} (p={p_vals[i]:.3f}):\n")
                    f.write(", ".join(sig_sensors) + "\n\n")

            print(f"Completed {substr}-{band} ({ch_type}): {len(sig_idx)} significant clusters")


def main():
    paths = setup_paths()
    extract_all_band_power(paths)
    save_spearman_correlations(paths)
    run_cluster_test_from_raw_corr(paths, ch_type='mag')
    run_cluster_test_from_raw_corr(paths, ch_type='grad')


if __name__ == '__main__':
    main()
