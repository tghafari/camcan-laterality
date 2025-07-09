"""
Unified script for computing lateralized MEG-sensor correlations with subcortical volumes,
performing cluster-based permutation testing on sensor-level correlations,
and visualizing significant clusters via topographic maps.

Steps:
1. **Setup file paths** based on platform ('mac' or 'bluebear').

## Steps 2 and 3 only need to be run once ##
2. **Extract lateralized power** per frequency band (Delta, Theta, Alpha, Beta) for each sensor pair.
3. **Compute Spearman correlations** between band power and subcortical volume lateralization for specified mappings.
########

4. **Read MEG sensor info** and build a custom adjacency for the selected channel type (mag or grad).
5. **For each band-structure combination**:
   a. Load lateralized power data for sensors.
   b. Compute observed Spearman correlations across subjects.
   c. Build a null distribution by shuffling subject labels and computing Fisher z-transformed correlations.
   d. Compute non-parametric p-values for each sensor.
   e. Identify significant sensors (p < 0.05).
   f. Cluster significant sensors using the adjacency matrix (connected components).
   g. Visualize results: topographic map of correlation values, mask significant sensors, and annotate clusters.

Requirements:
- `li_data`: lateralized index data per sensor (subjects × sensors).
- `lv_vals`: lateralized volume values per subject.
- `info`: MNE Info object with selected channels.
- `adjacency`: sparse adjacency matrix for sensor neighborhoods.

Written by: Tara Ghafari
tara.ghafari@gmail.com
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import mne
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from itertools import product

from mne.channels import find_ch_adjacency
from scipy import sparse
from mne.channels import find_layout


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
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/bands'),
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_nooutliers.csv'),  # outliers are smaller than 10th percentile
        'sub_list': op.join(quinna_dir, 'dataman/data_information/FINAL_sublist-LV-LI-outliers-removed.csv'),
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_signif_correlations_subtraction_nooutlier-psd'), # for alph thal, puta beta, hipp delta
        'all_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nooutlier-psd'),  # for all combinations of bands and substrs
        'significant_cluster_perm_dir' : op.join(jenseno_dir, '/subcortical-structures/resting-state/results/CamCan/Results/Correlation_topomaps/bands/subtraction_nonoise_nooutliers-psd-clusterpermutation'),
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout': op.join(quinna_dir, 'dataman/data_information/sensors_layout_names.csv'),
        'spectra_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_nooutliers_absolute-thresh')
    }
    return paths


def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """Combines spectra and volume lateralization data for one sensor pair."""
    pair_file = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')
    spectrum_df = pd.read_csv(pair_file).rename(columns={'Unnamed: 0': 'subject_ID'})
    working_df = spectrum_df.merge(substr_lat_df, on='subject_ID').dropna()
    freqs = [float(f) for f in spectrum_df.columns[1:]]
    return working_df, freqs


def calculate_band_power(working_df, freqs, band):
    """Computes average power for a given frequency band from the DataFrame."""
    band_freqs = [str(round(f, 1)) for f in freqs if band[0] <= f <= band[1] and str(round(f, 1)) in working_df.columns]
    if not band_freqs:
        raise ValueError(f"No frequencies found for band {band}.")
    return working_df[band_freqs].mean(axis=1)


def extract_all_band_power(paths):
    """Extracts band power values from sensor pairs and saves them per band."""
    sensor_pairs = pd.read_csv(paths['sensor_layout'])[['left_sensors', 'right_sensors']].dropna()
    substr_lat_df = pd.read_csv(paths['LV_csv'])
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 40)}

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
    """Computes Spearman r-values between MEG band power and subcortical LVs and saves:
    1. CSVs with correlation and p-values per sensor pair,
    2. CSV of matched subject_IDs used in the analysis.
    """
        
    bands = ['Delta', 'Theta', 'Alpha', 'Beta']
    structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
    lv_df = pd.read_csv(paths['LV_csv'])
    used_subjects = set()

    for band, substr in product(bands, structures):
        band_file = op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv')
        if not op.exists(band_file):
            print(f"Missing band file: {band_file}")
            continue

        band_df = pd.read_csv(band_file)
        if substr not in lv_df.columns:
            print(f"Missing structure {substr} in LV file.")
            continue

        merged = band_df.merge(lv_df[['subject_ID', substr]], on='subject_ID', how='inner').dropna()
        if merged.empty:
            print(f"No overlapping subjects for {band}-{substr}")
            continue

        lv_vals = merged[substr].values
        used_subjects.update(merged['subject_ID'].tolist())

        rval_list = []
        pval_list = []

        for col in band_df.columns[1:]:  # Skip subject_ID
            r, p = spearmanr(merged[col], lv_vals)
            rval_list.append(r)
            pval_list.append(p)

        out_df = pd.DataFrame({
            'sensor_pair': band_df.columns[1:],
            f'{band.lower()}_rval': rval_list,
            f'{band.lower()}_pval': pval_list
        })
        out_df.to_csv(op.join(paths['all_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv'), index=False)
    
    # Save subject IDs used in the merged data (intersection of LI and LV with no NaNs)
    subject_df = pd.DataFrame(sorted(used_subjects), columns=['subject_ID'])
    subject_df.to_csv(paths['sub_list'], index=False)

def read_raw_info(paths, ch_type):
    """Reads sensor info from a MEG file for adjacency computation.
    raw_mag.info is only for illustration purposes of grads, as planars are combined."""

    raw = mne.io.read_raw_fif(paths['sample_meg_file'], verbose='ERROR')
    layout = pd.read_csv(paths['sensor_layout'])
    right_sensors = layout['right_sensors'].dropna().tolist()

    if ch_type == 'mag':
        channels = [ch for ch in right_sensors if ch.endswith('1')]
        raw.pick('mag').pick(channels)
        
        return raw, raw.info
    
    else:
        channels_mag = [ch for ch in right_sensors if ch.endswith('1')]
        raw_mag = raw.copy().pick('mag').pick(channels_mag)
        channels = [ch for ch in right_sensors if not ch.endswith('1')]
        raw.pick('grad').pick(channels)
        # bear in mind that plotting grads requires a mag info to plot positive and negative values
        # it also requires grad info for adjacency info
        
        return raw, raw.info, raw_mag.info

def find_custom_adjacency(info, ch_type):
    """Returns sparse adjacency matrix and channel names for selected sensor type."""
    full_adj, full_names = find_ch_adjacency(info, ch_type=ch_type)
    mask = [name in info['ch_names'] for name in full_names]
    adjacency = full_adj[mask][:, mask]
    return sparse.csr_matrix(adjacency), [name for name in full_names if name in info['ch_names']]


def run_cluster_test_from_raw_corr(paths, ch_type, n_permutations=1000):
    """Performs cluster-based permutation test on Spearman correlations between sensor lateralized power and subcortical volumes."""
    lv_df = pd.read_csv(paths['LV_csv'])

    # Example mapping; significant plots from before, not used now
    # substrs_bands = [{'Thal': 'Alpha'}, {'Puta': 'Beta'}, {'Hipp': 'Delta'}]
    
    bands = ['Delta', 'Theta', 'Alpha', 'Beta']
    substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
    
    # this line receives three outputs if 'grad' and two if 'mag'. 
    # info_mag for plotting is in rest tuple and later called
    raw, info, *other = read_raw_info(paths, ch_type)  
    adjacency, _ = find_custom_adjacency(info, ch_type)

    for substr, band in product(substrs, bands):
        corr_path = op.join(paths['all_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
        if not op.exists(corr_path):
            print(f"Skipping {substr}-{band}, file not found: {corr_path}")
            continue

        corr_df = pd.read_csv(corr_path)
        
        # Filter sensor pairs based on magnetometer or gradiometer channel suffix
        if ch_type == 'mag':
            sensor_mask = corr_df['sensor_pair'].str.endswith('1')
        elif ch_type == 'grad':
            sensor_mask = corr_df['sensor_pair'].str.endswith('2') | corr_df['sensor_pair'].str.endswith('3')
        else:
            raise ValueError(f"Unsupported ch_type: {ch_type}")

        filtered_df = corr_df[sensor_mask]
        if filtered_df.empty:
            print(f"No matching sensors for {ch_type} in {substr}-{band}")
            continue

        # Apply the mask to select rows
        r_obs = filtered_df[f'{band.lower()}_rval'].to_numpy()
        spearman_p = filtered_df[f'{band.lower()}_pval'].to_numpy()  # for before cluster permutation p-values
        # Fisher z-transform (from r to z)
        z_obs = np.arctanh(r_obs)

        # 2. Build null distribution by shuffling
        # Load lateralized power data for this band to calculate shuffled correlations
        li_df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv'))
        # Select columns matching channels in info, ensuring order matches info['ch_names']
        selected_cols = [c for c in li_df.columns if c.endswith('1')] if ch_type == 'mag' else [c for c in li_df.columns if c.endswith('2') or c.endswith('3')]
        li_data = li_df[selected_cols].to_numpy()
        n_sensors = li_data.shape[1]
        lv_vals = lv_df.set_index('subject_ID').loc[li_df['subject_ID']][substr].values
        rng = np.random.RandomState(42)
        z_null = np.zeros((n_permutations, n_sensors))
        for p in range(n_permutations):
            lv_shuff = rng.permutation(lv_vals)
            r_null = [spearmanr(lv_shuff, li_data[:, i])[0] for i in range(n_sensors)]
            z_null[p, :] = np.arctanh(r_null)

        # 3. Compute non-parametric p-values (two-tailed)
        p_vals = np.mean(np.abs(z_null) >= np.abs(z_obs), axis=0)
        significant_mask = p_vals < 0.05
        significant_mask_spearman = spearman_p < 0.05  # if you want to plot without cluster permutation tests
        print(f"{substr}-{band} ({ch_type}): {significant_mask.sum()} significant sensors")

        # 4. Cluster significant sensors using adjacency
        sig_idx = np.where(significant_mask)[0]
        # sig_idx_spearman = np.where(significant_mask_spearman)[0]
        if sig_idx.size > 0:
            sub_adj = adjacency[np.ix_(sig_idx, sig_idx)]
            n_comp, labels = connected_components(csr_matrix(sub_adj), directed=False, return_labels=True)
            clusters = {i: sig_idx[labels == i] for i in range(n_comp)}
            print(f"Found {len(clusters)} permutated cluster(s) for {substr}-{band} ({ch_type}):")
            for cid, nodes in clusters.items():
                print(f"  Cluster {cid}: indices {nodes}")

            # think about how to sum over the zvalues in each spearman and permutated cluster and then 
            # pick as significant those cluster's whose zvalues were larger than permutated ones.
            # or something like that
            # clusters_spearman = {i: sig_idx_spearman[labels == i] for i in range(n_comp)}            
            # print(f"Found {len(clusters_spearman)} spearman cluster(s) for {substr}-{band} ({ch_type}):")
            # for cid, nodes in clusters_spearman.items():
            #     print(f"  Cluster {cid}: indices {nodes}")
        else:
            print(f"No significant sensors to cluster for {substr}-{band} ({ch_type}).")
            clusters = {}

        # 5. Visualize topomap with significant mask and cluster labels
        fig, ax = plt.subplots()

        # Define the significant clusters
        mask = significant_mask
        mask_params = dict(
            marker='o', markerfacecolor='w', markeredgecolor='k',
            linewidth=1, markersize=10
        ) 

        # Prepare the data for visualisation
        """combines planar gradiometer to use mag info (to plot positive and negative values)"""
        if ch_type == 'grad':
            r_obs = (r_obs[::2] + r_obs[1::2]) / 2  
            mask = mask[::2]
            del info  # refresh info for plotting only in grad
            info = other[0]  # this is from read_raw_info when running with ch_type='grad'

        # Plot topomap
        im, cn = mne.viz.plot_topomap(
            r_obs, info, mask=mask, mask_params=mask_params,
            vlim=(min(r_obs), max(r_obs)), contours=0, image_interp='nearest', 
            cmap='RdBu_r', show=False, axes=ax
        )  
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Correlation Values', fontsize=14)
        ax.set_xlim(0, )  # remove the left half of topoplot
        ax.set_title(f'{substr}-{band} Spearman r ({ch_type})')
        plt.show()

def main():
    platform = 'bluebear'  # or 'bluebear'
    paths = setup_paths(platform)
    # extract_all_band_power(paths)  # only need to run once
    # save_spearman_correlations(paths)   # only need to run once
    # run_cluster_test_from_raw_corr(paths, ch_type='mag')
    run_cluster_test_from_raw_corr(paths, ch_type='grad')

if __name__ == "__main__":
    main()
