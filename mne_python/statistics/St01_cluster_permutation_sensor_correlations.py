"""
===========================================================
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
==============================================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd

import scipy.stats as stats
from scipy.stats import spearmanr
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
from itertools import product

import mne
from mne.channels import find_ch_adjacency
from mne.channels import find_layout


def setup_paths(platform='mac'):
    """Set up and return file paths based on the system platform."""
    if platform == 'bluebear':
        quinna_dir = '/rds/projects/q/quinna-camcan-1/'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan-1/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/bands'),
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_no-vol-outliers.csv'),  # vol outliers (< 0.01 quantile) removed
        'sub_list': op.join(quinna_dir, 'dataman/data_information/dblCheck_last_FINAL_sublist-vol-outliers-removed.csv'),  # this is just to ensure correct subjects are being used for the final analysis (last_FINAL_sublist-vol-outliers-removed.csv)
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_signif_correlations_subtraction_nooutlier-psd'), # for alph thal, puta beta, hipp delta
        'all_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nonoise_no-vol-outliers'),  # for all combinations of bands and substrs only excludeing vol outliers
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout': op.join(quinna_dir, 'dataman/data_information/combined_sensors_layout_names.csv'),  # combined grads end in '2', there is no sensor ending in '3'
        'spectra_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_no-vol-outliers_combnd-grads'),  # we use vol outliers removed list now (30/07/2025)
        'cluster_perm_signif_sensors': op.join(sub2ctx_dir,'derivatives/correlations/bands/bands_significant_sensors_cluster-perm'),
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
        df_out.to_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv'), index=False)


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
        band_file = op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv')
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

def r_to_t(r_vals, n):
    """Convert Spearman r-values to t-values."""
    r_vals = np.asarray(r_vals)
    return r_vals * np.sqrt((n - 2) / (1 - r_vals**2))

def read_raw_info(paths, ch_type):
    """Reads sensor info from a MEG file for adjacency computation.
    raw_mag.info is only for illustration purposes of grads, as planars are combined."""

    raw = mne.io.read_raw_fif(paths['sample_meg_file'])
    layout = pd.read_csv(paths['sensor_layout'])
    right_sensors = layout['right_sensors'].dropna().tolist()

    if ch_type == 'mag':
        channels = [ch for ch in right_sensors if ch.endswith('1')]
        raw.pick('mag').pick(channels)
        
        return raw, raw.info
    
    else:
        channels_mag = [ch for ch in right_sensors if ch.endswith('1')]
        raw_mag = raw.copy().pick('mag').pick(channels_mag)
        channels = [ch for ch in right_sensors if ch.endswith('2')]
        raw.pick('grad').pick(channels)
        # bear in mind that plotting grads requires a mag info to plot positive and negative values
        # it also requires grad info for adjacency info
        
        return raw, raw.info, raw_mag.info

def find_custom_adjacency(info, ch_type, plot_all=False):
    """Returns sparse adjacency matrix and channel names for selected sensor type."""
    full_adj, full_names = find_ch_adjacency(info, ch_type=ch_type)
    mask = [name in info['ch_names'] for name in full_names]
    adjacency = full_adj[mask][:, mask]
    names = [name for name in full_names if name in info['ch_names']]
    adjacency = sparse.csr_matrix(adjacency)

    if plot_all:
        pos = np.array([info['chs'][info['ch_names'].index(name)]['loc'][:2] for name in names])

        for i, name in enumerate(names):
            connected = adjacency[i].toarray().flatten()
            neighbor_idx = np.where(connected)[0]

            fig, ax = plt.subplots()
            ax.set_title(f"Sensor: {name} and its neighbors")
            ax.scatter(pos[:, 0], pos[:, 1], color='lightgray', label='All Sensors')
            ax.scatter(pos[i, 0], pos[i, 1], color='red', label='Current Sensor', zorder=3)
            ax.scatter(pos[neighbor_idx, 0], pos[neighbor_idx, 1], color='blue', label='Neighbors', zorder=2)

            for idx in neighbor_idx:
                ax.plot([pos[i, 0], pos[idx, 0]], [pos[i, 1], pos[idx, 1]], color='blue', alpha=0.5)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.legend()
            plt.axis('equal')
            plt.tight_layout()
            plt.show()

    return adjacency, names

def run_cluster_test_from_raw_corr(paths, substr, band, ch_type, n_permutations=1000, plot_adj=False, draw_cluster_lines=False, plot_topo_nulls=False):
    """Performs cluster-based permutation test on Spearman correlations between sensor 
    lateralised power and subcortical volumes."""
   
    ############################## LOAD AND PREPARE DATA ##############################
    lv_df = pd.read_csv(paths['LV_csv'])
    
    # this line receives three outputs if 'grad' and two if 'mag'. 
    # info_mag for plotting is in rest tuple and later called
    raw, info, *other = read_raw_info(paths, ch_type)  
    adjacency, names = find_custom_adjacency(info, ch_type)

    if plot_adj:
        # if you want to take a look at the adjacency matrix
        adj_dense = adjacency.toarray()

        plt.figure(figsize=(8, 8))
        plt.imshow(adj_dense, cmap='Greys', interpolation='none')
        plt.title('Sensor Adjacency Matrix')
        plt.xlabel('Sensor Index')
        plt.ylabel('Sensor Index')
        plt.colorbar(label='Connected (1) or Not (0)')
        plt.show()

    corr_path = op.join(paths['all_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
    if not op.exists(corr_path):
        print(f"Skipping {substr}-{band}, file not found: {corr_path}")
        return

    corr_df = pd.read_csv(corr_path)
    
    # Filter sensor pairs based on magnetometer or gradiometer channel suffix
    if ch_type == 'mag':
        sensor_mask = corr_df['sensor_pair'].str.endswith('1')
        central_sensors = ['MEG0811', 'MEG1011','MEG2121']  # should be ignored in cluster testing
        central_sensor_indices = [11, 12, 30]
    elif ch_type == 'grad':
        sensor_mask = corr_df['sensor_pair'].str.endswith('2')  # grads are combined and saved in sensors ending in '2', no sensors ending in '3'
        central_sensors = ['MEG0812', 'MEG1012', 'MEG2122']  # should be ignored in cluster testing
        central_sensor_indices = [48, 49, 50]
    else:
        raise ValueError(f"Unsupported ch_type: {ch_type}")

    filtered_df = corr_df[sensor_mask]
    if filtered_df.empty:
        print(f"No matching sensors for {ch_type} in {substr}-{band}")
        return

    # Load lateralized power data for this band to calculate shuffled correlations and use variables for plotting
    li_df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv'))
    r_obs = filtered_df[f'{band.lower()}_rval'].to_numpy()    

    ############################## OBSERVED CLUSTERS ##############################

    # Compute significancy threshold from t distribution 
    # Here we use a two-tailed test, hence we need to divide alpha by 2.
    alpha = 0.05  # we basically are using one tailed for either positve or negative (=alpha = 0.05)
    n_subjects = len(li_df)  # in our last_FINAl list this is 532 (vol outliers and errored subjects in lateralised psd calculation are removed)
    df = n_subjects - 1  # degrees of freedom
    t_thresh = stats.distributions.t.ppf(1 - alpha / 2, df=df)
    t_obs = r_to_t(r_obs, n=n_subjects)  # indices here correspond to r_obs and not adjacency
    # remember we only use adjacency indices for csr analyses, everything else uses info/filtered_df indices

    significant_obs = (np.array(t_obs) > abs(t_thresh)) | (np.array(t_obs) < -abs(t_thresh))
    sig_obs = np.where(significant_obs)[0]  

    # Extract (channel_name, index) pairs from correct order
    sig_pairs = filtered_df.loc[significant_obs, 'sensor_pair'].tolist()
    chan_name_and_idx = [
        (pair.split('_')[1], idx)
        for pair, idx in zip(sig_pairs, sig_obs)
    ]

    # Filter out central sensors safely, preserving alignment
    chan_name_and_idx = [
        (ch, idx)
        for ch, idx in chan_name_and_idx
        if ch not in central_sensors and idx not in central_sensor_indices
    ]

    # Unpack - this is the final names and idx after removing central sensors
    sig_chan_names = [ch for ch, _ in chan_name_and_idx]
    sig_obs = np.array([idx for _, idx in chan_name_and_idx])

    if sig_obs.size <= 0:
        print("No significant r values found in sensors")
        mask = np.zeros(len(info['ch_names']), dtype=bool)  # no masking on the topoplot

    elif sig_obs.size > 0:
        # significant masks on the topoplot
        mask = np.zeros(len(info['ch_names']), dtype=bool)
        mask[sig_obs] = True  # these indices are the same as in significant_clusters_t

        index_in_adjacency = np.array([names.index(ch) for ch in sig_chan_names])  # info.ch_names is not in the same order as names
        sub_adj_obs = adjacency[np.ix_(index_in_adjacency, index_in_adjacency)]
        n_comp_obs, labels_obs = connected_components(
            csr_matrix(sub_adj_obs), directed=False, return_labels=True)
        
        clusters_obs = {i: index_in_adjacency[labels_obs == i] for i in range(n_comp_obs)}  # for the adjacency indices
        # Build a map from channel name to index in sig_null (i.e., index into filtered_df/t_null)
        chan_to_sig_idx_obs = {
            ch: sig_obs[i]
                for i, ch in enumerate(sig_chan_names)
            }  # removing the central sensors might misalign the indices

        # Map clusters of adjacency indices back to filtered_df indices
        clusters_rt_obs = {  # same indices as significant_clusters_t
            i: np.array([
                chan_to_sig_idx_obs[names[idx]]
                for idx in index_in_adjacency[np.where(labels_obs == i)[0]]
                if names[idx] in chan_to_sig_idx_obs
            ])
            for i in range(n_comp_obs)
        }  # for the r_obs and t_obs indices (filtered_df)

        print(f"Found {len(clusters_obs)} permutated cluster(s) for {substr}-{band} ({ch_type}):")
        for cid, nodes in clusters_obs.items():
            print(f"  Cluster {cid}: indices {nodes} -> channels: {[names[ch] for ch in nodes]}")  # these indices are different from those in significant_clusters_t
                                                                                                   # but they refer to the same channels

        cluster_t_rt_sums_obs = {label: np.sum(t_obs[sensors]) for label, sensors in clusters_rt_obs.items()}
        for cluster_id_obs, t_sum_obs in cluster_t_rt_sums_obs.items():
            print(f"  Cluster {cluster_id_obs}: summed t = {t_sum_obs:.3f}")

    ############################## TOPOPLOT OBSERVED CLUSTERS ##############################        
    # Compute t-transformed values and use that to find significant clusters
    print("Plotting observed significant sensors")
    
    # Prepare the data for visualisation
    """we use the positions of channels for grads.
    using mag info messes up the index of channels and is therefore not correct."""
    if ch_type == 'grad':  
        info_mag = other[0]
        # trying out different methods to plot grads in the correct position - looks ok just needs to align to right sensor locations
        grad_picks = mne.pick_types(info, meg='grad')
        grad_names = [info['ch_names'][p] for p in grad_picks]  # this is essentialy identical to 
                                                                #sensor_names = [pair.split('_')[1] for pair in filtered_df['sensor_pair']]
                                                                # we are just using 'info' to be more principled so:
                                                                # In [198]: grad_names == sensor_names
                                                                # Out[198]: True

        pos2d = np.vstack([info['chs'][p]['loc'][:2] for p in grad_picks])
        pos_info = pos2d
    else:
        pos_info = info

    # Define the significant clusters
    mask_params = dict(
        marker='o', markerfacecolor='w', markeredgecolor='k',
        linewidth=1, markersize=10
    ) 

    fig, ax = plt.subplots()

    im, cn = mne.viz.plot_topomap(
        t_obs, pos_info, mask=mask, mask_params=mask_params, #names=grad_names,  
        vlim=(min(t_obs), max(t_obs)), contours=0, image_interp='nearest', 
        cmap='RdBu_r', show=False, axes=ax
    )  

    if draw_cluster_lines:  # this section plots lines but not at the exact location - TODO: doesn't work yet
        # --- Draw cluster-specific lines ---
        # Get 2D positions of all sensors
        pos = np.array([info['chs'][info['ch_names'].index(name)]['loc'][:2] for name in info['ch_names']])

        # Loop over each cluster of sig_obs indices
        for cluster_idx, sig_indices in clusters_rt_obs.items():
            if len(sig_indices) < 2:
                continue  # Skip single-sensor clusters

            # Get 2D coordinates of sensors in this cluster
            cluster_pos = pos[sig_indices]

            # Draw lines between all pairs in the cluster
            for i in range(len(cluster_pos)):
                for j in range(i + 1, len(cluster_pos)):
                    ax.plot(
                        [cluster_pos[i][0], cluster_pos[j][0]],
                        [cluster_pos[i][1], cluster_pos[j][1]],
                        color='black', linewidth=1, alpha=0.8
                    )

    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation Values', fontsize=14)
    ax.set_xlim(0, )  # remove the left half of topoplot
    ax.set_title(f'{substr}-{band} t transformed Spearman r ({ch_type})- before cluster testing')
    plt.show()

    ############################## PERMUTATION TESTING ##############################
    # Select columns matching channels in info, ensuring order matches info['ch_names']
    selected_cols = [c for c in li_df.columns if c.endswith('1')] if ch_type == 'mag' else [c for c in li_df.columns if c.endswith('2') or c.endswith('3')]
    li_data = li_df[selected_cols].to_numpy()
    n_sensors = li_data.shape[1]
    lv_vals = lv_df.set_index('subject_ID').loc[li_df['subject_ID']][substr].values

    if sig_obs.size > 0:
        # 2. Build null distribution by shuffling
        rng = np.random.RandomState(42)
        max_t_sums_pos = np.zeros(n_permutations)  # to collect max positive cluster sum
        min_t_sums_neg = np.zeros(n_permutations)  # to collect min negative cluster sum

        print('Running permutation testing for t values...')
        for p in range(n_permutations):
            lv_shuff = rng.permutation(lv_vals)

            # compute correlation per sensor
            results = [spearmanr(lv_shuff, li_data[:, i]) for i in range(n_sensors)]

            r_null = np.array([res.correlation for res in results])
            t_null = r_to_t(r_null, n=n_subjects)

            significant_nulls = (np.array(t_null) > abs(t_thresh)) | (np.array(t_null) < -abs(t_thresh))
            sig_null = np.where(significant_nulls)[0]
            sig_pairs_null = filtered_df.loc[significant_nulls, 'sensor_pair'].tolist()
            chan_name_and_idx = [
                (pair.split('_')[1], idx)
                for pair, idx in zip(sig_pairs_null, sig_null)
            ]
            chan_name_and_idx = [
                (ch, idx)
                for ch, idx in chan_name_and_idx
                if ch not in central_sensors and idx not in central_sensor_indices
            ]
            sig_chan_names_null = [ch for ch, _ in chan_name_and_idx]
            sig_null = np.array([idx for _, idx in chan_name_and_idx])

            if sig_null.size > 0:
                index_in_adjacency_null = np.array([names.index(ch) for ch in sig_chan_names_null])
                sub_adj_null = adjacency[np.ix_(index_in_adjacency_null, index_in_adjacency_null)]
                n_comp_null, labels_null = connected_components(
                    csr_matrix(sub_adj_null), directed=False, return_labels=True)
                clusters_null = {i: index_in_adjacency_null[labels_null == i] for i in range(n_comp_null)}  
                chan_to_sig_idx_null = {
                    ch: sig_null[i]
                    for i, ch in enumerate(sig_chan_names_null)
                }
                clusters_rt_null = {
                    i: np.array([
                        chan_to_sig_idx_null[names[idx]]
                        for idx in index_in_adjacency_null[np.where(labels_null == i)[0]]
                        if names[idx] in chan_to_sig_idx_null  # safety check
                    ])
                    for i in range(n_comp_null)
                }

                if plot_topo_nulls:
                    # Compute t-transformed values and use that to find significant clusters
                    print("Plotting null significant sensors")

                    # Plot significant sensors before cluster permutations
                    fig, ax = plt.subplots()
                    del mask  # to prevent contamination from observed masks
                    mask = np.zeros(len(info['ch_names']), dtype=bool)
                    mask[sig_null] = True

                    # Plot topomap for null distribution
                    im, cn = mne.viz.plot_topomap(
                        t_null, pos_info, mask=mask, mask_params=mask_params, names=grad_names,
                        vlim=(min(t_null), max(t_null)), contours=0, image_interp='nearest', 
                        cmap='RdBu_r', show=False, axes=ax
                    )  
                    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
                    cbar.ax.tick_params(labelsize=10)
                    cbar.set_label('Correlation Values', fontsize=14)
                    ax.set_xlim(0, )  # remove the left half of topoplot
                    ax.set_title(f'{substr}-{band} t transformed Spearman r ({ch_type})- null distribution')
                    plt.show()

                cluster_t_sums_null = [np.sum(t_null[sensors]) for sensors in clusters_rt_null.values()]
            
                # For this permutation, store:
                pos_sums = [s for s in cluster_t_sums_null if s > 0]
                neg_sums = [s for s in cluster_t_sums_null if s < 0]

                max_t_sums_pos[p] = np.max(pos_sums) if pos_sums else 0
                min_t_sums_neg[p] = np.min(neg_sums) if neg_sums else 0

            else:
                max_t_sums_pos[p] = 0
                min_t_sums_neg[p] = 0

        ############################## EVALUATE CLUSTER SIGNIFICANCE ##############################
        # Compute two-tailed thresholds
        upper_threshold = np.percentile(max_t_sums_pos, 95)
        print(f'upper: {upper_threshold}')
        lower_threshold = np.percentile(min_t_sums_neg, 5)
        print(f'lower: {lower_threshold}')

        plt.figure(figsize=(10, 6))
        plt.hist(max_t_sums_pos, bins=30, color='salmon', edgecolor='black', alpha=0.6, label='Max positive cluster sums')
        plt.hist(min_t_sums_neg, bins=30, color='lightblue', edgecolor='black', alpha=0.6, label='Min negative cluster sums')
        plt.axvline(lower_threshold, color='lightblue', linestyle='--', label='5th percentile (neg)')
        plt.axvline(upper_threshold, color='salmon', linestyle='--', label='95th percentile (pos)')
    
        # Dictionary to store significant clusters
        significant_clusters = {}

        # Plot observed cluster t sums
        for cluster_id_obs, t_sum_obs in cluster_t_rt_sums_obs.items():
            if t_sum_obs > upper_threshold:
                color = 'green'
                label = f'Cluster {cluster_id_obs}: {t_sum_obs:.2f}'
            elif t_sum_obs < lower_threshold:
                color = 'green'
                label = f'Cluster {cluster_id_obs}: {t_sum_obs:.2f}'
            else:
                color = 'yellow'
                label = f'Cluster {cluster_id_obs}: {t_sum_obs:.2f} (ns)'

            plt.axvline(t_sum_obs, color=color, linestyle='-', label=label)

            if color == 'green':
                if band not in significant_clusters:
                    significant_clusters[band] = {}
                if substr not in significant_clusters[band]:
                    significant_clusters[band][substr] = {}
                significant_clusters[band][substr][f'cluster_{cluster_id_obs}'] = clusters_obs[cluster_id_obs].tolist()

        plt.title(f'Null Distribution of Max/Min t-sums ({substr}-{band}, {ch_type})')
        plt.xlabel('Summed t-values in cluster')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.show()

        ############################## FINAL TOPOPLOT FOR SIGNIFICANT CLUSTERS ##############################
        # Dictionary to store significant clusters based on t-values
        significant_clusters_t = {}

        for cluster_id_obs, t_sum_obs in cluster_t_rt_sums_obs.items():
            if t_sum_obs > upper_threshold or t_sum_obs < lower_threshold:
                # Save significant cluster sensors
                if band not in significant_clusters_t:
                    significant_clusters_t[band] = {}
                if substr not in significant_clusters_t[band]:
                    significant_clusters_t[band][substr] = {}
                significant_clusters_t[band][substr][f'cluster_{cluster_id_obs}'] = clusters_rt_obs[cluster_id_obs].tolist()
                print(f"[t] Significant cluster {cluster_id_obs} for {substr}-{band}: "
                f"sum(t) = {t_sum_obs:.2f}, sensors = {clusters_rt_obs[cluster_id_obs].tolist()}")
        
        if band in significant_clusters_t and substr in significant_clusters_t[band] and significant_clusters_t[band][substr]:
            # Flatten all sensor indices in significant clusters for this band and substr
            all_sig_indices = []
            for cluster_id, indices in significant_clusters_t[band][substr].items():
                all_sig_indices.extend(indices)

            # Save significant_clusters_t[band][substr] into file:
            idx_to_chan_name = {idx: name for name, idx in chan_to_sig_idx_obs.items()}

            # Flatten and prepare the data for saving
            rows = []
            for band_name, substr_dict in significant_clusters_t.items():
                for substr_name, clusters in substr_dict.items():
                    for cluster_id, sensor_indices in clusters.items():
                        # Get corresponding sensor names
                        sensor_names = [idx_to_chan_name.get(idx, f"Unknown_{idx}") for idx in sensor_indices]
                        rows.append({
                            'band': band_name,
                            'structure': substr_name,
                            'cluster': cluster_id,
                            'sensors': ';'.join(map(str, sensor_indices)),       # indices
                            'sensor_names': ';'.join(sensor_names)               # names
                        })

            # Convert to DataFrame
            df = pd.DataFrame(rows)

            # Save to CSV
            df.to_csv(op.join(paths['cluster_perm_signif_sensors'], f'{substr}_{band}_{ch_type}_signif_sensors_after_cluster_perm.csv'), index=False) 

            # 5. Visualize topomap with significant mask and cluster labels
            fig, ax = plt.subplots()

            # Define the significant clusters
            del mask  # to prevent contamination from observed masks
            mask = np.zeros(len(info['ch_names']), dtype=bool)
            mask[all_sig_indices] = True

            # Plot topomap
            im, cn = mne.viz.plot_topomap(
                t_obs, pos_info, mask=mask, mask_params=mask_params,
                vlim=(min(t_obs), max(t_obs)), contours=0, image_interp='nearest', 
                cmap='RdBu_r', show=False, axes=ax
            )  
            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Correlation Values', fontsize=14)
            ax.set_xlim(0, )  # remove the left half of topoplot
            ax.set_title(f'{substr}-{band} t transformed Spearman r ({ch_type})-after cluster permutation')
            plt.show()

    else:
        print(f"no significant clusters for {substr}-{band} ({ch_type})")

def cluster_permutation():
    platform = 'mac'  # or 'bluebear'
    paths = setup_paths(platform)
    # extract_all_band_power(paths)  # only need to run once
    # save_spearman_correlations(paths)   # only need to run once
    # substr = input('Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu):').strip()
    # band = input('Enter band (Delta, Theta, Alpha, Beta):').strip()
    # ch_type = input('Enter sensortype (mag or grad):').strip()
    # # run_cluster_test_from_raw_corr(paths, substr, band, ch_type, n_permutations=1000)

    # or run on all
    substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
    bands = ['Delta', 'Theta', 'Alpha', 'Beta']
    ch_types = ['grad', 'mag']
    for substr in substrs:
        for band in bands:
            for ch_type in ch_types:
                run_cluster_test_from_raw_corr(paths, substr, band, ch_type, n_permutations=1000)

if __name__ == "__main__":
    cluster_permutation()
