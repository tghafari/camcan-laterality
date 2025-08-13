
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
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_no-vol-outliers.csv'),  # vol outliers (< 0.01 quantile) removed
        'sub_list': op.join(quinna_dir, 'dataman/data_information/dblCheck_last_FINAL_sublist-vol-outliers-removed.csv'),  # this is just to ensure correct subjects are being used for the final analysis (last_FINAL_sublist-vol-outliers-removed.csv)
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_signif_correlations_subtraction_nooutlier-psd'), # for alph thal, puta beta, hipp delta
        'all_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nonoise_no-vol-outliers'),  # for all combinations of bands and substrs only excludeing vol outliers
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout': op.join(quinna_dir, 'dataman/data_information/combined_sensors_layout_names.csv'),  # combined grads end in '2', there is no sensor ending in '3'
        'spectra_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_no-vol-outliers_combnd-grads')  # we use vol outliers removed list now (30/07/2025)
    }
    return paths

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

platform = 'mac'  # or 'bluebear'
paths = setup_paths(platform)
substr = input('Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu):').strip()
band = input('Enter band (Delta, Theta, Alpha, Beta):').strip()
ch_type = input('Enter sensortype (mag or grad):').strip()

############################## LOAD AND PREPARE DATA ##############################
lv_df = pd.read_csv(paths['LV_csv'])

# this line receives three outputs if 'grad' and two if 'mag'. 
# info_mag for plotting is in rest tuple and later called
raw, info, *other = read_raw_info(paths, ch_type)  
adjacency, names = find_custom_adjacency(info, ch_type)

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

if band in significant_clusters_t and substr in significant_clusters_t[band] and significant_clusters_t[band][substr]:
    # Flatten all sensor indices in significant clusters for this band and substr
    all_sig_indices = []
    for cluster_id, indices in significant_clusters_t[band][substr].items():
        all_sig_indices.extend(indices)

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