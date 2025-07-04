"""
====================================
P01_plotting_sensor_srource_power_lat_corr

This code is run to double check all the 
steps we took before by plotting:
    1) sensor band power:
        reads subjects MEG file and plots a 
        topoplot for mags and grads separately
        files are in quinna-camcan
    2) sensor lateralised band power:
        reads lateralised band power and
        plots a topo plot
        files are in sub2ctx/meg/lat/bands
    3) source power band:
        reads grid_stc for all freqs in a
        band
        averages over those values to generate
        band power
        plots the band power on a Volume
        Estimate
        files are in sub2ctx/camcan/derivatives/meg/source/freesurfer/sub-CC110182/grid_perHz
    4) source lateralised band power:
        reads lateralised stc for band power
        plots the lateralised band power on a 
        Volume Estimate
        files are in jenseno-sub2ctx/camcan/derivatives/meg/source/freesurfer/all_subs/all_subs_lateralised_src_power_grad_{band}_band_avg.csv
    5) plots corelation of sensor lat power
        and volume lat on a topoplot for a given substr 
        files are in 
        jenseno-sub2ctx/camcan/derivatives/correlations/bands/bands_all_correlations_subtraction_nooutlier-psd
    6) plots correlation of source lat
        power and volume lat on a Volume Estimate for 
        a given substr
        files are in
        /jenseno-sub2ctx/camcan/derivatives/correlations/src_lat_grid_vol_correlation_nooutliers/spearman-pval_src_lat_power_vol_grad_{band}.csv
        


    for 1 to 4, the code inputs subject id and
    band and plots them all in a 2,2 subplot
    fir 5 and 6 the code inputs substr and
    separately plotts them in a 1,2 subplot
====================================
"""


import os
import os.path as op
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from mne.datasets import fetch_fsaverage

#===========================================
# P01_plotting_sensor_srource_power_lat_corr
#===========================================

# 2) Sensor lateralised band power
# 3) Source band power
# 4) Source lateralised band power
# 5) Sensor-lat vs volume-lat correlation topomap
# 6) Source-lat vs volume-lat correlation Volume Estimate
#===========================================

def setup_paths(sensortype, subjectID, platform='mac'):

    if platform == 'bluebear':
        sub2ctx = '/rds/projects/j/jenseno-sub2ctx/camcan'
        quinna = '/rds/projects/q/quinna-camcan'
    else:
        sub2ctx = '/Volumes/jenseno-sub2ctx/camcan'
        quinna = '/Volumes/quinna-camcan'
    
    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(op.join(sub2ctx, 'derivatives/meg/source/freesurfer'), fs_sub[:-4])
    
    paths = {
        'sample_meg_file': op.join(quinna, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'epoched_dir': op.join(sub2ctx, 'derivatives/meg/sensor/epoched-7min50'),
        'sensor_layout': op.join(quinna, 'dataman/data_information/sensors_layout_names.csv'),
        'LI_dir': op.join(sub2ctx, 'derivatives/meg/sensor/lateralized_index/bands'),
        'meg_source_dir': op.join(sub2ctx, 'derivatives/meg/source/freesurfer'),
        'fs_sub_dir': op.join(quinna, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'all_subs_lat_src': op.join(sub2ctx, 'derivatives/meg/source/freesurfer/all_subs'),
        'corr_sensor_dir': op.join(sub2ctx, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nooutlier-psd'),
        'corr_source_dir': op.join(sub2ctx, 'derivatives/correlations/src_lat_grid_vol_correlation_nooutliers'),
    }
    return paths

# 1) Sensor band power
def plot_sensor_power(subject_id, band, paths):
    """Plot raw MEG band power topomap for grads and mags by computing PSD via Welch."""
    # Define frequency bands
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 40)}
    if band not in bands:
        raise ValueError(f"Unknown band {band}")
    fmin, fmax = bands[band]

    # Create fixed-length epochs from continuous data (2s each)
    epoched_fname = 'sub-CC' + str(subject_id) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(paths['epoched_dir'], epoched_fname)
    epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs

    # Welch parameters
    n_fft = 500
    welch_params = dict(fmin=fmin, fmax=fmax, picks='meg', n_fft=n_fft, n_overlap=int(n_fft / 2))

    # Compute PSD for grads
    epochs_grad = epochs.copy().pick('grad')
    psd_grad = epochs_grad.compute_psd(method='welch', **welch_params, n_jobs=4, verbose=False)
    # Average over frequencies and epochs
    grad_power = psd_grad.get_data().mean(axis=(0, 2))

    # Compute PSD for mags
    epochs_mag = epochs.copy().pick('mag')
    psd_mag = epochs_mag.compute_psd(method='welch', **welch_params, n_jobs=4, verbose=False)
    mag_power = psd_mag.get_data().mean(axis=(0, 2))

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im, cn = mne.viz.plot_topomap(grad_power, epochs_grad.info, axes=axes[0], show=False, cmap='RdBu_r',
                         vlim=(min(grad_power), max(grad_power)), contours=0, image_interp='nearest')
    axes[0].set_title(f'Sensor {band} Band Power (Grads)')
    cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Power Values')
    mne.viz.plot_topomap(mag_power, epochs_mag.info, axes=axes[1], show=False, cmap='RdBu_r',
                          vlim=(min(mag_power), max(mag_power)), contours=0, image_interp='nearest')
    axes[1].set_title(f'Sensor {band} Band Power (Mags)')
    cbar = fig.colorbar(im, ax=axes[1], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Power Values')
    fig.suptitle(f'Subject {subject_id}', fontsize=20) # or plt.suptitle('Main title')
    plt.show()

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
        
        return raw, raw.info, raw_mag.info
    
def plot_sensor_lat_power(subject_id, band, paths):
    """Plot lateralised band power topomap for mags and grads"""
    df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise.csv'))
    row = df[df['subject_ID']==subject_id]
    
    # Determine magnetometer and gradiometer columns
    sensor_cols = row.columns.drop('subject_ID')
    grad_cols = [c for c in sensor_cols if c.endswith('2') or c.endswith('3')]
    mag_cols = [c for c in sensor_cols if c.endswith('1')]

    # Extract values
    grad_vals = row[grad_cols].values.flatten()
    mag_vals = row[mag_cols].values.flatten()

    # Read info
    _, grad_info, *rest = read_raw_info(paths, ch_type='grad')  
    _, mag_info, *rest = read_raw_info(paths, ch_type='mag')  

    fig, axes = plt.subplots(1,2, figsize=(10,5))
    mne.viz.plot_topomap(grad_vals, grad_info, axes=axes[0], show=False)
    im, cn = mne.viz.plot_topomap(grad_vals, grad_info, axes=axes[0], show=False, cmap='RdBu_r',
                        vlim=(min(grad_vals), max(grad_vals)), contours=0, image_interp='nearest')
    axes[0].set_title(f'Sensor {band} Band Lateralised Power (Grads)')
    cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Lateralised Power Values')
    axes[0].set_xlim(0, )  # remove the left half of topoplot

    mne.viz.plot_topomap(mag_vals, mag_info, axes=axes[1], show=False)
    im, cn = mne.viz.plot_topomap(mag_vals, mag_info, axes=axes[1], show=False, cmap='RdBu_r',
                        vlim=(min(mag_vals), max(mag_vals)), contours=0, image_interp='nearest')
    axes[1].set_title(f'Sensor {band} Band Lateralised Power (Mags)')
    cbar = fig.colorbar(im, ax=axes[1], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Lateralised Power Values')
    axes[1].set_xlim(0, )  # remove the left half of topoplot

    fig.suptitle(f'Subject {subject_id}', fontsize=20) # or plt.suptitle('Main title')
    plt.show()


def plot_source_band_power(subject_id, band, paths):
    """Plot source-space band power (VolSourceEstimate) by averaging per-Hz STC files in the band."""
    # Define bands
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 40)}
    if band not in bands:
        raise ValueError(f"Unknown band {band}")
    fmin, fmax = bands[band]

    # Directory containing per-Hz STC files for this subject
    stc_dir = op.join(paths['meg_source_dir'], f'sub-CC{subject_id}', 'stc_morphd_perHz')
    if not op.isdir(stc_dir):
        raise FileNotFoundError(f"STC directory not found: {stc_dir}")

    # Gather STC files within freq range
    stc_grad_files = []
    stc_mag_files = []
    for fname in os.listdir(stc_dir):
        if fname.endswith('.stc') and '_' in fname:
            # extract frequency from filename, e.g. '_4.0-vl.stc'
            parts = fname[:-7].split('_')
            try:
                freq = float(parts[-1])
                if fmin <= freq < fmax:
                    if 'mag' in fname:
                        stc_mag_files.append(op.join(stc_dir, fname))
                    elif 'grad' in fname:
                        stc_grad_files.append(op.join(stc_dir, fname))
            except ValueError:
                continue
    if not stc_grad_files or not stc_mag_files:
        raise FileNotFoundError(f"No STC files found in {stc_dir} for band {band}")

    # Prepare for plotting on fs
    fetch_fsaverage(paths["fs_sub_dir"])  # ensure fsaverage src exists
    fname_fsaverage_src = op.join(paths["fs_sub_dir"], "fsaverage", "bem", "fsaverage-vol-5-src.fif")

    src_fs = mne.read_source_spaces(fname_fsaverage_src)
    initial_pos=np.array([19, -50, 29]) * 0.001

    # Read all STCs and average data
    # Grads
    stc_grads = [mne.read_source_estimate(fpath) for fpath in sorted(stc_grad_files)]
    data_stack_grads = np.stack([stc.data[:, 0] for stc in stc_grads], axis=1)
    mean_data_grads = data_stack_grads.mean(axis=1)
    stc_grads_band = stc_grads[0]
    stc_grads_band.data = mean_data_grads[:, np.newaxis] # data should have 2 dimensions
    stc_morphd_band_dir = op.join(paths['meg_source_dir'], f'sub-CC{subject_id}', 'stc_morphd_band')
    if not op.exists(stc_morphd_band_dir):
        os.makedirs(stc_morphd_band_dir)
    stc_grads_band.save(op.join(stc_morphd_band_dir,f'sub-CC120469_fsmorph_stc_multitaper_grad_{band}'))
    # Mags
    stc_mags = [mne.read_source_estimate(fpath) for fpath in sorted(stc_mag_files)]
    data_stack_mags = np.stack([stc.data[:, 0] for stc in stc_mags], axis=1)
    mean_data_mags = data_stack_mags.mean(axis=1)
    stc_mags_band = stc_grads[0]
    stc_mags_band.data = mean_data_mags[:, np.newaxis] # data should have 2 dimensions
    stc_mags_band.save(op.join(stc_morphd_band_dir,f'sub-CC120469_fsmorph_stc_multitaper_mag_{band}'))

    # Plot
    stc_grads_band.plot(
        src=src_fs,
        mode="stat_map",
        subjects_dir=paths["fs_sub_dir"],
        initial_pos=initial_pos,
        verbose=True,
    )
    stc_mags_band.plot(
        src=src_fs,
        mode="stat_map",
        subjects_dir=paths["fs_sub_dir"],
        initial_pos=initial_pos,
        verbose=True,
    )


def plot_source_lat_power(band, paths):
    """Plot source lateralised band power Volume Estimate"""
    csv = op.join(paths['all_subs_lat_src'], f'all_subs_lateralised_src_power_grad_{band}_band_avg.csv')
    data = pd.read_csv(csv, index_col=0).mean(axis=1).values
    # Placeholder stc
    stc = mne.VolSourceEstimate(data[:,None], vertices=[np.arange(len(data))], tmin=0, tstep=1, subject='fsaverage')
    brain = stc.plot(subjects_dir=None,  initial_time=0, cmap='RdBu_r')
    brain.show()


def plot_sensor_vol_corr(substr, band, paths):
    """Plot sensor lat-power vs volume-lat correlation topomap"""
    df = pd.read_csv(op.join(paths['corr_sensor_dir'], f'{substr}_allpairs_{band}_spearmanr.csv'))
    raw = mne.io.read_raw_fif(paths['sample_meg_file'], verbose=False)
    # assume df['rval'] matches raw.ch_names
    data = df[f'{band.lower()}_rval'].values
    mne.viz.plot_topomap(data, raw.info)
    plt.show()


def plot_source_vol_corr(substr, band, paths):
    """Plot source lat-power vs volume-lat correlation Volume Estimate"""
    csv = op.join(paths['corr_source_dir'], f'spearman-pval_src_lat_power_vol_grad_{band}.csv')
    df = pd.read_csv(csv, index_col=0)
    data = df[substr].values
    stc = mne.VolSourceEstimate(data[:,None], vertices=[np.arange(len(data))], tmin=0, tstep=1, subject='fsaverage')
    stc.plot(subjects_dir=None, hemi='both', initial_time=0, cmap='coolwarm').show()


def main():
    paths = setup_paths('mac')
    subject_id = '110182'
    band = 'Alpha'
    substr = 'Thal'

    # 1 & 2: sensor power
    plot_sensor_power(subject_id, band, paths)
    plot_sensor_lat_power(subject_id, band, paths)

    # 3 & 4: source power
    plot_source_band_power(subject_id, band, paths)
    plot_source_lat_power(band, paths)

    # 5 & 6: correlations
    plot_sensor_vol_corr(substr, band, paths)
    plot_source_vol_corr(substr, band, paths)

if __name__ == '__main__':
    main()
