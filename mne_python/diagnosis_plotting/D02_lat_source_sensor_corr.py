"""
====================================
D02_lat_source_sensor_corr

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
import sys
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import mne
from mne.datasets import fetch_fsaverage
import nibabel

# Add custom function path
main_dir = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/programming/camcan-laterality/mne_python'
if main_dir not in sys.path:
    sys.path.append(main_dir)

from source.S05_computing_lateralised_source_power import (
    compute_hemispheric_index,
    order_grid_positions,
    calculate_grid_lateralisation,
)
import correlations.CG03_visualising_grid_vol_correlation_ongrids as cg



# -----------------------------------------
# Path setup
# -----------------------------------------

def setup_paths(platform='mac'):

    if platform == 'bluebear':
        sub2ctx = '/rds/projects/j/jenseno-sub2ctx/camcan'
        quinna = '/rds/projects/q/quinna-camcan'
    else:
        sub2ctx = '/Volumes/jenseno-sub2ctx/camcan'
        quinna = '/Volumes/quinna-camcan'
    
    paths = {
        'sample_meg_file': op.join(quinna, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'epoched_dir': op.join(sub2ctx, 'derivatives/meg/sensor/epoched-7min50'),
        'sensor_layout': op.join(quinna, 'dataman/data_information/sensors_layout_names.csv'),
        'LI_dir': op.join(sub2ctx, 'derivatives/meg/sensor/lateralized_index/bands'),
        'meg_source_dir': op.join(sub2ctx, 'derivatives/meg/source/freesurfer'),
        'sub_list': op.join(quinna, 'dataman/data_information/FINAL_sublist-LV-LI-outliers-removed.csv'),
        'fs_sub_dir': op.join(quinna, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'all_subs_lat_src': op.join(sub2ctx, 'derivatives/meg/source/freesurfer/all_subs'),
        'corr_sensor_dir': op.join(sub2ctx, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nooutlier-psd'),
        'corr_source_dir': op.join(sub2ctx, 'derivatives/correlations/src_lat_grid_vol_correlation_nooutliers'),
    }
    return paths


def construct_paths(subject_id, paths, sensortype, csd_method, space):
    """
    Construct required file paths for a given subject and frequency band.
    runs per sensorytype and csd_method

    Parameters:
    - subject_id (str): Subject ID.
    - paths (dict): Dictionary of data paths.
    - sensortype (str): 'grad' or 'mag'.
    - csd_method (str): 'fourier' or 'multitaper'. only works if S02a and b have been run on that method.
    - space (str): 'vol' or 'surf'.

    Returns:
    - dict: File paths for the subject and frequency band.
    """

    fs_sub = f'sub-CC{subject_id}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    file_paths = {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        f'fwd_{space}': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-{space}.fif'),
        f'{sensortype}_{csd_method}_stc': op.join(deriv_folder,'stc_perHz', f'{fs_sub[:-4]}_stc_{csd_method}_{sensortype}'),
        f'fsmorph_{sensortype}_{csd_method}_stc_fname': op.join(deriv_folder, 'stc_morphd_perHz', f'{fs_sub[:-4]}_fsmorph_stc_{csd_method}_{sensortype}'),  # this is what we use to plot power in source after morphing
        f'grid_stc_{sensortype}_{csd_method}_csv': op.join(deriv_folder, 'grid_perHz', f'grid_stc_{sensortype}_{csd_method}'),  # this is the final grid power file before lateralisation after morphing
        f'grid_positions_{sensortype}_{csd_method}_csv': op.join(deriv_folder, 'grid_perHz', f'grid_positions_{sensortype}_{csd_method}'),
        f'grid_indices_{sensortype}_{csd_method}_csv': op.join(deriv_folder, 'grid_perHz', f'grid_indices_{sensortype}_{csd_method}'),
        f'lateralised_src_power_{sensortype}_{csd_method}_csv': op.join(deriv_folder, 'lat_source_perHz', f'lateralised_src_power_{sensortype}_{csd_method}'),
        f'lateralised_grid_{sensortype}_{csd_method}_figname': op.join(deriv_folder, 'plots', f'lateralised_grid_{sensortype}_{csd_method}'),
        'stc_VolEst_lateral_power_figname': op.join(deriv_folder, 'plots', f'stc_VolEst_lateral_power_{sensortype}_{csd_method}'),
        'stc_fsmorphd_figname': op.join(deriv_folder, 'plots', f'stc_fsmorphd_{sensortype}_{csd_method}'),
        f'stc_to_nifti_{sensortype}': op.join(deriv_folder, 'plots', f'stc_morphd_tonifti_{sensortype}')
    }
    return file_paths



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
    lim_grad = np.max(np.abs(grad_power))
    im_grad, _ = mne.viz.plot_topomap(grad_power, epochs_grad.info, axes=axes[0], show=False, cmap='RdBu_r',
                         vlim=(-lim_grad, lim_grad), contours=0, image_interp='nearest')
    axes[0].set_title(f'Sensor {band} Band Power (Grads)')
    cbar = fig.colorbar(im_grad, ax=axes[0], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Power Values')
    lim_mag = np.max(np.abs(mag_power))
    im_mag, _ = mne.viz.plot_topomap(mag_power, epochs_mag.info, axes=axes[1], show=False, cmap='RdBu_r',
                          vlim=(-lim_mag, lim_mag), contours=0, image_interp='nearest')
    axes[1].set_title(f'Sensor {band} Band Power (Mags)')
    cbar = fig.colorbar(im_mag, ax=axes[1], orientation='horizontal', location='bottom')
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
    row = df[df['subject_ID']==int(subject_id)]
    
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
    lim_grad = np.max(np.abs(grad_vals))
    # mne.viz.plot_topomap(grad_vals, grad_info, axes=axes[0], show=False)
    im, cn = mne.viz.plot_topomap(grad_vals, grad_info, axes=axes[0], show=False, cmap='RdBu_r',
                        vlim=(-lim_grad, lim_grad), contours=0, image_interp='nearest')
    axes[0].set_title(f'Sensor {band} Band Lateralised Power (Grads)')
    cbar = fig.colorbar(im, ax=axes[0], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Lateralised Power Values')
    axes[0].set_xlim(0, )  # remove the left half of topoplot

    lim_mag = np.max(np.abs(mag_vals))
    # mne.viz.plot_topomap(mag_vals, mag_info, axes=axes[1], show=False)
    im, cn = mne.viz.plot_topomap(mag_vals, mag_info, axes=axes[1], show=False, cmap='RdBu_r',
                        vlim=(-lim_mag, lim_mag), contours=0, image_interp='nearest')
    axes[1].set_title(f'Sensor {band} Band Lateralised Power (Mags)')
    cbar = fig.colorbar(im, ax=axes[1], orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Lateralised Power Values')
    axes[1].set_xlim(0, )  # remove the left half of topoplot

    fig.suptitle(f'Subject {subject_id}', fontsize=20) # or plt.suptitle('Main title')
    plt.show()

def plot_source_band_power(subject_id, band, paths, src_fs, sensortype, csd_method, space):
    """Plot source-space band power (VolSourceEstimate) for a given sensor type ('mag' or 'grad') 
    by averaging per-Hz STC files in the specified frequency band."""
    
    # Validate sensor type
    if sensortype not in ['mag', 'grad']:
        raise ValueError(f"sensortype must be either 'mag' or 'grad', not '{sensortype}'")

    # Define bands
    bands = {'Delta': (1, 4), 'Theta': (4, 8), 'Alpha': (8, 14), 'Beta': (14, 40)}
    if band not in bands:
        raise ValueError(f"Unknown band {band}")
    fmin, fmax = bands[band]

    # Directory containing per-Hz STC files
    stc_dir = op.join(paths['meg_source_dir'], f'sub-CC{subject_id}', 'stc_morphd_perHz')
    if not op.isdir(stc_dir):
        raise FileNotFoundError(f"STC directory not found: {stc_dir}")
    
    file_paths = construct_paths(subject_id, paths, sensortype, csd_method, space)

    # Find relevant STC files for the given sensor type
    stc_files = []
    for fname in os.listdir(stc_dir):
        if fname.endswith('.stc') and '_' in fname and sensortype in fname:
            try:
                freq = float(fname[:-7].split('_')[-1])
                if fmin <= freq < fmax:
                    stc_files.append(op.join(stc_dir, fname))
            except ValueError:
                continue

    if not stc_files:
        raise FileNotFoundError(f"No STC files found for sensor type '{sensortype}' in {stc_dir} for band {band}")

    # Load and average the STC data
    stcs = [mne.read_source_estimate(fpath) for fpath in sorted(stc_files)]
    data_stack = np.stack([stc.data[:, 0] for stc in stcs], axis=1)
    mean_data = data_stack.mean(axis=1)
    stc_band = stcs[0].copy()
    stc_band.data = mean_data[:, np.newaxis]

    # Output directory
    stc_morphd_band_dir = op.join(paths['meg_source_dir'], f'sub-CC{subject_id}', 'stc_morphd_band')
    os.makedirs(stc_morphd_band_dir, exist_ok=True)

    # Save averaged STC
    stc_band.save(op.join(stc_morphd_band_dir, f'sub-CC{subject_id}_fsmorph_stc_multitaper_{sensortype}_{band}'),
                  overwrite=True)

    # Convert to NIfTI
    img = stc_band.as_volume(src_fs, dest='mri', mri_resolution=True, format='nifti1')
    nibabel.nifti1.save(img, f'{op.join(file_paths[f'stc_to_nifti_{sensortype}'])}_{band}.nii.gz')

    # Plot
    stc_band.plot(
        src=src_fs,
        mode="stat_map",
        subjects_dir=paths["fs_sub_dir"],
        verbose=True,
    )

    return stc_band

def plot_source_lat_power(subject_id, band, paths, src_fs, stc_band, sensortype,
                          csd_method, space, do_plot_3d=False):
    """
    Plot source lateralised band power on a Volume Estimate.
    
    Parameters:
    - band: str, one of ['Delta', 'Theta', 'Alpha', 'Beta']
    - paths: dict, general path dictionary
    - src_fs: mne.SourceSpaces, fsaverage volumetric source space
    - stc_band: mne.SourceEstimate, band-averaged morphed STC object
    - sensortype: str, 'grad' or 'mag'
    - csd_method: str, default 'multitaper'
    - do_plot_3d: bool, whether to also show a 3D interactive plot
    """

    file_paths = construct_paths(subject_id, paths, sensortype, csd_method, space)

    # --- Compute lateralised source power
    right_tc, left_tc, right_pos, left_pos, right_idx, left_idx, right_reg_idx, left_reg_idx = \
        compute_hemispheric_index(stc_band, src_fs)

    (_, _, _, _, ord_right_reg_idx, _,
     ord_right_tc, ord_left_tc) = order_grid_positions(
        right_pos, left_pos,
        right_idx, left_idx,
        right_reg_idx, left_reg_idx,
        right_tc, left_tc,
        file_paths, band, sensortype, csd_method
    )

    # Compute lateralisation index (R - L) at each ordered right grid location
    lateralised_power_arr = calculate_grid_lateralisation(
        ord_right_tc,
        ord_left_tc,
        file_paths, sensortype, csd_method, band
    )
    
    # Step 1: Build VolSourceEstimate object
    """Create an mne.VolSourceEstimate object for lateralised_power_arr, 
    ensuring the data structure is correctly formatted"""
    # Initialize an empty array with zeros for all dipoles in the source space
    n_dipoles = sum(len(s['vertno']) for s in src_fs)
    lateralised_data = np.zeros((n_dipoles, 1))  # One timepoint
    # Fill the right side of the vol estimate with lateralised powers (left side is all zero)
    for i, idx in enumerate(ord_right_reg_idx):
        lateralised_data[idx, 0] = lateralised_power_arr[i]

    # Step 2: Create the VolSourceEstimate object
    vertices = [np.array(src_fs[0]['vertno'])]
    
    stc_lat = mne.VolSourceEstimate(
        data=lateralised_data,
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject='fsaverage'
    )

    # Step 3: Plot the lateralized power on the brain
    initial_pos=np.array([19, -50, 29]) * 0.001
    brain = stc_lat.plot(
        src=src_fs,
        subject='fsaverage',
        subjects_dir=paths["fs_sub_dir"],
        mode='stat_map',
        colorbar=True,
        # initial_pos=initial_pos,
        verbose=True
    )
    os.makedirs(op.join(file_paths['deriv_folder'], 'plots'), exist_ok=True)
    brain.savefig(f"{file_paths['stc_VolEst_lateral_power_figname']}_{sensortype}_{band}.png")

    # Convert to NIfTI
    img = stc_lat.as_volume(src_fs, dest='mri', mri_resolution=True, format='nifti1')
    nibabel.nifti1.save(img, f'lateralised_{op.join(file_paths[f'stc_to_nifti_{sensortype}'])}_{band}.nii.gz')

    # --- Optional interactive 3D plot
    if do_plot_3d:
        # Plot in 3d
        kwargs = dict(
            subjects_dir=paths["fs_sub_dir"],
            hemi='both',
            size=(600, 600),
            views='axial',
            brain_kwargs=dict(silhouette=True),
            initial_time=0.087,
            verbose=True,
        )
        stc_lat.plot_3d(
            src=src_fs,
            **kwargs,
        )

def plot_sensor_vol_corr(substr, band, paths, sensortype):
    """Plot sensor lat-power vs volume-lat correlation topomap"""
    corr_df = pd.read_csv(op.join(paths['corr_sensor_dir'], f'{substr}_allpairs_{band}_spearmanr.csv'))
    raw, info, *other = read_raw_info(paths, sensortype)  
    # Filter sensor pairs based on magnetometer or gradiometer channel suffix
    if sensortype == 'mag':
        sensor_mask = corr_df['sensor_pair'].str.endswith('1')
    elif sensortype == 'grad':
        sensor_mask = corr_df['sensor_pair'].str.endswith('2') | corr_df['sensor_pair'].str.endswith('3')
    else:
        raise ValueError(f"Unsupported ch_type: {sensortype}")

    filtered_df = corr_df[sensor_mask]
    if filtered_df.empty:
        print(f"No matching sensors for {sensortype} in {substr}-{band}")

    # Apply the mask to select rows
    r_obs = filtered_df[f'{band.lower()}_rval'].to_numpy()
    spearman_p = filtered_df[f'{band.lower()}_pval'].to_numpy()  # for before cluster permutation p-values
    significant_mask_spearman = spearman_p < 0.05  # if you want to plot without cluster permutation tests
    print(f"{substr}-{band} ({sensortype}): {significant_mask_spearman.sum()} significant sensors")

    # 5. Visualize topomap with significant mask and cluster labels
    fig, ax = plt.subplots()

    # Define the significant clusters
    mask = significant_mask_spearman
    mask_params = dict(
        marker='o', markerfacecolor='w', markeredgecolor='k',
        linewidth=1, markersize=10
    ) 

    # Prepare the data for visualisation
    """combines planar gradiometer to use mag info (to plot positive and negative values)"""
    if sensortype == 'grad':
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
    ax.set_title(f'{substr}-{band} Spearman r ({sensortype})')
    plt.show()


def diagnosis_plotting():

    # Sanity check with random participants
    to_tests = np.arange(0,20)

    paths = setup_paths('mac')
    # subject_id = input("Enter subject ID (e.g., 120469)").strip()
    substr = input("Enter subcortical structure (e.g., Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    band = input("Enter band (e.g., Delta, Theta, Alph, Beta): ").strip()
    csd_method = 'multitaper'  # these should be kept fix for now
    space = 'vol'  # these should be kept fix for now

    for _ in to_tests:
        sub_list = pd.read_csv(paths['sub_list'])
        sub_list = sub_list.to_numpy()
        random_subject = sub_list[np.random.randint(0, len(sub_list))][0]
        
        # # # 1 & 2: sensor power
        plot_sensor_power(random_subject, band, paths)
        plot_sensor_lat_power(random_subject, band, paths)

        # 3 & 4: source power
        # Read forward model for volume plots
        fetch_fsaverage(paths["fs_sub_dir"])  # ensure fsaverage src exists
        fname_fsaverage_src = op.join(paths["fs_sub_dir"], "fsaverage", "bem", "fsaverage-vol-5-src.fif")
        src_fs = mne.read_source_spaces(fname_fsaverage_src)

        # stc_grads_band = plot_source_band_power(random_subject, band, paths, src_fs, 'grad', csd_method, space)
        # plot_source_lat_power(random_subject, band, paths, src_fs, stc_grads_band, 'grad',
        #                     csd_method, space, do_plot_3d=True)
        # input("Press Enter to continue to the next plot...")

        stc_mags_band = plot_source_band_power(random_subject, band, paths, src_fs, 'mag', csd_method, space)
        plot_source_lat_power(random_subject, band, paths, src_fs, stc_mags_band, 'mag',
                            csd_method, space, do_plot_3d=True)
        # input("Press Enter to continue to the next plot...")

    # 5 & 6: correlations
    plot_sensor_vol_corr(substr, band, paths, 'grad')
    input("Press Enter to continue to the next plot...")

    plot_sensor_vol_corr(substr, band, paths, 'mag')
    input("Press Enter to continue to the next plot...")

    cg.visualising_grid_vol_correlation()  # costum script from before

if __name__ == '__main__':
    diagnosis_plotting()
