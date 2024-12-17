"""
===============================================
S02. Using beamformer to localize oscillatory 
power modulations

This script uses DICS to localize 
oscillatory power modulations based on spatial
filtering (DICS: in frequency domain). 
Multitaper CSD and the epochs have been 
prepared in S02a and will be read in this 
script.

written by Tara Ghafari
==============================================
"""

import os
import os.path as op
import pandas as pd
import mne
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import read_csd


def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'rds_dir': rds_dir,
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'meg_source_dir': op.join(rds_dir, 'derivatives/meg/source/freesurfer'),
        'meg_sensor_dir': op.join(rds_dir, 'derivatives/meg/sensor/epoched-1to8sec'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
    }
    return paths


def load_subjects(good_sub_sheet):
    """Load subject IDs from the CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    return good_subject_pd.set_index('Unnamed: 0')


def construct_paths(subjectID, paths, fr_band='alpha'):
    """Construct file paths for a given subject, space type, and frequency band."""
    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    file_paths = {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        'fwd_surf_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-surf.fif'),
        'fwd_vol_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-vol.fif'),
        'mag_epoched_fname': op.join(paths['meg_sensor_dir'], f'{fs_sub[:-4]}_mag_epod-epo.fif'),
        'grad_epoched_fname': op.join(paths['meg_sensor_dir'], f'{fs_sub[:-4]}_grad_epod-epo.fif'),
        'mag_csd_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_mag_csd_multitaper_{fr_band}.h5'),
        'grad_csd_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_grad_csd_multitaper_{fr_band}.h5'),
        'mag_stc_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_mag_stc_multitaper_{fr_band}'),
        'grad_stc_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_grad_stc_multitaper_{fr_band}'),
        'stc_mag_plot_fname': op.join(deriv_folder, f"{file_paths['fs_sub']}_mag_dics_plot{fr_band}.png"),
        'stc_grad_plot_fname': op.join(deriv_folder, f"{file_paths['fs_sub']}_grad_dics_plot{fr_band}.png")
    }

    return file_paths


def check_existing_dics(file_paths):
    """Check if DICS results already exist for a subject."""
    if op.exists(file_paths['mag_stc_fname']) and op.exists(file_paths['grad_stc_fname']):
        print(f"DICS results already exist for {file_paths['fs_sub']}. Skipping...")
        return True
    return False


def run_dics(subjectID, paths, space='volume', fr_band='alpha'):
    """Run DICS for a given subject."""
    file_paths = construct_paths(subjectID, paths, fr_band)

    # Skip subjects with existing DICS results
    if check_existing_dics(file_paths):
        return

    print(f"Running DICS for subject {subjectID}, space: {space}, frequency band: {fr_band}")
    reg = 0.01  # defined here for easier modifications
    print(f"regularisation = {reg}")

    print('Reading forward model')
    forward = mne.read_forward_solution(file_paths['fwd_vol_fname'] if space == 'volume' else file_paths['fwd_surf_fname'])

    print('Source reconstruction on magnetometers and gradiometers separately')
    mags = mne.read_epochs(file_paths['mag_epoched_fname'], preload=True, verbose=True, proj=False)
    grads = mne.read_epochs(file_paths['grad_epoched_fname'], preload=True, verbose=True, proj=False)

    print('Computing rank')
    rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative', proj=False)
    rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative', proj=False)

    print('Reading CSD')
    csd_mag = read_csd(file_paths['mag_csd_fname'])
    csd_grad = read_csd(file_paths['grad_csd_fname'])

    print('Create DICS filters and apply')
    filters_mag = make_dics(mags.info, 
                            forward, 
                            csd_mag.mean(), 
                            noise_csd=None, 
                            reg=reg, 
                            pick_ori='max-power', 
                            reduce_rank=True, 
                            real_filter=True, 
                            rank=rank_mag, 
                            depth=None, 
                            inversion='matrix', 
                            weight_norm="unit-noise-gain")
    stc_mag, _ = apply_dics_csd(csd_mag.mean(), filters_mag)

    filters_grad = make_dics(grads.info, 
                             forward, 
                             csd_grad.mean(), 
                             noise_csd=None, 
                             reg=reg, 
                             pick_ori='max-power', 
                             reduce_rank=True, 
                             real_filter=True, 
                             rank=rank_grad, 
                             depth=None, 
                             inversion='matrix', 
                             weight_norm="unit-noise-gain")
    stc_grad, _ = apply_dics_csd(csd_grad.mean(), filters_grad)

    # Plot and save results
    print("Plotting results for double-checking...")

    stc_mag.plot(src=forward["src"], 
                 subject=file_paths['fs_sub'], 
                 subjects_dir=paths['fs_sub_dir'], 
                 mode='stat_map',
                 verbose=True).savefig(file_paths['stc_mag_plot_fname'])
    
    stc_grad.plot(src=forward["src"], 
                  subject=file_paths['fs_sub'], 
                  subjects_dir=paths['fs_sub_dir'], 
                  mode='stat_map', 
                  verbose=True).savefig(file_paths['stc_grad_plot_fname'])

    # Save DICS results
    stc_mag.save(file_paths['mag_stc_fname'], overwrite=True)
    stc_grad.save(file_paths['grad_stc_fname'], overwrite=True)
    print(f"DICS results successfully saved for {subjectID}")


def main():
    platform = 'mac'  # Set platform: 'mac' or 'bluebear'
    fr_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Frequency bands to process
    space = 'volume'  # Space type: 'surface' or 'volume'

    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])

    for subjectID in good_subjects.index:
        for fr_band in fr_bands:
            try:
                run_dics(subjectID, paths, space=space, fr_band=fr_band)
            except Exception as e:
                print(f"Error processing subject {subjectID}: {e}")

if __name__ == "__main__":
    main()
