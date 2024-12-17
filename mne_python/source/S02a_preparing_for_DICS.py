# -*- coding: utf-8 -*-
"""
===============================================
S02a. Preparing for DICS

This script processes all subjects and computes
the cross-spectral density (CSD) matrices using
csd_multitaper on resting-state data for multiple
frequency bands.

written by Tara Ghafari
===============================================
"""

import os
import os.path as op
import pandas as pd
import mne
from mne.time_frequency import csd_multitaper

def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'rds_dir': rds_dir,
        'epoched_dir': op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50'),
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
        'deriv_folder_sensor': op.join(rds_dir, 'derivatives/meg/sensor/epoched-1to8sec'),
        'deriv_folder': op.join(rds_dir, 'derivatives/meg/source/freesurfer')
    }
    return paths

def load_subjects(good_sub_sheet):
    """Load subject IDs from a CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # Set subject IDs as the index
    return good_subject_pd

def get_fr_band_params(fr_band):
    """Return frequency band parameters based on the selected band."""
    bands = {
        'delta': {'fmin': 1, 'fmax': 4, 'bandwidth': 1., 'duration': 8},
        'theta': {'fmin': 4, 'fmax': 8, 'bandwidth': 1., 'duration': 4},
        'alpha': {'fmin': 8, 'fmax': 12, 'bandwidth': 1., 'duration': 2},
        'beta': {'fmin': 12, 'fmax': 30, 'bandwidth': 1., 'duration': 2},
        'gamma': {'fmin': 30, 'fmax': 60, 'bandwidth': 4., 'duration': 1}
    }
    if fr_band not in bands:
        raise ValueError(f"Invalid frequency band: {fr_band}")
    return bands[fr_band]

def epoching_epochs(epoched_fif, duration):
    """This definition inputs the epoched data called epoched_fif and 
    epoch them into shorter epochs with epoched_epochs_duration.
    this is to reduce the computation time of csd_multitaper."""
    
    print('Reading epochs')
    epochs = mne.read_epochs(epoched_fif, 
                             preload=True, 
                             proj=False, 
                             verbose=True)

    print(f'Epoching to {duration} seconds')
    for epochs_data in epochs:
        raw_epoch = mne.io.RawArray(epochs_data, 
                                    epochs.info)
        epoched_epochs = mne.make_fixed_length_epochs(raw_epoch, 
                                                      duration=duration, 
                                                      overlap=0.5, 
                                                      preload=True)
    return epoched_epochs

def process_subject(subjectID, paths, fr_band):
    """
    Process a single subject for a given frequency band.
    Computes and saves the cross-spectral density matrices.
    """
    print(f"Processing subject {subjectID} for {fr_band} band...")

    # Get frequency band parameters
    band_params = get_fr_band_params(fr_band)
    fmin, fmax, bandwidth, duration = band_params['fmin'], band_params['fmax'], band_params['bandwidth'], band_params['duration']

    # Define file paths
    fs_sub = f"sub-CC{subjectID}_T1w"
    epoched_fname = op.join(paths['epoched_dir'], f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif')
    deriv_folder_sensor = paths['deriv_folder_sensor']
    deriv_folder = op.join(paths['deriv_folder'], f'{fs_sub[:-4]}')
    deriv_mag_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_mag_{duration}sec_epod-epo.fif')
    deriv_grad_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_grad_{duration}sec_epod-epo.fif')
    deriv_mag_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_mag_csd_multitaper_{fr_band}')
    deriv_grad_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_grad_csd_multitaper_{fr_band}')

    # Skip if CSD files already exist
    if op.exists(deriv_mag_csd_fname) and op.exists(deriv_grad_csd_fname):
        print(f"CSD already exists for subject {subjectID} in {fr_band} band. Skipping...")
        return
    
    if not op.exists(deriv_folder_sensor):
        os.makedirs(deriv_folder_sensor)
    # Epoch the data
    epoched_epochs = epoching_epochs(epoched_fname, duration)

    # Separate mags and grads
    mags = epoched_epochs.copy().pick("mag")
    grads = epoched_epochs.copy().pick("grad")

    # Save mags and grads
    mags.save(deriv_mag_epoched_fname, overwrite=True)
    grads.save(deriv_grad_epoched_fname, overwrite=True)

    # Compute CSDs
    print(f'Calculating CSD matrices for {fr_band} band...')
    csd_mag = csd_multitaper(mags, 
                             fmin=fmin, 
                             fmax=fmax, 
                             tmin=mags.tmin, 
                             tmax=mags.tmax, 
                             bandwidth=bandwidth, 
                             low_bias=True, 
                             verbose=False, 
                             n_jobs=-1)
    csd_grad = csd_multitaper(grads, 
                              fmin=fmin, 
                              fmax=fmax, 
                              tmin=grads.tmin, 
                              tmax=grads.tmax, 
                              bandwidth=bandwidth, 
                              low_bias=True, 
                              verbose=False, 
                              n_jobs=-1)

    # Save CSDs
    csd_mag.save(deriv_mag_csd_fname, overwrite=True)
    csd_grad.save(deriv_grad_csd_fname, overwrite=True)

    print(f"Subject {subjectID} processed successfully for {fr_band} band.")

def main():
    platform = 'mac'  # Change to 'bluebear' if running on BlueBear
    fr_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Frequency bands to process

    # Set up paths and load subjects
    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])

    # Process each subject and frequency band
    for subjectID in good_subjects.index:
        for fr_band in fr_bands:
            try:
                process_subject(subjectID, paths, fr_band)
            except Exception as e:
                print(f"Error processing subject {subjectID} in {fr_band} band: {e}")

if __name__ == "__main__":
    main()
