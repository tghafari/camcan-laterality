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
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'rds_dir': rds_dir,
        'epoched_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-7min50'),
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
        'meg_sensor_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-2sec'),
        'deriv_folder': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer')
    }
    return paths

def load_subjects(good_sub_sheet):
    """Load subject IDs from a CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # Set subject IDs as the index
    return good_subject_pd

def epoching_epochs(epoched_fif, duration=2):
    """This definition inputs the epoched data called epoched_fif and 
    epoch them into shorter epochs with epoched_epochs_duration.
    this is to reduce the computation time of csd_multitaper.
    duraion = 2 seconds, keep it equal to sensor level
    welch window."""
    
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
def compute_csd(epochs, fmin=1, fmax=60, n_fft=500):
    """ n_fft = 2*info['sfreq'] = n_fft in welch method for sensor level analyses """
    welch_params = dict(fmin=fmin, fmax=fmax, picks="meg", n_fft=n_fft)

    print(f'Calculating CSD matrices for {fmin} to {fmax}...')
    csd_fourier = csd_fourier(epochs, 
                             fmin=fmin, 
                             fmax=fmax, 
                             tmin=epochs.tmin, 
                             tmax=epochs.tmax, 
                             n_fft=n_fft, 
                             verbose=False, 
                             n_jobs=-1)
    csd_multitaper = csd_multitaper(epochs, 
                             fmin=fmin, 
                             fmax=fmax, 
                             tmin=epochs.tmin, 
                             tmax=epochs.tmax, 
                             bandwidth=1, 
                             low_bias=True, 
                             verbose=False, 
                             n_jobs=-1)
    return csd_fourier, csd_multitaper


def process_subject(subjectID, paths, fr_band):
    """
    Process a single subject for a given frequency band.
    Computes and saves the cross-spectral density matrices.
    """
    print(f"Processing subject {subjectID} for {fr_band} band...")

    # Define file paths
    fs_sub = f"sub-CC{subjectID}_T1w"
    epoched_fname = op.join(paths['epoched_dir'], f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif')
    meg_sensor_dir = paths['meg_sensor_dir']
    deriv_folder = op.join(paths['deriv_folder'], f'{fs_sub[:-4]}')
    deriv_epoched_epo_fname = op.join(meg_sensor_dir, f'{fs_sub[:-4]}_2sec_epod-epo.fif')
    deriv_csd_fourier_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_csd_fourier')
    deriv_csd_multitaper_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_csd_multitaper')

    # Skip if CSD files already exist
    if op.exists(deriv_csd_fourier_fname) and op.exists(deriv_csd_multitaper_fname):
        print(f"CSD already exists for subject {subjectID}. Skipping...")
        return
    
    if not op.exists(meg_sensor_dir):
        os.makedirs(meg_sensor_dir)

    # Epoch the data
    epoched_epochs = epoching_epochs(epoched_fname)
    epoched_epochs.save(deriv_epoched_epo_fname, overwrite=True)

    # Compute CSDs
    csd_fourier, csd_multitaper = compute_csd(epoched_epochs)

    # Save CSDs
    csd_fourier.save(deriv_mag_csd_fname, overwrite=True)
    csd_multitaper.save(deriv_grad_csd_fname, overwrite=True)

    print(f"Subject {subjectID} processed successfully for {fr_band} band.")

def main():
    platform = 'mac'  # Change to 'bluebear' if running on BlueBear
    fr_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Frequency bands to process

    # Set up paths and load subjects
    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])

    # Process each subject and frequency band
    for subjectID in good_subjects.index[0:10]:
        for fr_band in fr_bands:
            try:
                process_subject(subjectID, paths, fr_band)
            except Exception as e:
                print(f"Error processing subject {subjectID} in {fr_band} band: {e}")

if __name__ == "__main__":
    main()
