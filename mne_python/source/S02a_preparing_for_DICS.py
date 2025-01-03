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
from mne.time_frequency import csd_multitaper, csd_fourier

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
        'meg_source_dir': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer')
    }
    return paths

def construct_paths(subjectID, paths, csd_method='multitaper'):
    """Construct file paths for a given subject, space type, and frequency band.
    csd_method = 'fourier' or 'multitaper' """

    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    file_paths = {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        'epoched_fname': op.join(paths['epoched_dir'], f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif'),
        'deriv_epoched_epo_fname': op.join(paths['meg_sensor_dir'], f'{fs_sub[:-4]}_2sec_epod-epo.fif'),
        f'deriv_csd_{csd_method}_mag_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_csd_{csd_method}_mag'),
        f'deriv_csd_{csd_method}_grad_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_csd_{csd_method}_grad'),
    }
    return file_paths

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

def compute_csd(epochs, fmin=1, fmax=60, n_fft=500, csd_method='multitaper'):
    """ n_fft = 2*info['sfreq'] = n_fft in welch method for sensor level analyses 
    csd_method= 'fourier' or 'multitaper' """

    if csd_method == 'fourier':
        print(f'Calculating CSD fourier for mags for {fmin} to {fmax}Hz...')
        csd_output_mag = csd_fourier(epochs, 
                                fmin=fmin, 
                                fmax=fmax, 
                                tmin=epochs.tmin, 
                                tmax=epochs.tmax, 
                                picks="mag",
                                n_fft=n_fft, 
                                verbose=False, 
                                n_jobs=-1)
        print(f"Calculating CSD fourier for grads for {fmin} to {fmax}Hz")
        csd_output_grad = csd_fourier(epochs, 
                                fmin=fmin, 
                                fmax=fmax, 
                                tmin=epochs.tmin, 
                                tmax=epochs.tmax, 
                                picks="grad",
                                n_fft=n_fft, 
                                verbose=False, 
                                n_jobs=-1)
        print('CSD fourier done!')

    elif csd_method == 'multitaper':
        print(f'Calculating CSD multitaper for mags for {fmin} to {fmax}Hz...')
        csd_output_mag = csd_multitaper(epochs, 
                                fmin=fmin, 
                                fmax=fmax, 
                                tmin=epochs.tmin, 
                                tmax=epochs.tmax, 
                                picks="mag",
                                bandwidth=1, 
                                low_bias=True, 
                                verbose=False, 
                                n_jobs=-1)
        print(f"Calculating CSD multitaper for grads for {fmin} to {fmax}Hz...")
        csd_output_grad = csd_multitaper(epochs, 
                                fmin=fmin, 
                                fmax=fmax, 
                                tmin=epochs.tmin, 
                                tmax=epochs.tmax, 
                                picks="grad",
                                bandwidth=1, 
                                low_bias=True, 
                                verbose=False, 
                                n_jobs=-1)
                
        print('CSD multitaper done!')

    return csd_output_mag, csd_output_grad

def process_subject(subjectID, paths, csd_method='multitaper'):
    """
    Process a single subject for a given frequency band.
    Computes and saves the cross-spectral density matrices.
    """
    print(f"Processing subject {subjectID} ...")

    file_paths = construct_paths(subjectID, paths, csd_method=csd_method)

    # Skip if CSD files already exist
    if op.exists(file_paths[f'deriv_csd_{csd_method}_mag_fname']) and \
        op.exists(file_paths[f'deriv_csd_{csd_method}_grad_fname']):
        print(f"CSD already exists for subject {subjectID}. Skipping...")
        return
    
    if not op.exists(paths["meg_sensor_dir"]):
        os.makedirs(paths["meg_sensor_dir"])

    # Epoch the data
    epoched_epochs = epoching_epochs(file_paths["epoched_fname"])
    epoched_epochs.save(file_paths["deriv_epoched_epo_fname"], overwrite=True)

    # Compute CSDs
    csd_output_mag, csd_output_grad = compute_csd(epoched_epochs, fmin=1, fmax=60, n_fft=500, csd_method=csd_method)

    # Save CSDs
    csd_output_mag.save(file_paths[f'deriv_csd_{csd_method}_mag_fname'], overwrite=True)
    csd_output_grad.save(file_paths[f'deriv_csd_{csd_method}_grad_fname'], overwrite=True)

    print(f"Subject {subjectID} processed successfully.")

def main():
    platform = 'bluebear'  # Change to 'bluebear' if running on BlueBear
    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])
    csd_method='multitaper'

    # Process each subject and frequency band
    for subjectID in good_subjects.index[1:50]:
        try:
            process_subject(subjectID, paths, csd_method=csd_method)
        except Exception as e:
            print(f"Error processing subject {subjectID}: {e}")

if __name__ == "__main__":
    main()
