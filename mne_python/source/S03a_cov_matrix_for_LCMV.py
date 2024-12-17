# -*- coding: utf-8 -*-
"""
===============================================
S03a. Preparation for LCMV

This script reads in epochs and band-pass filters them
to the oscillatory band of interest. It then computes 
a common covariance matrix and saves both the filtered 
epochs and covariance matrix. These outputs will be used 
in S03b for LCMV.

Updates:
- Modularized code with reusable functions
- Support for all frequency bands
- Runs for all subjects listed in the demographics file
- Skips subjects if outputs already exist

written by Tara Ghafari
===============================================
"""

import os.path as op
import pandas as pd
import mne
from mne.cov import compute_covariance

# Utility functions
def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    return {
        'rds_dir': rds_dir,
        'epoched_dir': op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50'),
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
        'meg_source_dir': op.join(rds_dir, 'derivatives/meg/source/freesurfer'),
        'filtered_epo_dir': op.join(rds_dir, 'derivatives/meg/sensor/filtered'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat')
    }

def load_subjects(good_sub_sheet):
    """Load subject IDs from the demographics CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    return good_subject_pd.set_index('Unnamed: 0')

def construct_paths(subjectID, paths, fr_band):
    """Construct file paths for a given subject and frequency band."""
    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])
    deriv_folder_sensor = paths['filtered_epo_dir']
    
    return {
        'epoched_fname': op.join(paths['epoched_dir'], f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif'),
        'mag_filtered_fname': op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_mag_{fr_band}-epo.fif'),
        'grad_filtered_fname': op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_grad_{fr_band}-epo.fif'),
        'mag_cov_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_mag_cov_{fr_band}.fif'),
        'grad_cov_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_grad_cov_{fr_band}.fif')
    }

def check_existing(paths_dict):
    """Check if filtered data and covariance already exist."""
    if op.exists(paths_dict['mag_cov_fname']):
        print(f"Magnetometer files already exist for {paths_dict['mag_cov_fname']}. Skipping...")
        return True
    if op.exists(paths_dict['grad_cov_fname']):
        print(f"Gradiometer files already exist for {paths_dict['grad_cov_fname']}. Skipping...")
        return True
    return False

# Main script
def process_subject(subjectID, fr_band, paths):
    """Process a single subject for a given frequency band."""
    paths_dict = construct_paths(subjectID, paths, fr_band)
    
    if check_existing(paths_dict):
        return  # Skip processing if files already exist

    # Set frequency range for the band
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 60)
    }
    if fr_band not in freq_bands:
        raise ValueError(f"Invalid frequency band: {fr_band}")
    fmin, fmax = freq_bands[fr_band]

    # Load epochs
    print(f"Preparing LCMV for subject {subjectID}, band: {fr_band}")
    epochs = mne.read_epochs(paths_dict['epoched_fname'], preload=True, verbose=True)
    
    # Filter epochs to the desired band
    mags = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("mag")
    grads = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("grad")
    
    # Save filtered data
    mags.save(paths_dict['mag_filtered_fname'], overwrite=True)
    grads.save(paths_dict['grad_filtered_fname'], overwrite=True)
    
    # Compute covariance matrices
    print("Calculating covariance matrices...")
    """compute rank again in the main LCMV file."""
    rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative')
    common_cov_mag = compute_covariance(mags, 
                                        method='empirical', 
                                        rank=rank_mag,
                                        n_jobs=4, 
                                        verbose=True)
    rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative')
    common_cov_grad = compute_covariance(grads, 
                                         method='empirical', 
                                        rank=rank_grad,
                                         n_jobs=4, 
                                         verbose=True)
    

    # Save covariance matrices
    common_cov_mag.save(paths_dict['mag_cov_fname'], overwrite=True)
    common_cov_grad.save(paths_dict['grad_cov_fname'], overwrite=True)
    print(f"Finished processing subject {subjectID}, band: {fr_band}")

def main():
    platform = 'mac'  # Change to 'bluebear' if running on a different platform
    paths = setup_paths(platform)
    good_subject_pd = load_subjects(paths['good_sub_sheet'])
    
    # Iterate over all subjects and frequency bands
    freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

    for subjectID in good_subject_pd.index:
        for fr_band in freq_bands:
            try:
                process_subject(subjectID, fr_band, paths)
            except Exception as e:
                print(f"Error processing subject {subjectID}: {e}")

if __name__ == "__main__":
    main()
