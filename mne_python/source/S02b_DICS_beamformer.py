# -*- coding: utf-8 -*-
"""
===============================================
S02b. Using beamformer to localize oscillatory 
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

import numpy as np

import os
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import read_csd


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
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'meg_source_dir': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer'),
        'meg_sensor_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-2sec'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
    }
    return paths


def load_subjects(good_sub_sheet):
    """Load subject IDs from the CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    return good_subject_pd.set_index('Unnamed: 0')


def construct_paths(subjectID, paths, csd_method='multitaper'):
    """Construct file paths for a given subject, space type, and frequency band.
    csd_method = 'fourier' or 'multitaper' """

    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    file_paths = {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        'fwd_surf_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-surf.fif'),
        'fwd_vol_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-vol.fif'),
        'epoched_epo_fname': op.join(paths['meg_sensor_dir'], f'{fs_sub[:-4]}_2sec_epod-epo.fif'),
        f'csd_{csd_method}_mag_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_csd_{csd_method}_mag.h5'),
        f'csd_{csd_method}_grad_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_csd_{csd_method}_grad.h5'),
        'mag_stc_fname': op.join(deriv_folder, 'stc_perHz', f'{fs_sub[:-4]}_stc_{csd_method}_mag'),
        'grad_stc_fname': op.join(deriv_folder,'stc_perHz', f'{fs_sub[:-4]}_stc_{csd_method}_grad'),
        'stc_mag_plot_fname': op.join(deriv_folder, 'plots', f"{fs_sub}_dics_{csd_method}_mag"),
        'stc_grad_plot_fname': op.join(deriv_folder, 'plots', f"{fs_sub}_dics_{csd_method}_grad")
    }

    return file_paths

def read_forward_rank_csd(file_paths, space='volume', csd_method='multitaper'):

    print('Reading forward model')
    forward = mne.read_forward_solution(file_paths['fwd_vol_fname'] if space == 'volume' else file_paths['fwd_surf_fname'])

    print('Source reconstruction on magnetometers and gradiometers separately')  #separate mags and grads from here.
    epoched_epochs = mne.read_epochs(file_paths['epoched_epo_fname'], preload=True, verbose=True, proj=False)
    mags = epoched_epochs.copy().pick("mag")
    grads = epoched_epochs.copy().pick("grad")

    print('Computing rank')
    rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative', proj=False)
    rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative', proj=False)

    print('Reading CSD')
    csd_mags = read_csd(file_paths[f'csd_{csd_method}_mag_fname'])
    csd_grads = read_csd(file_paths[f'csd_{csd_method}_grad_fname'])

    return forward, mags, grads, rank_mag, rank_grad, csd_mags, csd_grads

def plotting_stc(mags, grads, csd_mags, csd_grads, rank_mag, rank_grad,
                 forward, file_paths, paths, reg=0.01):
    """Makes dics and  applys on csd on whole spectrum for
    plotting."""
    # Plot and save results
    print("Plotting results for double-checking...")
    filters_mag = make_dics(mags.info, 
                            forward, 
                            csd_mags.mean(), 
                            noise_csd=None, 
                            reg=reg, 
                            pick_ori='max-power', 
                            reduce_rank=True, 
                            real_filter=True, 
                            rank=rank_mag, 
                            depth=None, 
                            inversion='matrix', 
                            weight_norm="unit-noise-gain")
    stc_mag,_ = apply_dics_csd(csd_mags.mean(), filters_mag)
   
    filters_grad = make_dics(grads.info, 
                            forward, 
                            csd_grads.mean(), 
                            noise_csd=None, 
                            reg=reg, 
                            pick_ori='max-power', 
                            reduce_rank=True, 
                            real_filter=True, 
                            rank=rank_grad, 
                            depth=None, 
                            inversion='matrix', 
                            weight_norm="unit-noise-gain")
    stc_grad,_ = apply_dics_csd(csd_grads.mean(), filters_grad)

    if not op.exists(op.join(file_paths["deriv_folder"], 'plots')):
        os.makedirs(op.join(file_paths["deriv_folder"], 'plots'))

    stc_mag.plot(src=forward["src"], 
                    subject=file_paths['fs_sub'], 
                    subjects_dir=paths['fs_sub_dir'], 
                    mode='stat_map',
                    verbose=True).savefig(f"{file_paths['stc_mag_plot_fname']}.png")

    stc_grad.plot(src=forward["src"], 
                    subject=file_paths['fs_sub'], 
                    subjects_dir=paths['fs_sub_dir'], 
                    mode='stat_map', 
                    verbose=True).savefig(f"{file_paths['stc_grad_plot_fname']}.png")

def run_dics_per_Hz(mags, grads, freq, forward, csd_mags, csd_grads, 
             rank_mag, rank_grad, file_paths, reg=0.01, csd_method='multitaper'):
    """Run DICS for a given subject for the given freqs (1hz by 1hz)."""

    print(f'Create DICS filters on {csd_method} csd and apply with egularisation = {reg} for {freq}Hz')
    csd_mags_freq = csd_mags.copy().pick_frequency(freq)
    filters_mag = make_dics(mags.info, 
                            forward, 
                            csd_mags_freq, 
                            noise_csd=None, 
                            reg=reg, 
                            pick_ori='max-power', 
                            reduce_rank=True, 
                            real_filter=True, 
                            rank=rank_mag, 
                            depth=None, 
                            inversion='matrix', 
                            weight_norm="unit-noise-gain")
    stc_mag_freq, freq = apply_dics_csd(csd_mags_freq, filters_mag)

    csd_grads_freq = csd_grads.copy().pick_frequency(freq)
    filters_grad = make_dics(grads.info, 
                            forward, 
                            csd_grads_freq, 
                            noise_csd=None, 
                            reg=reg, 
                            pick_ori='max-power', 
                            reduce_rank=True, 
                            real_filter=True, 
                            rank=rank_grad, 
                            depth=None, 
                            inversion='matrix', 
                            weight_norm="unit-noise-gain")
    stc_grad_freq, _ = apply_dics_csd(csd_grads_freq, filters_grad)
    
    # Save DICS results
    if not op.exists(op.join(file_paths["deriv_folder"], 'stc_perHz')):
        os.makedirs(op.join(file_paths["deriv_folder"], 'stc_perHz'))

    stc_mag_freq.save(f"{file_paths['mag_stc_fname']}_{freq}", overwrite=True)
    stc_grad_freq.save(f"{file_paths['grad_stc_fname']}_{freq}", overwrite=True)

    print(f"DICS results successfully saved for {freq}Hz")

def check_existing_dics(file_paths, freq):
    """Check if DICS results already exist for a subject."""
    if op.exists(f"{file_paths['mag_stc_fname']}_[{freq}]-vl.stc") and \
        op.exists(f"{file_paths['grad_stc_fname']}_[{freq}]-vl.stc"):
        print(f"DICS results already exist for {file_paths['fs_sub']} in {freq}. Skipping...")
        return True
    return False

def process_subject(subjectID, paths, plot, csd_method, space='volume', reg=0.01):

    print(f"Processing subject {subjectID}...")
    file_paths = construct_paths(subjectID, paths, csd_method=csd_method)

    (forward, mags, grads, rank_mag, rank_grad, 
        csd_mags, csd_grads) = read_forward_rank_csd(file_paths, 
                                                     space=space, 
                                                     csd_method=csd_method)
    if plot:
        plotting_stc(mags, grads, csd_mags, csd_grads, rank_mag, rank_grad,
                    forward, file_paths, paths, reg=reg)
    
    return file_paths, forward, mags, grads, rank_mag, rank_grad, csd_mags, csd_grads


def main():
    
    platform = 'bluebear'  # Set platform: 'mac' or 'bluebear'
    freqs = np.arange(1, 60.5, 0.5)  # range of frequencies for dics
    space = 'volume'  # Space type: 'surface' or 'volume'
    csd_method = 'multitaper'
    reg = 0.01
    plot = False

    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])

    for subjectID in good_subjects.index[200:630]:
        try:
            print(f"Running DICS with {csd_method} csd for subject {subjectID}, space: {space}")
            (file_paths, forward, 
            mags, grads, 
            rank_mag, rank_grad, 
            csd_mags, csd_grads) = process_subject(subjectID, 
                                                    paths, 
                                                    plot,
                                                    csd_method=csd_method, 
                                                    space=space, 
                                                    reg=reg)
            for freq in freqs:
                # Skip if forward solution already exists
                if not check_existing_dics(file_paths, freq):
                    print(f'Running DICS on {freq}Hz')
                    run_dics_per_Hz(mags, grads, freq, forward, csd_mags, csd_grads, 
                            rank_mag, rank_grad, file_paths, reg=reg, csd_method=csd_method)
        except Exception as e:
            print(f"Error processing subject {subjectID}: {e}")

if __name__ == "__main__":
    main()
