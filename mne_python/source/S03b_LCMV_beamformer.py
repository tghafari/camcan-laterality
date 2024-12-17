# -*- coding: utf-8 -*-
"""
===============================================
S03b. Using beamformer to localize oscillatory 
power modulations

This script uses LCMV to localize 
oscillatory power modulations based on spatial
filtering (DICS: in frequency domain). 

written by Tara Ghafari
t.ghafari@bham.ac.uk
==============================================
Notes:
    - Step 1: Compute source space
    - Step 2: Forward model
"""

import os.path as op
import pandas as pd
import mne
from mne.beamformer import make_lcmv, apply_lcmv_cov

def setup_paths(platform="mac"):
    """
    Set up the directory paths based on the platform (mac or bluebear).
    """
    if platform == "bluebear":
        rds_dir = "/rds/projects/q/quinna-camcan"
    elif platform == "mac":
        rds_dir = "/Volumes/quinna-camcan"
    else:
        raise ValueError("Invalid platform. Choose 'bluebear' or 'mac'.")
    
    return {
        "rds_dir": rds_dir,
        "epoched_dir": op.join(rds_dir, "derivatives/meg/sensor/epoched-7min50"),
        "info_dir": op.join(rds_dir, "dataman/data_information"),
        "fs_sub_dir": op.join(rds_dir, "cc700/mri/pipeline/release004/BIDS_20190411/anat"),
        "filtered_epo_dir": op.join(rds_dir, "derivatives/meg/sensor/filtered"),
        "meg_source_dir": op.join(rds_dir, "derivatives/meg/source/freesurfer"),
    }

def load_subjects(info_dir):
    """
    Load the list of subjects with good preprocessed data.
    """
    good_sub_sheet = op.join(info_dir, "demographics_goodPreproc_subjects.csv")
    good_subject_pd = pd.read_csv(good_sub_sheet)
    return good_subject_pd.set_index("Unnamed: 0")

def construct_paths(subject_id, paths, fr_band, space):
    """
    Construct file paths for filtered data, covariance matrices, and forward models.
    note that space can be "vol" or "surf"
    """
    fs_sub = f"sub-CC{subject_id}_T1w"
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])
    base_name = op.join(paths["filtered_epo_dir"], f"{fs_sub[:-4]}_")
    
    return {
        "mag_filtered": base_name + f"mag_{fr_band}-epo.fif",
        "grad_filtered": base_name + f"grad_{fr_band}-epo.fif",
        "mag_cov": op.join(deriv_folder, f"{fs_sub[:-4]}_mag_cov_{fr_band}.fif"),
        "grad_cov": op.join(deriv_folder, f"{fs_sub[:-4]}_grad_cov_{fr_band}.fif"),
        "forward_model": op.join(deriv_folder, f"{fs_sub[:-4]}_fwd-{space}.fif"),
        "mag_plot_fname": op.join(deriv_folder, f"{subject_id}_mag_{fr_band}_{space}.png"),
        "grad_plot_fname": op.join(deriv_folder, f"{subject_id}_grad_{fr_band}_{space}.png"),
    }

def save_plot(stc, file_name, space, **kwargs):
    """
    Save plots of source estimates for visualization.
    """
    if space == "vol":
        img = stc.plot(mode="stat_map", clim='auto', **kwargs)
    elif space == "surf":
        img = stc.plot(smoothing_steps=7, **kwargs)

    img.save_image(file_name)

def process_subject(subject_id, fr_band, space, paths):
    """
    Process a single subject for a given frequency band and space.
    """
    paths_subject = construct_paths(subject_id, paths, fr_band, space)
    reg = 0.01  # defined here for easier modifications

    print(f"Processing subject: {subject_id}, Frequency band: {fr_band}, Space: {space}")
    
    # Load filtered epochs and covariance matrices
    mags = mne.read_epochs(paths_subject["mag_filtered"], preload=True, proj=False)
    grads = mne.read_epochs(paths_subject["grad_filtered"], preload=True, proj=False)
    common_cov_mag = mne.read_cov(paths_subject["mag_cov"], verbose=None)
    common_cov_grad = mne.read_cov(paths_subject["grad_cov"], verbose=None)
    
    # Compute rank
    rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind="relative", proj=False)
    rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind="relative", proj=False)
    
    # Read forward model
    forward = mne.read_forward_solution(paths_subject["forward_model"])
    
    print('Making filter and apply LCMV')
    # Create and apply LCMV for magnetometers
    filters_mag = make_lcmv(
                    mags.info,
                    forward,
                    common_cov_mag,
                    reg=reg,
                    noise_cov=None, 
                    rank=rank_mag,
                    pick_ori="max-power",
                    reduce_rank=True,
                    depth=None,
                    inversion='matrix',
                    weight_norm="unit-noise-gain",
                    )
    stc_mag = apply_lcmv_cov(common_cov_mag, filters_mag)
    
    # Create and apply LCMV for gradiometers
    filters_grad = make_lcmv(
                    grads.info,
                    forward,
                    common_cov_grad,
                    reg=reg,
                    noise_cov=None, 
                    rank=rank_grad,
                    pick_ori="max-power",
                    reduce_rank=True,
                    depth=None,
                    inversion='matrix',
                    weight_norm="unit-noise-gain",
                    )
    stc_grad = apply_lcmv_cov(common_cov_grad, filters_grad)
    
    # Save plots
    plot_kwargs = dict(
        src=forward["src"],
        subject=paths_subject["fs_sub"],
        subjects_dir=paths["fs_sub_dir"],
        initial_time=0.087,
        verbose=True,
    )
    
    save_plot(stc_mag, paths_subject["grad_plot_fname"], **plot_kwargs)
    save_plot(stc_grad, paths_subject["grad_plot_fname"], **plot_kwargs)

def main():
    """
    Run the processing for all subjects and all frequency bands.
    """
    platform = "mac"  # Change to your platform
    paths = setup_paths(platform)
    good_subjects = load_subjects(paths["info_dir"])
    frequency_bands = {"delta": (1, 4), 
                       "theta": (4, 8), 
                       "alpha": (8, 12), 
                       "beta": (12, 30), 
                       "gamma": (30, 60)}
    space = "vol"  # Adjust spaces as needed- "vol" or "surf"
    
    for subject_id in good_subjects.index:
        for fr_band, (fmin, fmax) in frequency_bands.items():
            try:
                process_subject(subject_id, fr_band, space, paths)
            except Exception as e:
                print(f"Error processing subject {subject_id}, band {fr_band}, space {space}: {e}")

if __name__ == "__main__":
    main()
