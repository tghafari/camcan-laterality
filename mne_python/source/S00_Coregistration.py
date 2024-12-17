# -*- coding: utf-8 -*-
"""
===============================================
Coregistration and Preparing Transformation File
===============================================

This script coregisters MEG data with MRI and generates a `trans` file, 
which is necessary for BIDS conversion and source-based analysis.

Steps:
1. Create Boundary Element Model (BEM) for each subject.
2. Perform coregistration:
   - Automatic coregistration (preferred for CamCAN data).
   - Manual coregistration (optional).
3. Save coregistration results and transformation files.

The script supports batch processing for multiple subjects using automatic coregistration 
and can also handle manual coregistration for specific subjects.

To run this code on all subjects, define the number
of subjects in main definition and then type
"python S00_coregistration.py" in an mne environment.

Written by: Tara Ghafari
t.ghafari@bham.ac.uk
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan-1'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'rds_dir': rds_dir,
        'epoched_dir': op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50'),
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
    }
    return paths

def load_subjects(good_sub_sheet):
    """Load subject IDs from a CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # Set subject IDs as the index
    return good_subject_pd

def create_bem(subject, fs_dir, bem_fname):
    """Create and save a BEM solution for a subject."""
    conductivity = (0.3,)  # Single-layer conductivity model
    model = mne.make_bem_model(subject=subject, subjects_dir=fs_dir, ico=4, conductivity=conductivity)
    bem = mne.make_bem_solution(model)
    mne.write_bem_solution(bem_fname, bem, overwrite=True)
    return bem

def perform_coregistration(info, fs_sub, fs_dir, auto=True):
    """Perform coregistration (automatic or manual)."""
    fiducials = "estimated" if auto else None
    coreg = mne.coreg.Coregistration(info, subject=fs_sub, subjects_dir=fs_dir, fiducials=fiducials)
    if auto:
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=6, nasion_weight=1.0)
        tr1 = coreg.trans
        coreg.omit_head_shape_points(distance=5 / 1000)  # Omit points farther than 5mm
        coreg.fit_icp(n_iterations=20, nasion_weight=10.0)
        tr2 = coreg.trans

        # Before saving the trans file check if second fit icp is doing something.
        print(f"{tr1} \n{tr2}")  # Check if the transformation matrix is valid

    else:
        mne.gui.coregistration(subject=fs_sub, subjects_dir=fs_dir)
    return coreg

def save_coregistration_results(coreg, info, coreg_figname, trans_fname, coreg_plot_kwargs):
    """Save coregistration results: screenshots and transformation files."""
    # Visualize final coregistration
    coreg_fig = mne.viz.plot_alignment(info, trans=coreg.trans, **coreg_plot_kwargs)

   # Check the distances after coregistration
    try:
        dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
        print(f"Distances for subject : mean={np.mean(dists):.2f}, min={np.min(dists):.2f}, max={np.max(dists):.2f}")
    except Exception as e:
        print(f"Error computing distances for subject : {e}")

    text = f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm\n{np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
    # Save screenshot
    screenshot = coreg_fig.plotter.screenshot()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot, origin='upper')
    ax.set_axis_off()
    fig.text(0.1, 0.1, text, fontsize=12, ha='left', va='center', color='white', bbox=dict(facecolor='black', alpha=0.5))
    fig.tight_layout()
    fig.savefig(coreg_figname, dpi=150)
    plt.close(fig)

    coreg_fig.plotter.close()

    # Save transformation file
    mne.write_trans(trans_fname, coreg.trans, overwrite=True)

def process_subject(subjectID, paths, coreg_type='auto'):
    """
    Process a single subject: create BEM, perform coregistration, and save results.
    Skips processing if coregistration file already exists.

    Parameters:
    -----------
    subjectID : str
        Subject ID to process.
    paths : dict
        Dictionary containing paths for directories like 'rds_dir', 'fs_sub_dir', and 'epoched_dir'.
    coreg_type : str, optional
        Type of coregistration ('auto' or 'manual'), default is 'auto'.

    Returns:
    --------
    None
    """

    try:
        print(f"Processing subject {subjectID}...")

        # Define subject-specific paths and filenames
        fs_sub = f"sub-CC{subjectID}_T1w"
        deriv_folder = op.join(paths['rds_dir'], 'derivatives/meg/source/freesurfer', fs_sub[:-4])

        if not op.exists(deriv_folder):
            os.makedirs(deriv_folder)

        trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_coreg-trans.fif')

        # Check if coregistration file already exists
        if op.exists(trans_fname):
            print(f"Coregistration already done for subject {subjectID}, skipping...")
            return

        bem_fname = trans_fname.replace('coreg-trans', 'bem-sol')
        coreg_figname = bem_fname.replace('bem-sol', 'final_coreg').replace('.fif', '.png')

        # Create BEM
        _ = create_bem(fs_sub, paths['fs_sub_dir'], bem_fname)

        # Read MEG info
        epoched_fname = f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif'
        epoched_fif = op.join(paths['epoched_dir'], epoched_fname)
        info = mne.read_epochs(epoched_fif, preload=True, verbose=True).info

        # Coregistration
        coreg = perform_coregistration(info, fs_sub, paths['fs_sub_dir'], 
                                    auto=(coreg_type == 'auto'))

        # Save results
        plot_kwargs = dict(subject=fs_sub, subjects_dir=paths['fs_sub_dir'], 
                        surfaces="head-dense", dig=True, eeg=[], meg='sensors', 
                        show_axes=True, coord_frame='meg')
        save_coregistration_results(coreg, info, coreg_figname, 
                                    trans_fname, plot_kwargs)

        print(f"Subject {subjectID} processed successfully.")

    except Exception as e:
        print(f"Error processing subject {subjectID}: {e}")

def main():
    platform = 'mac'  # Change to 'bluebear' if running on BlueBear
    coreg_type = 'auto'  # Change to 'manual' for manual coregistration
    paths = setup_paths(platform)
    good_subject_pd = load_subjects(paths['good_sub_sheet'])

    for subjectID in good_subject_pd.index[6:40]:
        process_subject(subjectID, paths, coreg_type=coreg_type)


if __name__ == "__main__":
    main()