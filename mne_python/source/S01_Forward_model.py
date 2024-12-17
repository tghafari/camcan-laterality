# -*- coding: utf-8 -*-
"""
===============================================
S01. Constructing the forward model

This script constructs the head model (lead field
 matrix) for source modeling.
It aligns the model to the subject's head 
position in the MEG system.

written by Tara Ghafari
adapted from flux pipeline
===============================================
"""

import os
import os.path as op
import pandas as pd
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
        'meg_source_dir': op.join(rds_dir, 'derivatives/meg/source/freesurfer'),
    }
    return paths


def load_subjects(good_sub_sheet):
    """Load subject IDs from the CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    return good_subject_pd.set_index('Unnamed: 0')


def construct_paths(subjectID, paths):
    """Construct file paths for a given subject."""
    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])
    trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_coreg-trans.fif')
    bem_fname = trans_fname.replace('coreg-trans', 'bem-sol')
    surf_src_fname = trans_fname.replace('coreg-trans', 'surf-src')
    vol_src_fname = surf_src_fname.replace('surf-src', 'vol-src')
    fwd_surf_fname = surf_src_fname.replace('surf-src', 'fwd-surf')
    fwd_vol_fname = surf_src_fname.replace('surf-src', 'fwd-vol')

    # Define FreeSurfer surface path
    inner_skull_fname = op.join(paths['fs_sub_dir'], fs_sub, 'bem', 'inner_skull.surf')

    return {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        'trans_fname': trans_fname,
        'bem_fname': bem_fname,
        'surf_src_fname': surf_src_fname,
        'vol_src_fname': vol_src_fname,
        'fwd_surf_fname': fwd_surf_fname,
        'fwd_vol_fname': fwd_vol_fname,
        'inner_skull_fname': inner_skull_fname
    }


def check_existing_files(paths_dict, space):
    """Check if forward solution already exists for a subject."""
    if space == 'surface' and op.exists(paths_dict['fwd_surf_fname']):
        print("Forward solution already exists for surface space. Skipping...")
        return True
    elif space == 'volume' and op.exists(paths_dict['fwd_vol_fname']):
        print("Forward solution already exists for volume space. Skipping...")
        return True
    return False


def compute_source_space(space, fs_sub, fs_sub_dir, surface_path, mindist=5.0):
    """Compute the source space based on the specified type."""
    if space == 'surface':
        spacing = 'oct6'  # 4098 sources per hemisphere, ~4.9 mm spacing
        src = mne.setup_source_space(subject=fs_sub, 
                                     subjects_dir=fs_sub_dir, 
                                     spacing=spacing,
                                     add_dist='patch')
    elif space == 'volume':
        src = mne.setup_volume_source_space(subject=fs_sub,
                                            subjects_dir=fs_sub_dir,
                                            surface=surface_path,
                                            mri='T1.mgz',
                                            mindist=mindist,
                                            verbose=True)
    return src


def process_subject(subjectID, paths, space='volume'):
    """
    Process a subject: compute source space, and forward solution.
    Skips processing if files already exist.
    """
    print(f"Processing subject {subjectID}...")
    subject_paths = construct_paths(subjectID, paths)

    # Skip if forward solution already exists
    if check_existing_files(subject_paths, space):
        return

    # Step 1: Compute Source Space
    src = compute_source_space(space, 
                               subject_paths['fs_sub'], 
                               paths['fs_sub_dir'], 
                               subject_paths['inner_skull_fname'])

    # Step 2: Load MEG Info
    epoched_fname = op.join(paths['epoched_dir'], f'sub-CC{subjectID}_ses-rest_task-rest_megtransdef_epo.fif')
    info = mne.read_epochs(epoched_fname, preload=True).info

    # Step 3: Compute Forward Solution
    """ 
    The last step is to construct the forward model by assigning a lead-field 
    to each source location in relation to the head position with respect to 
    the sensors. This will result in the lead-field matrix.
    """
    fwd = mne.make_forward_solution(info,
                                    trans=subject_paths['trans_fname'],
                                    src=src,
                                    bem=subject_paths['bem_fname'],
                                    meg=True,
                                    eeg=False,
                                    verbose=True)

    # Save Source Space and Forward Solution
    if space == 'surface':
        mne.write_source_spaces(subject_paths['surf_src_fname'], src, overwrite=True)
        mne.write_forward_solution(subject_paths['fwd_surf_fname'], fwd, overwrite=True)

    elif space == 'volume':
        mne.write_source_spaces(subject_paths['vol_src_fname'], src, overwrite=True)
        mne.write_forward_solution(subject_paths['fwd_vol_fname'], fwd, overwrite=True)

    print(f"Subject {subjectID} forward model complete.")


def main():
    platform = 'mac'  # 'bluebear' or 'mac'
    space = 'volume'  # 'surface' or 'volume'
    paths = setup_paths(platform)

    # Load subjects with good preprocessing
    good_subject_pd = load_subjects(paths['good_sub_sheet'])

    for subjectID in good_subject_pd.index:
        try:
            process_subject(subjectID, paths, space=space)
        except Exception as e:
            print(f"Error processing subject {subjectID}: {e}")


if __name__ == "__main__":
    main()

