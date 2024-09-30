# -*- coding: utf-8 -*-
"""
===============================================
S01. Constructing the forward model

This script constructs the head model to be used
as a lead field matrix, in source modelling. 
This is based on the T1 MRI. This model will 
be aligned to head position of the subject in the
MEG system. 

written by Tara Ghafari
adapted from flux pipeline
==============================================
ToDos:
    1) 
    
Issues/ contributions to community:
    1) 
    
Questions:
    1)

Notes:
    Step 1: Computing source space
    Step 2: Forward model

"""

import os.path as op
import os
import pandas as pd

import mne


# subject info 
subjectID = '120469'  # FreeSurfer subject name
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

platform = 'mac'  # are you running on bluebear or windows or mac?
# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# OSL settings
space = 'volume'  # what to use for source modeling? surface or volume- from OSL
gridstep=8  # from OSL
mindist=4.0

# Specific file names
meg_extension = '.fif'
meg_suffix = 'meg'
trans_suffix = 'coreg-trans'
bem_suffix = 'bem-sol'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_suffix = 'fwd'

fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])

trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + trans_suffix + meg_extension)
bem_fname = trans_fname.replace(trans_suffix, bem_suffix)  

surf_fname = trans_fname.replace(trans_suffix, surf_suffix)  # only used for suffices that are not recognizable to bids 
vol_fname = surf_fname.replace(surf_suffix, vol_suffix)  # save in the bids folder
bem_fname = surf_fname.replace(surf_suffix, bem_suffix)  
fwd_fname = surf_fname.replace(surf_suffix, fwd_suffix)

surface = op.join(fs_sub_dir, fs_sub, 'bem', 'inner_skull.surf')

# Step 1: Compute source space according to BEM 
""" The source space is defined by a grid covering brain volume.
--volume vs surface is for next steps of analysis"""

if space == 'surface':
    # Surface-based source space
    spacing = 'oct6'  # 4098 sources per hemisphere, 4.9 mm spacing
    src = mne.setup_source_space(subject=fs_sub, 
                                 subjects_dir=fs_sub_dir, 
                                 spacing=spacing, 
                                 add_dist='patch')
    mne.write_source_spaces(surf_fname, src, overwrite=True)
    
elif space == 'volume':
    # Volumetric source space (BEM required)
    src = mne.setup_volume_source_space(subject=fs_sub,
                                        subjects_dir=fs_sub_dir,
                                        surface=surface,
                                        mri='T1.mgz',
                                        verbose=True,
                                        gridstep=gridstep, 
                                        mindist=mindist)
    mne.write_source_spaces(vol_fname, src, overwrite=True)
    
# Visualize source space and BEM
mne.viz.plot_bem(subject=fs_sub, 
                 subjects_dir=fs_sub_dir,
                 brain_surfaces='white', 
                 src=src, 
                 orientation='coronal')

# Visualize sources in 3D space
if space == 'surface':
    fig = mne.viz.plot_alignment(subject=fs_sub, 
                                 subjects_dir=fs_sub_dir,
                                 trans=trans_fname, 
                                 surfaces='white',
                                 coord_frame='head', 
                                 src=src)
    mne.viz.set_3d_view(fig, 
                        azimuth=173.78, 
                        elevation=101.75,
                        distance=0.35, 
                        focalpoint=(-0.03, 0.01, 0.03))
    
# Step 2: Construct the forward model
""" 
The last step is to construct the forward model by assigning a lead-field 
to each source location in relation to the head position with respect to 
the sensors. This will result in the lead-field matrix.
"""
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
epoched_fif = op.join(epoched_dir, epoched_fname)

    # try:
    #     print(f'Reading subject # {i}')
                    
info = mne.read_epochs(epoched_fif, preload=True, verbose=True).info  # one 7min50sec epochs

fwd = mne.make_forward_solution(info, 
                                trans_fname, 
                                src, 
                                bem=bem_fname,
                                meg=True, 
                                eeg=False, 
                                verbose=True,) 
                                # mindist=5.) # could be 2.5

mne.write_forward_solution(fwd_fname, fwd)

# Print some details
print(f'\nNumber of vertices: {fwd["src"]}')
leadfield = fwd['sol']['data']  # size of leadfield
print("\nLeadfield size: %d sensors x %d dipoles" %leadfield.shape)


























