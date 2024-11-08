"""
===============================================
S02. Using beamformer to localize oscillatory 
power modulations

This script uses DICS to localize 
oscillatory power moduations based on spatial
filtering (DICS: in frequency domain). 
multitaper csd and the epochs have been 
prepared in S02a and will be read in this 
script.

written by Tara Ghafari
==============================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd

import mne
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import read_csd

# subject info 
subjectID = '120469'  # FreeSurfer subject name - will go in the below for loop
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

space = 'volume' # which space to use, surface or volume?
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?

meg_extension = '.fif'
meg_suffix = 'meg'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_vol_suffix = 'fwd-vol'
fwd_surf_suffix = 'fwd-surf'
mag_epoched_extension = 'mag_epod-epo'
grad_epoched_extension = 'grad_epod-epo'
csd_extension = '.h5'
mag_csd_extension = f'mag_csd_multitaper_{fr_band}'
grad_csd_extension = f'grad_csd_multitaper_{fr_band}'
mag_stc_extension = f'mag_stc_multitaper_{fr_band}'
grad_stc_extension = f'grad_stc_multitaper_{fr_band}'
label_fname = 'aparc+aseg.mgz'

platform = 'bluebear'  # are you running on bluebear or mac?
# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx'

info_dir = op.join(rds_dir, 'dataman/data_information')
# good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# # Read only data from subjects with good preprocessed data
# good_subject_pd = pd.read_csv(good_sub_sheet)
# good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/epoched-1sec')
fwd_vol_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_vol_suffix + meg_extension)
fwd_surf_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_surf_suffix + meg_extension)
label_fpath = op.join(fs_sub_dir, f'{fs_sub}/mri', label_fname)

# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # try:
    #     print(f'Reading subject # {i}')

mag_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_epoched_extension + meg_extension)
grad_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_epoched_extension + meg_extension)
mag_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_csd_extension + csd_extension)
grad_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_csd_extension + csd_extension)

deriv_mag_stc_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_stc_extension)
deriv_grad_stc_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_stc_extension)

print('Reading forward model')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

print('Source reconstruction on magnetometers and gradiometers separately')
mags = mne.read_epochs(mag_epoched_fname, preload=True, verbose=True, proj=False)  
grads = mne.read_epochs(grad_epoched_fname, preload=True, verbose=True, proj=False)  

print('Computing rank')
rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative', proj=False)
rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative', proj=False)

print('Reading CSD')
csd_mag = read_csd(mag_csd_fname)
csd_grad = read_csd(grad_csd_fname)

print('Making filter and apply DICS')
filters_mag = make_dics(mags.info, 
                     forward, 
                     csd_mag.mean() , # we don't have conditions to calculate common csd
                     noise_csd=None, 
                     reg=0.01,  # because reduce rank results in mne python computing a truncated pseudo-inverse we don't need regularisation (I think!)
                     pick_ori='max-power', 
                     reduce_rank=True, 
                     real_filter=True, 
                     rank=rank_mag, 
                     depth=None,
                     inversion='matrix',
                     weight_norm="unit-noise-gain") # "nai" or "unit-noise-gain" only work with reduce_rank=False and results in rubbish

stc_mag, freqs = apply_dics_csd(csd_mag.mean(), filters_mag) 

filters_grad = make_dics(grads.info, 
                     forward, 
                     csd_grad.mean() , 
                     noise_csd=None, 
                     reg=0.01, 
                     pick_ori='max-power', 
                     reduce_rank=True, 
                     real_filter=True, 
                     rank=rank_grad, 
                     depth=None,
                     inversion='matrix',
                     weight_norm="unit-noise-gain") # "nai" or "unit-noise-gain" only work with reduce_rank=False and results in rubbish

stc_grad, freqs = apply_dics_csd(csd_grad.mean(), filters_grad)

# Plot source results to confirm
stc_mag.plot(src=forward["src"],
            subject=fs_sub,  # the FreeSurfer subject name
            subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
            mode='stat_map', 
            verbose=True)

stc_grad.plot(src=forward["src"],
            subject=fs_sub,  # the FreeSurfer subject name
            subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
            mode='stat_map', 
            verbose=True)

stc_grad.plot_3d(src=forward["src"],
            subject=fs_sub,  # the FreeSurfer subject name
            subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
            time_viewer=True,
            verbose=True)

# Save the stcs
stc_mag.save(deriv_mag_stc_fname, overwrite=True)
stc_grad.save(deriv_grad_stc_fname, overwrite=True)