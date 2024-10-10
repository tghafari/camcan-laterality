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
adapted from flux pipeline
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
subjectID = '121795'  # FreeSurfer subject name - will go in the below for loop
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

space = 'volume' # which space to use, surface or volume?
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?

meg_extension = '.fif'
meg_suffix = 'meg'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_vol_suffix = 'fwd-vol'
fwd_surf_suffix = 'fwd-surf'
mag_epoched_extension = 'mag_epoched-epo'
grad_epoched_extension = 'grad_epoched-epo'
csd_extension = '.h5'
mag_csd_extension = f'mag_csd_multitaper_{fr_band}'
grad_csd_extension = f'grad_csd_multitaper_{fr_band}'

platform = 'mac'  # are you running on bluebear or mac?
# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/epoched-1sec')
fwd_vol_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_vol_suffix + meg_extension)
fwd_surf_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_surf_suffix + meg_extension)


# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # try:
    #     print(f'Reading subject # {i}')

mag_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_epoched_extension + meg_extension)
grad_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_epoched_extension + meg_extension)
mag_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_csd_extension + csd_extension)
grad_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_csd_extension + csd_extension)

print('Reading epochs- magnetometers and gradiometers separately')
mags = mne.read_epochs(mag_epoched_fname, preload=True, verbose=True, proj=False)  
grads = mne.read_epochs(grad_epoched_fname, preload=True, verbose=True, proj=False)  

print('Compute rank of mags and grads')
rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative', proj=False)
rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative', proj=False)

print('Reading csds')
csd_mag = read_csd(mag_csd_fname)
csd_grad = read_csd(grad_csd_fname)

print('Reading forward model')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

print('Making filter and apply DICS')
filters_mag = make_dics(mags.info, 
                     forward, 
                     csd_mag.mean() , # we don't have conditions to calculate common csd
                     noise_csd=None, 
                     reg=0,  # because reduce rank results in mne python computing a truncated pseudo-inverse we don't need regularisation (I think!)
                     pick_ori='max-power', 
                     reduce_rank=True, 
                     real_filter=True, 
                     rank=rank_mag, 
                     depth=0)
                    #  weight_norm="unit-noise-gain")  # "unit-noise-gain" or 'nai', defaults to None where The unit-gain LCMV beamformer will be computed
stc_mag, freqs = apply_dics_csd(csd_mag.mean(), filters_mag) 

filters_grad = make_dics(grads.info, 
                     forward, 
                     csd_grad.mean() , 
                     noise_csd=None, 
                     reg=0, 
                     pick_ori='max-power', 
                     reduce_rank=True, 
                     real_filter=True, 
                     rank=rank_grad, 
                     depth=0)
                     #=weight_norm="nai" or "unit-noise-gain" only work with reduce_rank=False and results in rubbish
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