"""
===============================================
S03. Using beamformer to localize oscillatory 
power modulations

This script uses LCMV to localize 
oscillatory power moduations based on spatial
filtering (DICS: in frequency domain). 

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
import pandas as pd

import mne
from mne.beamformer import make_lcmv, apply_lcmv_cov


# subject info 
subjectID = '120264'  # FreeSurfer subject name - will go in the below for loop
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

space = 'volume' # which space to use, surface or volume?
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?

meg_extension = '.fif'
meg_suffix = 'meg'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_vol_suffix = 'fwd-vol'
fwd_surf_suffix = 'fwd-surf'
mag_filtered_extension = f'mag_{fr_band}-epo'
grad_filtered_extension = f'grad_{fr_band}-epo'
mag_cov_extension = f'mag_cov_{fr_band}'
grad_cov_extension = f'grad_cov_{fr_band}'

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
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/filtered')
fwd_vol_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_vol_suffix + meg_extension)
fwd_surf_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_surf_suffix + meg_extension)


# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # Read forward model

mag_filtered_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_filtered_extension + meg_extension)
grad_filtered_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_filtered_extension + meg_extension)
mag_cov_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_cov_extension + meg_extension)
grad_cov_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_cov_extension + meg_extension)

    # try:
    #     print(f'Reading subject # {i}')
                    
print('Reading epochs- magnetometers and gradiometers separately')
mags = mne.read_epochs(mag_filtered_fname, preload=True, verbose=True, proj=False)  
grads = mne.read_epochs(grad_filtered_fname, preload=True, verbose=True, proj=False)  

print('Compute rank of mags and grads')
rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative', proj=False)
rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative', proj=False)

print('Reading common covariance matrices')
common_cov_mag = mne.read_cov(mag_cov_fname, verbose=None)
common_cov_grad = mne.read_cov(grad_cov_fname, verbose=None)

print('Reading forward model')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

print('Making filter and apply LCMV')
filters_mag = make_lcmv(mags.info, 
                    forward, 
                    common_cov_mag, 
                    reg=0.05,  # OSL:reg=0, Ole: 0.05
                    noise_cov=None,  # OSL: None
                    rank=rank_mag,  
                    pick_ori="max-power",  # OSL:pick_ori="max-power-pre-weight-norm"  isn't an original parameter, Ole: 'max-power'
                    reduce_rank=True,
                    depth=None,  # How to weight (or normalize) the forward using a depth prior.
                    inversion='matrix',
                    weight_norm="unit-noise-gain" # "unit-noise-gain" OSL:weight_norm="unit-noise-gain-invariant", Ole: 'unit-noise-gain', 'nai' when no empty room
                    ) 
stc_mag = apply_lcmv_cov(common_cov_mag, filters_mag)

filters_grad = make_lcmv(grads.info, 
                    forward, 
                    common_cov_grad, 
                    reg=0.05,  # OSL:reg=0, Ole: 0.05
                    noise_cov=None,  # OSL: None
                    rank=rank_grad,  
                    pick_ori="max-power",  # OSL:pick_ori="max-power-pre-weight-norm"  isn't an original parameter, Ole: 'max-power'
                    reduce_rank=True,
                    depth=None,  # How to weight (or normalize) the forward using a depth prior.
                    inversion='matrix',
                    weight_norm="unit-noise-gain" # "unit-noise-gain" OSL:weight_norm="unit-noise-gain-invariant", Ole: 'unit-noise-gain', 'nai' when no empty room
                    ) 
stc_grad = apply_lcmv_cov(common_cov_grad, filters_grad)

# Plot source results to confirm
initial_time = 0.087

if space == 'volume':
    kwargs = dict(
        src=forward["src"],
        subject=fs_sub,  # the FreeSurfer subject name
        subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
        initial_time=initial_time,
        verbose=True,
        )
    stc_mag.plot(mode="stat_map", clim='auto', **kwargs)
    stc_grad.plot(mode="stat_map", clim='auto', **kwargs)
elif space == 'surface':
    lims = [0.3, 0.45, 0.6]
    brain = stc_grad.plot(
        src=forward["src"],
        subject=fs_sub, 
        subjects_dir=fs_sub_dir,
        initial_time=initial_time,
        smoothing_steps=7,
    )
