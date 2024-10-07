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
from mne.cov import compute_covariance, compute_raw_covariance
from mne.beamformer import make_lcmv, apply_lcmv_cov, make_dics, apply_dics_csd
from mne.time_frequency import csd_multitaper

def calculate_spectral_power(epochs, n_fft, fmin, fmax):
    """
    The data are divided into sections being 2 s long. 
      (n_fft = 500 samples) with a 1 s overlap 
      (250 samples). This results in a 0.5 Hz 
      resolution Prior to calculating the FFT of each 
      section a Hamming taper is multiplied.
      n_fft=500, fmin=1, fmax=60"""
    
   # define constant parameters
    welch_params = dict(fmin=fmin, fmax=fmax, picks="meg", n_fft=n_fft, n_overlap=int(n_fft/2))

    # calculate power spectrum for right and left sensors separately
    """the returned array will have the same
      shape as the input data plus an additional frequency dimension"""
    epochspectrum = epochs.compute_psd(method='welch',  
                                        **welch_params,
                                        n_jobs=30,
                                        verbose=True)

    return epochspectrum

# subject info 
subjectID = '121795'  # FreeSurfer subject name
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

platform = 'mac'  # are you running on bluebear or windows or mac?
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
noise_dir = op.join(sub2ctx_dir, 'camcan/MEG1/meg_emptyroom') 
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Specific file names
meg_extension = '.fif'
meg_suffix = 'meg'
trans_suffix = 'coreg-trans'
bem_suffix = 'bem-sol'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_vol_suffix = 'fwd-vol'
fwd_surf_suffix = 'fwd-surf'

fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
fwd_vol_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_vol_suffix + meg_extension)
fwd_surf_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_surf_suffix + meg_extension)

space = 'volume' # which space to use, surface or volume?
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?
if fr_band == 'alpha':
   fmin = 7
   fmax = 13
   bandwidth = 2.

elif fr_band == 'gamma':
    fmin = 60
    fmax = 90
    bandwidth = 4.
else:
    raise ValueError("Error: 'fr_band' value not valid")
# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # Read forward model

epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
epoched_fif = op.join(epoched_dir, epoched_fname)
noise_fname = op.join('sub-CC' + str(subjectID), 'emptyroom', 'emptyroom_CC'+ str(subjectID) + meg_extension) 
noise_fif = op.join(noise_dir, noise_fname)
    # try:
    #     print(f'Reading subject # {i}')
                    
epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250
epochspectrum.plot()
epochspectrum.plot_topomap(bands={fr_band:(fmin, fmax)}, ch_type="grad", normalize=True)

print('calculating the covariance matrix')

# Compute rank - should be similar to OSL, but double check with Mats
"""computing lcmv separately for mags and grads as noise_covariance can only be None if
data is not mixed."""
mags = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("mag")
grads = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("grad")

rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative')
common_cov_mag = compute_covariance(mags, 
                                method='empirical',
                                rank=rank_mag,
                                n_jobs=4,
                                verbose=True)
common_cov_mag.plot(mags.info)


rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative')
common_cov_grad = compute_covariance(grads, 
                                method='empirical',
                                rank=rank_grad,
                                n_jobs=4,
                                verbose=True)
common_cov_grad.plot(grads.info)

"""this part is ignored for now
# print('Estimating noise covariance with the empty room data')
# noise_raw = mne.io.read_raw_fif(noise_fif, allow_maxshield=True, preload=True, verbose=True)
# noise_raw_filterd = noise_raw.copy().filter(l_freq=fmin, h_freq=fmax)
# noise_cov = compute_raw_covariance(noise_raw_filterd, 
#                                    tmin=0, #epochs.tmin
#                                    tmax=None, #epochs.tamx
#                                    tstep=0.2, 
#                                    reject=None, 
#                                    flat=None, 
#                                    picks=None, 
#                                    method='empirical', 
#                                    method_params=None, 
#                                    cv=3, 
#                                    scalings=None, 
#                                    n_jobs=None, 
#                                    return_estimators=False, 
#                                    reject_by_annotation=True, 
#                                    rank=rank, 
#                                    verbose=None)
"""
print('Derive and apply spatial filters')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

filters_mag = make_lcmv(mags.info, 
                    forward, 
                    common_cov_mag, 
                    reg=0.05,  # OSL:reg=0, Ole: 0.05
                    noise_cov=None,  # OSL: None
                    rank=rank_mag,  
                    pick_ori="max-power",  # OSL:pick_ori="max-power-pre-weight-norm"  isn't an original parameter, Ole: 'max-power'
                    reduce_rank=True,
                    depth=0,  # How to weight (or normalize) the forward using a depth prior.
                    inversion='matrix',
                    weight_norm="nai" # "unit-noise-gain" OSL:weight_norm="unit-noise-gain-invariant", Ole: 'unit-noise-gain' 
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
                    depth=0,  # How to weight (or normalize) the forward using a depth prior.
                    inversion='matrix',
                    weight_norm="nai" # "unit-noise-gain" OSL:weight_norm="unit-noise-gain-invariant", Ole: 'unit-noise-gain' 
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
        subject=fs_sub,  # the FreeSurfer subject name
        subjects_dir=fs_sub_dir,
        initial_time=initial_time,
        smoothing_steps=7,
    )
    # clim=dict(kind="value", pos_lims=lims)