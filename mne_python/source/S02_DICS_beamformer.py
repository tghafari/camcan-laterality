"""
===============================================
S02. Using beamformer to localize oscillatory 
power modulations

This script uses DICS to localize 
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

import os
import os.path as op
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
from mne.cov import compute_covariance
from mne.beamformer import make_dics, apply_dics_csd
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
subjectID = '220843'  # FreeSurfer subject name
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
n_fft = int(2 * epochs.info['sfreq'])  # used for psd and multitaper csd
epochspectrum = calculate_spectral_power(epochs, n_fft=n_fft, fmin=1, fmax=120)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250
# epochspectrum.plot()
# epochspectrum.plot_topomap(bands={fr_band:(fmin, fmax)}, ch_type="grad", normalize=True)

# Epoch the one long epoch of resting state data to be able to run csd_multitaper
for epochs_data in epochs:
    raw_epoch = mne.io.RawArray(epochs_data, epochs.info)
    epoched_epochs = mne.make_fixed_length_epochs(raw_epoch, duration=5, preload=True)


print('calculating the covariance matrix')
# Compute rank - should be similar to OSL, but double check with Mats
"""computing dics separately for mags and grads as noise_csd can only be None if
data is not mixed."""
mags = epoched_epochs.copy().pick("mag")
grads = epoched_epochs.copy().pick("grad")

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

print('Calculating the cross-spectral density matrices for the alpha band')
csd_mag = csd_multitaper(mags, 
                         fmin=fmin, 
                         fmax=fmax, 
                         tmin=mags.tmin, 
                         tmax=mags.tmax, 
                         bandwidth=3, 
                        #  n_fft=n_fft,
                         low_bias=True, 
                         verbose=False, 
                         n_jobs=-1)
csd_grad = csd_multitaper(grads,
                          fmin=fmin, 
                          fmax=fmax, 
                          tmin=grads.tmin, 
                          tmax=grads.tmax, 
                          bandwidth=3, 
                          low_bias=True, 
                          verbose=False, 
                          n_jobs=-1)

# Plot csds for double checking 
plot_dict = {
    "multitaper csd: magnetometers": csd_mag,
    "multitaper csd: gradiometers": csd_grad,
}
for title, csd in plot_dict.items():
    fig, = csd.mean().plot()
    fig.suptitle(title)

rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative')
rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative')

print('Derive and apply spatial filters')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

filters_mag = make_dics(mags.info, 
                     forward, 
                     csd_mag.mean() , 
                     noise_csd=None, 
                     reg=0, 
                     pick_ori='max-power', 
                     reduce_rank=True, 
                     real_filter=True, 
                     rank=rank_mag, 
                     depth=0)
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
stc_grad, freqs = apply_dics_csd(csd_grad.mean(), filters_grad)

# Plot source results to confirm
stc_mag.plot(src=forward["src"],
            subject=fs_sub,  # the FreeSurfer subject name
            subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
            mode='stat_map', 
            verbose = True)
stc_grad.plot(src=forward["src"],
            subject=fs_sub,  # the FreeSurfer subject name
            subjects_dir=fs_sub_dir,  # the path to the directory containing the FreeSurfer subjects reconstructions.
            mode='stat_map', 
            verbose = True)