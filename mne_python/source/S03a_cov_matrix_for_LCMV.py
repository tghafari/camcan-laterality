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
import matplotlib.pyplot as plt

import mne
from mne.cov import compute_covariance

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
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?
plotting = True  # if you'd like to plot the outputs or not

meg_extension = '.fif'
meg_suffix = 'meg'
mag_filtered_extension = f'mag_{fr_band}-epo'
grad_filtered_extension = f'grad_{fr_band}-epo'
mag_cov_extension = f'mag_cov_{fr_band}'
grad_cov_extension = f'grad_cov_{fr_band}'

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
info_dir = op.join(rds_dir, 'dataman/data_information')
fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/filtered')

# Read only data from subjects with good preprocessed data
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

if fr_band == 'delta':
   fmin = 1
   fmax = 4

elif fr_band == 'theta':
   fmin = 4
   fmax = 8

elif fr_band == 'alpha':
   fmin = 8
   fmax = 12

elif fr_band == 'beta':
   fmin = 12
   fmax = 30

elif fr_band == 'gamma':
    fmin = 30
    fmax = 60

else:
    raise ValueError("Error: 'fr_band' value not valid")

# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # Read forward model

epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
epoched_fif = op.join(epoched_dir, epoched_fname)

deriv_mag_filtered_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_filtered_extension + meg_extension)
deriv_grad_filtered_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_filtered_extension + meg_extension)
deriv_mag_cov_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_cov_extension + meg_extension)
deriv_grad_cov_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_cov_extension + meg_extension)

    # try:
    #     print(f'Reading subject # {i}')
                    
epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
if plotting:
    epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250
    epochspectrum.plot()
    epochspectrum.plot_topomap(bands={fr_band:(fmin, fmax)}, ch_type="grad", normalize=True)

print(f"Filtering to {fr_band} and picking grads and mags")
"""computing lcmv separately for mags and grads as noise_covariance can only be None if
data is not mixed."""
mags = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("mag")
grads = epochs.copy().filter(l_freq=fmin, h_freq=fmax).pick("grad")

# Save mags and grads for later use
mags.save(deriv_mag_filtered_fname)
grads.save(deriv_grad_filtered_fname)

print('Calculating rank and the covariance matrix')
"""compute rank again in the main LCMV file."""
rank_mag = mne.compute_rank(mags, tol=1e-6, tol_kind='relative')
common_cov_mag = compute_covariance(mags, 
                                method='empirical',
                                rank=rank_mag,
                                n_jobs=4,
                                verbose=True)

rank_grad = mne.compute_rank(grads, tol=1e-6, tol_kind='relative')
common_cov_grad = compute_covariance(grads, 
                                method='empirical',
                                rank=rank_grad,
                                n_jobs=4,
                                verbose=True)
if plotting:
    common_cov_mag.plot(mags.info)
    common_cov_grad.plot(grads.info)

    topo_mag_cov = common_cov_mag.plot_topomap(mags.info)
    topo_mag_cov.suptitle("Common covariance: Magnetometers")
    topo_grad_cov = common_cov_grad.plot_topomap(grads.info)
    topo_grad_cov.suptitle("Common covariance: Gradiometers")

# Save the covariance matrices
common_cov_mag.save(deriv_mag_cov_fname, overwrite=True)
common_cov_grad.save(deriv_grad_cov_fname, overwrite=True)
