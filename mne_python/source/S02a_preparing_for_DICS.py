"""
===============================================
S02a. Preparing for DICS

This script reads in the epochs from resting 
state data and epoch them into shorter sections
and runs csd_multitaper on them.
Finally, the multitaper will be saved.

written by Tara Ghafari
==============================================
toDos:
    - plot csd topographically
"""

import os
import os.path as op
import numpy as np
import pandas as pd

import mne
from mne_bids import BIDSPath
from mne.time_frequency import csd_multitaper

def epoching_epochs(epoched_fif, epoched_epochs_duration):
    """This definition inputs the epoched data called epoched_fif and 
    epoch them into shorter epochs with epoched_epochs_duration.
    this is to reduce the computation time of csd_multitaper."""

    print('Reading epochs')
    epochs = mne.read_epochs(epoched_fif, 
                             preload=True, 
                             verbose=True, 
                             proj=False)  # one 7min50sec epochs
    print(f'epoching to {epoched_epochs_duration} seconds')
    for epochs_data in epochs:
        raw_epoch = mne.io.RawArray(epochs_data, 
                                    epochs.info)
        epoched_epochs = mne.make_fixed_length_epochs(raw_epoch, 
                                                      duration=epoched_epochs_duration, 
                                                      overlap=0.5, 
                                                      preload=True)
    return epoched_epochs

# subject info 
subjectID = '120264'  # FreeSurfer subject name
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?

meg_extension = '.fif'
meg_suffix = 'meg'
mag_epoched_extension = 'mag_epod-epo'
grad_epoched_extension = 'grad_epod-epo'
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
fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/epoched-1sec')

# Read only data from subjects with good preprocessed data
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

if fr_band == 'delta':
   fmin = 1
   fmax = 4
   bandwidth = 1.
   epoched_epochs_duration = 1  # duration of the epoched epochs

elif fr_band == 'theta':
   fmin = 4
   fmax = 8
   bandwidth = 1.
   epoched_epochs_duration = 1  

elif fr_band == 'alpha':
   fmin = 8
   fmax = 12
   bandwidth = 1.
   epoched_epochs_duration = 1 

elif fr_band == 'beta':
   fmin = 12
   fmax = 30
   bandwidth = 1.
   epoched_epochs_duration = 1  

elif fr_band == 'gamma':
    fmin = 30
    fmax = 60
    bandwidth = 4.
    epoched_epochs_duration = 1  

else:
    raise ValueError("Error: 'fr_band' value not valid")

# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # Read forward model

epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
epoched_fif = op.join(epoched_dir, epoched_fname)

deriv_mag_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_epoched_extension + meg_extension)
deriv_grad_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_epoched_extension + meg_extension)
deriv_mag_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_csd_extension)
deriv_grad_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_csd_extension)

    # try:
    #     print(f'Reading subject # {i}')
epoched_epochs = epoching_epochs(epoched_fif, epoched_epochs_duration)

# Pick mags and grads for later analyses
"""computing dics separately for mags and grads as noise_csd can only be None if
data is not mixed."""
mags = epoched_epochs.copy().pick("mag")
grads = epoched_epochs.copy().pick("grad")

# Save mags and grads for later use
mags.save(deriv_mag_epoched_fname, overwrite=True)
grads.save(deriv_grad_epoched_fname, overwrite=True)

print(f'Calculating the cross-spectral density matrices for the {fr_band} band')
csd_mag = csd_multitaper(mags, 
                         fmin=fmin, 
                         fmax=fmax, 
                         tmin=mags.tmin, 
                         tmax=mags.tmax, 
                         bandwidth=bandwidth, 
                         low_bias=True, 
                         verbose=False, 
                         n_jobs=-1)
csd_grad = csd_multitaper(grads,
                          fmin=fmin, 
                          fmax=fmax, 
                          tmin=grads.tmin, 
                          tmax=grads.tmax, 
                          bandwidth=bandwidth, 
                          low_bias=True, 
                          verbose=False, 
                          n_jobs=-1)

# Plot csds for double checking 
plot_dict = {
    "multitaper csd: magnetometers": csd_mag,
    "multitaper csd: gradiometers": csd_grad,
}
for title, csd in plot_dict.items():
    fig, = csd.mean().plot(mode='csd')
    fig.suptitle(title)

# Save the csds
csd_mag.save(deriv_mag_csd_fname, overwrite=True)
csd_grad.save(deriv_grad_csd_fname, overwrite=True)