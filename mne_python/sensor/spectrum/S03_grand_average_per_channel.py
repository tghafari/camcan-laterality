"""
===============================================
S03. Grand average per channel

The aim of this code is to find out if there's a 
systematic level of noise in the sensors across
all participants.

Thia code will plot the grand average psd for
each channel on the right side and plots the left
channels grand average on top of that for comparison
in a plot_topo.

This code will:
    1. read each participant's epochs
    2. calculates the spectra for each channel separately
    3. put the spectra of each channel of each subject in 
    one list (total a list of 306 columns (#sensors) by 
     590 rows (#subjects))
    4. loops over subjects and appends to the list
    5. calculates grand average for each column of that
    list
    6. plots grand average psd of right sensor and left 
    sensor on top of each other in half a plot_topo 


written by Tara Ghafari
==============================================
ToDos:

Issues/ contributions to community:
  
Questions:
"""
# import libraries
import os.path as op
import os
import pandas as pd
import numpy as np
import mne
import sys 
import matplotlib.pyplot as plt

platform = 'mac'  # are you running on bluebear or windows or mac?
test_plot = False  # do you want sanity check plots?

def calculate_spectral_power(epochs, n_fft, fmin, fmax):
    """
    The data are divided into sections being 2 s long. 
      (n_fft = 500 samples) with a 1 s overlap 
      (250 samples). This results in a 0.5 Hz 
      resolution Prior to calculating the FFT of each 
      section a Hamming taper is multiplied.
      n_fft=500, fmin=1, fmax=120"""
    
   # define constant parameters
    welch_params = dict(fmin=fmin, fmax=fmax, picks="meg", n_fft=n_fft, n_overlap=int(n_fft/2))

    # calculate power spectrum 
    """the returned array will have the same
      shape as the input data plus an additional frequency dimension"""
    epochspectrum = epochs.compute_psd(method='welch',  
                                        **welch_params,
                                        n_jobs=30,
                                        verbose=True)
    # get spectrum dala in numpy array format
    epochspectrum_arr = epochspectrum.get_data(return_freqs=True)  #shape: #epochs, #sensors, #freqs

    return epochspectrum, epochspectrum_arr


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
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')  #sensor_layout_name_grad_no_central.csv
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)
sub_IDs = []
psd_all_sens_all_subs = []

for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    try:
        print(f'Reading subject # {i}')
                    
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
        _, epochspectrum_arr = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)  #shape: #epochs, #sensors, #freqs

        # Average PSDs across epochs to get one PSD per channel - might not be needed
        psds_mean = np.mean(epochspectrum_arr, axis=0)
        
        # Append the PSDs to the list
        psd_all_sens_all_subs.append(psds_mean)

    except:
        print(f'an error occured while reading subject # {subjectID} - moving on to next subject')
        pass

# Convert list to numpy array for easier manipulation
psd_all_sens_all_subs_arr = np.array(psd_all_sens_all_subs)  # shape: #subjects, #channels, #freqs

# Calculate grand average PSD for each channel
grand_average_psd = np.mean(psd_all_sens_all_subs_arr, axis=0)  # shape: #channels, #freqs

 # Rename columns with sensor pair names
all_right_names = [f'{row["right_sensors"][0:8]}' for _, row in sensors_layout_names_df.iterrows()]
correlation_df.columns = all_right_names

# Create an EvokedArray object from the DataFrame
rightraw = raw.copy().pick(all_right_names)
evoked = mne.EvokedArray(correlation_df.values.T, rightraw.info, tmin=0, comment=f'spearmanr')

# Plot the correlation values with similar format as plot_topo
evoked_fig_output_fname = op.join(op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/correlation_plot_topos/subtraction-nonoise', f'{substr}.png'))
evoked_fig = evoked.plot_topo(title=f"correlation between frequency and {substr} laterality")
evoked_fig.savefig(evoked_fig_output_fname)

