
"""
===============================================
CS06_left_right_spectra_per_sub

the aim of this code is to plot the spectra for 
a few participants that were selected from 
the cloud plots on right and left, separately.


This code will:
    1. inputs participnats id
    2. input a list of sensors for which
    we'd want to plot the spectrum
    3. plot right and left spectra, separately
    4. plot both side spectra 


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

platform = 'mac'


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

def pick_sensor_pairs_epochspectrum(epochspectrum, right_sensor, left_sensor):
    """this code will pick sensor pairs for calculating lateralisation 
        from epochspectrum (output of previous function).
        the shape of psd is (1, 1, 119) = #epochs, #sensors, #freqs
        freqs = np.arange(1, 60.5, 0.5)"""
    
    psd_right_sensor, freqs = epochspectrum.copy().pick(picks=right_sensor).get_data(return_freqs=True)  # freqs is just for a reference
    psd_left_sensor = epochspectrum.copy().pick(picks=left_sensor).get_data()

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PSD for the right sensor
    psd_right_plot = epochspectrum.copy().pick(picks=right_sensor).plot(axes=axes[1])
    axes[1].set_title(f'Right sensor {right_sensor}')

    # Plot PSD for the left sensor
    psd_left_plot = epochspectrum.copy().pick(picks=left_sensor).plot(axes=axes[0])
    axes[0].set_title(f'Left sensor {left_sensor}')

    plt.tight_layout()
    plt.show()

    return psd_right_sensor, psd_left_sensor, freqs


def calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor):
    """ calculates lateralisation index for each pair of sensors."""

    psd_right_sensor = psd_right_sensor.squeeze()  # squeezable as there's only one sensor and one epoch
    psd_left_sensor = psd_left_sensor.squeeze()

    # Perform element-wise subtraction and division
    subtraction = psd_right_sensor - psd_left_sensor
    sum_element = psd_right_sensor + psd_left_sensor
    spectrum_lat_sensor_pairs = subtraction / sum_element

    return spectrum_lat_sensor_pairs

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
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Preallocate lists
subjectIDs_to_plot = [321506, 620935]
sensor_pairs_to_plot = [['MEG0422', 'MEG1112'],['MEG0233','MEG1343']]
sub_IDs = []
spec_lateralisation_all_sens_all_subs = []

for subjectID in subjectIDs_to_plot:
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    try:
        print(f'Reading subject # {subjectID}')
                    
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
        epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=60)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250 (not 1000Hz)

         # Read sensor pairs and calculate lateralisation for each
        for i in range(0, len(sensor_pairs_to_plot), 2):
             working_pair = sensor_pairs_to_plot[i]

             psd_right_sensor, psd_left_sensor, freqs = pick_sensor_pairs_epochspectrum(epochspectrum, 
                                                                                           working_pair[1], 
                                                                                           working_pair[0])     
             spectrum_lat_sensor_pairs = calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor)
             
             # Plot spectrum lateralisation of these sensors vs. frequency
             _, ax = plt.subplots()
             ax.plot(freqs, spectrum_lat_sensor_pairs)
             ax.set(title=f"lateralisation spectrum for {working_pair[1]}_{working_pair[0]}",
                    xlabel="Frequency (Hz)",
                    ylabel="Lateralisation Index",
                    )

    except:
        print(f'an error occured while reading subject # {subjectID} - moving on to next subject')
        pass

# Sanity check with plot_topos
to_tests = np.arange(0,6)
to_test_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/PSD_plot_topos')

for _ in to_tests:
    random_index = np.random.randint(0, len(good_subject_pd))
    # Get the subject ID at the random index
    subjectID = good_subject_pd.iloc[random_index]['SubjectID'][2:]
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
    epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=100)   

    # Plot the EpochSpectrum
    # fig = epochspectrum.plot_topo(color='k', fig_facecolor='w', axis_facecolor='w', show=False)  # raising a size error for no reason?
    # plt.title(f'Sub_{subjectID}', y=0.9)
    # fig.savefig(op.join(to_test_output_dir, f'sub_{subjectID}_epochspectrum_topo.png'))

    # Plot a couple sensors
    fig_sens = epochspectrum.plot()
    plt.title(f'sub_{subjectID}')
    fig_sens.savefig(op.join(to_test_output_dir, f'sub_{subjectID}_epochspectrum_psd.png'))

