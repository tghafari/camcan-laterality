# -*- coding: utf-8 -*-
"""
===============================================
S01b. Spectrum lateralisation

This code will:
    1. calculate the psd for all frequencies
    and all sensors for one subject
    2. pick right and corresponding left sensors 
    from the spectrum based on sensor-layout-sheet
    3. calculate LI for all freqs using log(power)
    4. loop over sensor pairs and append all sensor pairs
     for one subject to save and separate later
    5. loop over subjects
    6. change the structure from all sensor pairs all freqs
     one subject to all freqs all subjects one sensor pair
 


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
import math

platform = 'bluebear'  # are you running on bluebear or windows or mac?


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

    return psd_right_sensor, psd_left_sensor, freqs


def calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor):
    """ calculates lateralisation index for each pair of sensors."""

    psd_right_sensor = psd_right_sensor.squeeze()  # squeezable as there's only one sensor and one epoch
    psd_left_sensor = psd_left_sensor.squeeze()

    # Perform element-wise subtraction and division
    division = (psd_right_sensor/psd_left_sensor)
    log_division = np.log(division)
    multiplication = (psd_right_sensor*psd_left_sensor)
    log_lat_sensor_pairs = np.log(division/multiplication)

    return log_division

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
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_logarithm')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

sanity_check = False  # do you want to plot sanity check figures?

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Preallocate lists
sub_IDs = []
spec_lateralisation_all_sens_all_subs = []

for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    try:
        print(f'Reading subject # {i}')
                    
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
        epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=60)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250 (not 1000Hz)

         # Read sensor pairs and calculate lateralisation for each
        for _, row in sensors_layout_names_df.iterrows():
             print(f'Calculating lateralisation in {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}')
                   
             psd_right_sensor, psd_left_sensor, freqs = pick_sensor_pairs_epochspectrum(epochspectrum, 
                                                                                   row['right_sensors'][0:8], 
                                                                                   row['left_sensors'][0:8])
             spectrum_lat_sensor_pairs = calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor)

             # Reshape the array to have shape (119 (#freqs), 1) for stacking
             spectrum_lat_sensor_pairs = spectrum_lat_sensor_pairs.reshape(-1,1)
             # Append the reshaped array to the list - shape #sensor_pairs, #freqs, 1
             stacked_sensors.append(spectrum_lat_sensor_pairs)

        # Horizontally stack the spec_lateralisation_all_sens - shape #freqs, #sensor_pairs
        spec_lateralisation_all_sens = np.hstack(stacked_sensors)
        # Append all subjects together
        spec_lateralisation_all_sens_all_subs.append(spec_lateralisation_all_sens)  # shape = #sub, #freqs, #sensor_pairs
        sub_IDs.append(subjectID)

    except:
        print(f'an error occured while reading subject # {subjectID} - moving on to next subject')
        pass

# Prepare for enumerate over sensor pairs - #sensor_pairs, #subs, #freqs
all_freq_all_subs_transposed = np.transpose(spec_lateralisation_all_sens_all_subs, (2, 0, 1))  

sensor_dataframes = {}
for idx, array in enumerate(all_freq_all_subs_transposed):
    # Extract the slice along the third dimensio
    df_name = f"{sensors_layout_names_df.iloc[idx, 0]}_{sensors_layout_names_df.iloc[idx, 1]}"
    sensor_dataframes[df_name] = pd.DataFrame(array, index=sub_IDs, columns=freqs)
    # Save the dataframe as a CSV file
    sensor_dataframes[df_name].to_csv(op.join(output_dir, f"{df_name}.csv")) 

# Plot lateralisation indices for sanity check
if sanity_check:

    to_test_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots')

    for i in range(spec_lateralisation_all_sens.shape[0]):
        participant_data = spec_lateralisation_all_sens_all_subs[i]  # data for current participant
        sub = sub_IDs[i]

        # Plot data for each sensor (dimension 2)
        for j in range(participant_data.shape[1]):
            #plt.figure()  # only added for checking sensor pairs separately
            plt.plot(freqs, participant_data[:, j])
            #plt.savefig(op.join(to_test_output_dir, f'sensor_{j}_log_lat_one_sensor_psd.png'))  # only added for checking sensor pairs separately
            #plt.close()  # only added for checking sensor pairs separately

        plt.title(f'Participant {sub}')
        plt.xlabel('Frequency')
        plt.ylabel('Spec log lateralisation')
        plt.savefig(op.join(to_test_output_dir, f'sub_{sub}_log_unnorm_lat_psd.png'))
        plt.close()

    # Sanity check with plot_topos
    to_tests = np.arange(0,6)

    for _ in to_tests:
        random_index = np.random.randint(0, len(good_subject_pd))
        # Get the subject ID at the random index
        subjectID = good_subject_pd.iloc[random_index]['SubjectID'][2:]
        epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
        epoched_fif = op.join(epoched_dir, epoched_fname)
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
        epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=100)   

        # Plot the EpochSpectrum
        fig = epochspectrum.plot_topo(color='k', fig_facecolor='w', axis_facecolor='w', show=False)  # raising a size error for no reason?
        plt.title(f'Sub_{subjectID}', y=0.9)
        fig.savefig(op.join(to_test_output_dir, f'sub_{subjectID}_epochspectrum_topo.png'))

        # Plot a couple sensors
        fig_sens = epochspectrum.plot()
        plt.title(f'sub_{subjectID}')
        fig_sens.savefig(op.join(to_test_output_dir, f'sub_{subjectID}_epochspectrum_psd.png'))

