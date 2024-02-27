# -*- coding: utf-8 -*-
"""
===============================================
S01. Spectrum lateralisation

This code will:
    1. calculate the psd for all frequencies
    and all sensors for one subject
    2. pick right and corresponding left sensors 
    from the spectrum based on sensor-layout-sheet
    3. calculate LI for all freqs 
    4. loop over sensor pairs
    5. save each freq and each sensor for one
    subject in one dataframe
    2. loop over subjects
    3. append their dataframes to the previous
    subjects


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
platform = 'bluebear'  # are you running on bluebear or windows or mac?

def calculate_spectral_power(epochs, freqs, right_sensor, left_sensor):
   # define constant parameters
    welch_params = dict(n_fft=1000, n_per_seg=1000, overlap=500, average='mean')
    n_cycles = freqs / 2
    time_bandwidth = 2.0
    # calculate power spectrum for right and left sensors separately
    """the returned array will have the same
      shape as the input data plus an additional frequency dimension"""
    right_psd = epochs.compute_psd(method='welch',  
                                   **welch_params,
                                   time_bandwidth=time_bandwidth,
                                   n_cycles=n_cycles,
                                   n_jobs=30,
                                   verbose=True)
    
    left_psd = epochs.compute_psd(method='welch',  
                                   **welch_params,
                                   time_bandwidth=time_bandwidth,
                                   n_cycles=n_cycles,
                                   n_jobs=30,
                                   verbose=True)
    return right_psd, left_psd

def calculate_spectrum_lateralisation(right_psd, left_psd):
    # calculate lateralisation index for each pair of sensors
    right_power = right_psd.get_data()  # shape: #epochs, #sensors, #frequencies
    left_power = left_psd.get_data()  # shape: #epochs, #sensors, #frequencies
    spectrum_lat_sensor_pairs = (np.mean(right_power, axis=(0,2)) - np.mean(left_power, axis=(0,2))) /\
                                (np.mean(right_power, axis=(0,2)) + np.mean(left_power, axis=(0,2)))
    return spectrum_lat_sensor_pairs

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched_data')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')  #sensor_layout_name_grad_no_central.csv
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/frequency_bins')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# frequencies to calculate lateralisaion index for
# freqs = np.append(0.1, np.arange(1, 51))  # don't need this if running aray jobs
# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# HPC related code to accept command-line arguments for the frequencies.
#######
if __name__ == "__main__":
    # Read the frequency values from command-line arguments
    freqs = [float(arg) for arg in sys.argv[1:]]
######

for freq in freqs:
    # make one folder for each frequency bin
    output_freq_dir = op.join(output_dir, f'{int(freq)}')
    if not op.exists(output_freq_dir):
        os.makedirs(output_freq_dir)
    # read sensor pairs and calculate lateralisation for each
    for _, row in sensors_layout_names_df.head(75).iterrows():
        sub_IDs = []
        spec_lateralisation_all_subs = []

        # read subjects one by one and calculate lateralisation index for each pair of sensor
        for i, subjectID in enumerate(good_subject_pd.index):
            epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
            epoched_fif = op.join(epoched_dir, epoched_fname)
            sub_IDs.append(subjectID)

            try:
                print(f'calculating lateralisation of {freq} Hz in {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}\
                          in subject # {i}')
                          
                epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # 5-sec epochs
                right_psd, left_psd = calculate_spectral_power(epochs, freq, row['right_sensors'][0:8], row['left_sensors'][0:8])
                spectrum_lat_sensor_pairs = calculate_spectrum_lateralisation(right_psd, left_psd)
                # append all subjects together in one csv
                spec_lateralisation_all_subs.append(spectrum_lat_sensor_pairs)
                # keep track of the subjects with lateralisation indices                        
                
            except:
                spec_lateralisation_all_subs.append(np.nan)  # put nan for subjects for which we couldn't calculate alpha lat
                print(f'an error occured while reading subject # {subjectID}')
                pass
    
            # convert each pair into one dataframe and csv file
        data = {'subject_ID':sub_IDs, 'lateralised_spec':spec_lateralisation_all_subs}     
        spec_lateralisation_all_subs_df = pd.DataFrame(data=data)

        # save to disc 
        output_fname = op.join(output_freq_dir, f'{row["right_sensors"][0:8]}_{row["left_sensors"][0:8]}.csv')
        spec_lateralisation_all_subs_df.to_csv(output_fname)

