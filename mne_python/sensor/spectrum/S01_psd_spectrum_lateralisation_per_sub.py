# -*- coding: utf-8 -*-
"""
===============================================
S01. Spectrum lateralisation

This code will:
    1. calculate the psd for all frequencies
    and all sensors for one subject
    2. pick right and corresponding left sensors 
    from the spectrum based on sensor-layout-sheet
    3. calculate LI with three methods (subtraction, 
    sum over sub, and log transform) for all freqs 
    4. loop over sensor pairs and append all sensor pairs
     for one subject to save and separate later
    5. loop over subjects
    6. change the structure from all sensor pairs all freqs
     one subject to all freqs all subjects one sensor pair
    7. the code will also identify outliers based on their
    lateralised power. it will calculate zscore oflateralised 
    power for each sensor, and if a participant has over
    2/3 sensors identified as outlier, it will be removed
    from the whole dataset.
     

written by Tara Ghafari
==============================================
ToDos:

Issues/ contributions to community:
  
Questions:
"""
# import libraries
import os.path as op

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mne
from scipy import stats


platform = 'mac'  # are you running on bluebear or windows or mac?
test_plot = False  # do you want sanity check plots?
remove_outliers = False  # do you want to remove outliers based on lateralised power?


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

    return epochspectrum

def pick_sensor_pairs_epochspectrum(epochspectrum, right_sensor, left_sensor):
    """this code will pick sensor pairs for calculating lateralisation 
        from epochspectrum (output of previous function).
        the shape of psd is (1, 1, 239) = #epochs, #sensors, #freqs
        freqs = np.arange(1, 120, 0.5)"""
    
    psd_right_sensor, freqs = epochspectrum.copy().pick(picks=right_sensor).get_data(return_freqs=True)  # freqs is just for a reference
    psd_left_sensor = epochspectrum.copy().pick(picks=left_sensor).get_data()

    return psd_right_sensor, psd_left_sensor, freqs


def calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor):
    """ calculates lateralisation index for each pair of sensors."""

    psd_right_sensor = psd_right_sensor.squeeze()  # squeezable as there's only one sensor and one epoch
    psd_left_sensor = psd_left_sensor.squeeze()

    # Perform element-wise subtraction and division
    subtraction_sensor_pairs = psd_right_sensor - psd_left_sensor
    sum_element = psd_right_sensor + psd_left_sensor
    sumsub_lat_sensor_pairs = subtraction_sensor_pairs / sum_element

    # Log transformation
    division = (psd_right_sensor/psd_left_sensor)
    log_lat_sensor_pairs = np.log(division)

    return subtraction_sensor_pairs, sumsub_lat_sensor_pairs, log_lat_sensor_pairs

def remove_noise_bias(lateralised_power, freqs, h_fmin, h_fmax):
    """to eliminate noise bias in sensors, this definition 
    calculates the average of lateralised power in high 
    frequencies (between h_fmin=90, h_fmax=120), where we don't 
    expect much brain signal, and then subtracts
    that bias from the lateralisation index for each sensor pair
    lateralised_power: output of calculate_spectrum_lateralisation
    (works for all methods, i.e, subtraction, sumsub and log)
    freqs: output of pick_sensor_pairs_epochspectrum
    fmin and fmax: the high frequency section to use for nosie bias """
    
    print(f'Removing noise of {h_fmin} and {h_fmax} from lateralised power')

    # Calculate average power in the specified frequency range
    average_power_high_freqs = np.mean(lateralised_power[(freqs >= h_fmin) & (freqs <= h_fmax)])

    # Subtract the average power from all power values 
    bias_removed_log_lat = lateralised_power.copy()  
    bias_removed_log_lat -= average_power_high_freqs    

    return bias_removed_log_lat

def remove_outliers_and_save(sensor_dataframes, sub_IDs, freqs, output_dir, threshold=2.5, outlier_fraction=2/3):
    """Removes rows with outliers from all sensor dataframes and saves the cleaned dataframes.
    
    Args:
        sensor_dataframes (dict): Dictionary of dataframes to check for outliers.
        sub_IDs (list): List of subject IDs.
        freqs (array-like): Frequency values corresponding to the columns.
        output_dir (str): Directory where cleaned dataframes should be saved.
        threshold (float): Z-score threshold for identifying outliers.
        outlier_fraction (float): Fraction of columns that need to be outliers to consider a row as an outlier.
    """
    outlier_subjects = set()

    # First pass: Identify all outlier subject IDs across all dataframes
    for df_name, df in sensor_dataframes.items():
        # Calculate z-scores across the rows (i.e., across subjects for each frequency)
        zscore = stats.zscore(df, axis=0)  # axis=0 ensures calculation is done across subjects

        # Identify outliers (True where z-score exceeds threshold)
        outliers = abs(zscore) > threshold

        # Count the number of outliers in each row
        outlier_count = outliers.sum(axis=1)

        # Identify rows where outliers are >= 2/3 of total columns
        outlier_rows = outlier_count >= (outlier_fraction * df.shape[1])

        # Add the outlier subject IDs to the set
        outlier_subjects.update(df.index[outlier_rows].tolist())

    # Convert the set to a DataFrame (optional: save this information)
    outlier_subjectID_df = pd.DataFrame(list(outlier_subjects), columns=['SubjectID'])
    outlier_subjectID_df.to_csv(op.join(output_dir, "outlier_subjectIDs.csv"), index=False)

    # Second pass: Remove outlier rows from all sensor dataframes
    for df_name, df in sensor_dataframes.items():
        sensor_dataframes[df_name] = df.drop(index=outlier_subjects)
        # Save the cleaned dataframe as a CSV file
        sensor_dataframes[df_name].to_csv(op.join(output_dir, f"no_outlier-{df_name}.csv"))

    return outlier_subjectID_df


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
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_sumsub_nonoise')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

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
        epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250 (not 1000Hz)

         # Read sensor pairs and calculate lateralisation for each
        for _, row in sensors_layout_names_df.iterrows():
             print(f'Calculating lateralisation in {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}')
                   
             psd_right_sensor, psd_left_sensor, freqs = pick_sensor_pairs_epochspectrum(epochspectrum, 
                                                                                   row['right_sensors'][0:8], 
                                                                                   row['left_sensors'][0:8])
             subtraction_sensor_pairs, _, _ = calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor)

             # Remove noise bias
             bias_removed_lat = remove_noise_bias(subtraction_sensor_pairs, freqs, h_fmin=90, h_fmax=120)

             # Reshape the array to have shape (473 (#freqs), 1) for stacking
             bias_removed_lat = bias_removed_lat.reshape(-1,1)
             # Append the reshaped array to the list - shape #sensor_pairs, #freqs, 1
             stacked_sensors.append(bias_removed_lat)

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

if remove_outliers:
    # Remove outliers and save cleaned dataframes
    remove_outliers_and_save(sensor_dataframes, sub_IDs, freqs, output_dir)


if test_plot:

    # Sanity check with plot_topos
    to_tests = np.arange(0,6)
    to_test_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots')
    
    for i in range(np.shape(spec_lateralisation_all_sens_all_subs)[0]):
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
        plt.ylabel('Spec lateralisation')
        plt.savefig(op.join(to_test_output_dir, f'sub_{sub}_lat_psd.png'))
        plt.close()

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

