# -*- coding: utf-8 -*-
"""
===============================================
A01. Alpha lateralization

This code will calculate alpha power in right 
and left parietal and occipital cortex and 
computes the difference

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
import mne

plot = False

# Define where to read and write the data
epoched_dir = r'X:\derivatives\meg\sensor_analysis\epoched_data'
info_dir = r'X:\dataman\Data_information'
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')
output_dir = r'X:\derivatives\meg\sensor_analysis\lateralized_index'
output_fname = op.join(output_dir, 'alpha_lateralizations.csv')
alpha_lat_all_subs_fname = op.join(output_dir, 'alpha_lat_all_subs_all_sens.csv')
roi_fname = op.join(output_dir, 'ROI_sensors.csv') 

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Use a subject as an example for debugging
# subjectID = 110037

alpha_lat_all_subs_all_sens_ext_col = np.zeros(96)
lateralized_alpha = []
sub_IDs =[]
error_prones = []
index =[]
alpha_lat_all_subs_all_sens_tracker = np.full((1,len(good_subject_pd.index)),np.nan)

# Read preprocessed fif files (not epoched) sub-CC110033_ses-rest_task-rest_megtransdef_epo
for i, subjectID in enumerate(good_subject_pd.index):
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    print('calculating alpha lateralization in subject #', subjectID)
    try:
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)
        
        # Filter data in alpha band
        freqs = np.arange(8, 14)
        n_cycles = freqs / 2 
        time_bandwidth = 2.0
        
        def calculate_alpha_lat_sensor(right_sensor, left_sensor):
            tfr_right_sensor = mne.time_frequency.tfr_multitaper(epochs, 
                                                          freqs=freqs, 
                                                          n_cycles=n_cycles,
                                                          time_bandwidth=time_bandwidth, 
                                                          picks=right_sensor,
                                                          use_fft=True, 
                                                          return_itc=False,
                                                          average=True, 
                                                          decim=2,
                                                          n_jobs=4,
                                                          verbose=True)
            tfr_left_sensor = mne.time_frequency.tfr_multitaper(epochs, 
                                                          freqs=freqs, 
                                                          n_cycles=n_cycles,
                                                          time_bandwidth=time_bandwidth, 
                                                          picks=left_sensor,
                                                          use_fft=True, 
                                                          return_itc=False,
                                                          average=True, 
                                                          decim=2,
                                                          n_jobs=4,
                                                          verbose=True)
            alpha_lat_sensor = (tfr_right_sensor.data.mean() - tfr_left_sensor.data.mean()) /\
                (tfr_right_sensor.data.mean() + tfr_left_sensor.data.mean())
            return alpha_lat_sensor
        
        alpha_lateralization_all_sensors = [] 
        for _,row in sensors_layout_names_df.iterrows():
            alpha_lat_sensor = calculate_alpha_lat_sensor(row['SENSOR_R'][1:8], row['SENSOR_L'][1:8])
            alpha_lateralization_all_sensors.append(alpha_lat_sensor)
            alpha_lateralization_all_sensors_array = np.array(alpha_lateralization_all_sensors)
        
        alpha_lat_all_subs_all_sens_ext_col = np.column_stack((alpha_lat_all_subs_all_sens_ext_col, alpha_lateralization_all_sensors_array))
        alpha_lat_all_subs_all_sens_tracker[:,i] = 1  # put 1 for subjects for which we could calculate alpha lat                             
        sub_IDs.append(subjectID)
    except:
        alpha_lat_all_subs_all_sens_tracker[:,i] = 0  # put 0 for subjects for which we couldn't calculate alpha lat
        error_prones.append(subjectID)
        print(f'an error occured while reading subject # {subjectID}')
        pass
# Average over subjects to calculate ROI- 5 pairs of sensors that show highest absolute righ alpha - left alpha 
alpha_lat_all_subs_all_sens = alpha_lat_all_subs_all_sens_ext_col[:, 1:]  # remove the extra column which was only added for column_stack
alpha_lat_all_sens_mean_subs = np.hstack((np.expand_dims(np.arange(0,96), axis=1),
                                         np.expand_dims(np.absolute(np.nanmean(alpha_lat_all_subs_all_sens,axis=1)), axis=1))
                                         )  # expand_dims is only to make the arrays 2D
alpha_lat_sorted_sens_mean_subs = alpha_lat_all_sens_mean_subs[alpha_lat_all_sens_mean_subs[:,1].argsort()[::-1]]
roi_idx = alpha_lat_sorted_sens_mean_subs[0:5,0].astype(int)
roi_sens_names = sensors_layout_names_df.iloc[roi_idx,:]

# Put alpha lateralization index of the sensors in roi from all the participants in one array
alpha_lat_idx_all_roi_all_subs = alpha_lat_all_subs_all_sens[roi_idx,:].transpose()
alpha_lat_idx_mean_roi_all_subs = alpha_lat_idx_all_roi_all_subs.mean(axis=1) # ALI for all participants

# Convert to dataframes
data = {'subject_ID':sub_IDs, 'lateralized_alpha':alpha_lat_idx_mean_roi_all_subs} 
ROI_data = {'region_of_interest': [roi_idx, roi_sens_names]}

df = pd.DataFrame(data=data)
df_roi = pd.DataFrame(data=ROI_data)

# Save 
df.to_csv(output_fname)
df_roi.to_csv(roi_fname)
np.savetxt(alpha_lat_all_subs_fname, alpha_lat_all_subs_all_sens.transpose(), delimiter=",")

