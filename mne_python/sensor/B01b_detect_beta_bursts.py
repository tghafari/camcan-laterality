# -*- coding: utf-8 -*-
"""
===============================================
B01. Detect beta bursts

This code will apply hilbert transform on beta 
band and detects number of beta bursts and their
duration in right and left, separately. 
Output is lateralization index containing the
difference in number and duration of beta 
bursts in in right - left. Each participant 

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
preproc_dir = r'X:\camcan_bigglm\processed-data\CamCAN_firstlevel'
info_dir = r'X:\dataman\Data_information'
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
output_dir = r'X:\derivatives\meg\sensor_analysis\lateralized_index'
output_fname = op.join(output_dir, 'beta_bursts.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Use a subject as an example for debugging
# subjectID = 110037

lateralized_burst_num = []
lateralized_burst_dur = []
sub_IDs =[]

# Read preprocessed fif files (not epoched)
for subjectID in good_subject_pd.index:
    base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
    preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
    preproc_fif = op.join(preproc_dir, base_name, preproc_name)
    print('detecting burst in subject #', subjectID)

    raw = mne.io.read_raw_fif(preproc_fif, preload=True)
    
    # Filter data in beta band
    freq_band = [13, 30]
    raw.filter(freq_band[0], freq_band[1], fir_design='firwin')
    
    # Extract the amplitude envelope of the filtered data using the Hilbert transform
    raw.apply_hilbert(envelope=True)
    
    # Pick ROI
    picks_R = mne.pick_channels(raw.ch_names, include=mne.read_vectorview_selection(['Right-parietal',
                                                                                     'Right-occipital'],
                                                                                    info=raw.info))
    raw_R = raw.copy().pick(picks_R)
    
    picks_L = mne.pick_channels(raw.ch_names, include=mne.read_vectorview_selection(['Left-parietal',
                                                                                     'Left-occipital'],
                                                                                    info=raw.info))
    raw_L = raw.copy().pick(picks_L)
    
    raw_ROI = raw.copy().pick(np.concatenate((picks_R, picks_L), axis=0))
    
    # Plot the amplitude envelope (visually inspect for beta bursts)
    if plot:
        raw_ROI.plot(duration=60, scalings=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=5e-4,
                                        emg=1e-3, misc=1e-3),
                     butterfly=False, color='blue', 
                     show_scrollbars=False, title='MEG Amplitude Envelope')
        
    # Define a threshold for beta burst detection based on the ROI
    amp_envelope_threshold = raw_ROI.get_data()
    threshold = np.median(amp_envelope_threshold) * 6  # defined by Shin et al. 2017

    # Calculate the amplitude envelope for right and left separately
    amp_envelope_R = raw_R.get_data()
    amp_envelope_L = raw_L.get_data()
    
    
    # Detect beta bursts using the amplitude envelope and the threshold 
    # for right and left separately
    def detect_burst(amp_envelope, threshold):
        beta_bursts =[]
        bursts_duration =[]
        in_burst = False
        for i, amp in enumerate(amp_envelope[0]):
            if amp > threshold and not in_burst:
                in_burst = True
                burst_start = i
            elif amp <= threshold and in_burst:
                in_burst = False
                burst_end = i
                burst_duration = burst_end - burst_start
                if burst_duration >= 3: # only consider bursts that last at least3 time points
                   bursts_duration.append(burst_duration)               
                   beta_bursts.append([burst_start, burst_end])
        return beta_bursts, bursts_duration
    
    beta_bursts_R, bursts_duration_R = detect_burst(amp_envelope_R, threshold)
    beta_bursts_L, bursts_duration_L = detect_burst(amp_envelope_L, threshold)
    
    
    # Calculate difference between the number of bursts between right and left
    burst_num_diff = len(beta_bursts_R) - len(beta_bursts_L)
    burst_avg_dur_diff = np.mean(bursts_duration_R) - np.mean(bursts_duration_L)
    sub_IDs.append(subjectID)
    
    print(f"Detected {burst_num_diff} more beta bursts on right than left")  
    print(f"Duration of beta bursts were on average {burst_avg_dur_diff:.2f} longer on right than left")  

# Create an array of all subjects' lateralization indices and convert to dataframe
    lateralized_burst_num.append(burst_num_diff)
    lateralized_burst_dur.append(burst_avg_dur_diff)
    
data = {'subject_ID':sub_IDs, 'lateralized_burst_num':lateralized_burst_num,
        'lateralized_burst_dur':lateralized_burst_dur}
df = pd.DataFrame(data=data)

# Save 
df.to_csv(output_fname)
