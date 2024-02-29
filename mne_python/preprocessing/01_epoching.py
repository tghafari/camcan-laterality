# -*- coding: utf-8 -*-
"""
===============================================
01. Epoching raw data into fixed length events

This code will epoch continuous MEG signal into 
events with fixed lengths.

written by Tara Ghafari
==============================================
ToDos:
    1) read files from good subjects
    2) create events with fixed lengths for each file
    3) create epochs by findings events
    4) reject bad epochs
    
Issues/ contributions to community:
    1) 
Questions:
    1) 

"""
# import libraries
import os.path as op
import pandas as pd
import mne
platform = 'bluebear'  # are you running on bluebear or windows or mac?

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'


preproc_dir = op.join(rds_dir, 'camcan_bigglm/processed-data/CamCAN_firstlevel')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')  #sensor_layout_name_grad_no_central.csv
deriv_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Set the peak-peak amplitude threshold for trial rejection.
""" subject to change based on data quality"""
reject = dict(grad=5000e-13,  # T/m (gradiometers)
              mag=5e-12,      # T (magnetometers)
              )  # not useful for 7min50 epoch

# Length of epochs
start = 5
stop = 475  # remove first and last 5 seconds

# Read fif files and make fixed events
for subjectID in good_subject_pd.index:

    base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
    preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
    preproc_fif = op.join(preproc_dir, base_name, preproc_name)
    deriv_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif' 
    try:
        print('epoching subject #', subjectID)
        raw = mne.io.read_raw_fif(preproc_fif)
        events = mne.make_fixed_length_events(raw, 
                                                id=1, 
                                                start=start, 
                                                stop=stop,
                                                duration=stop-start,  # The duration to separate events by (in seconds).
                                                first_samp=True,
                                                overlap=0.0)    
        epochs = mne.Epochs(raw, 
                            events, 
                            tmin=start, 
                            tmax=stop, 
                            baseline=None, 
                            proj=True, 
                            picks='all', 
                            detrend=1, 
                            event_repeated='drop',
                            reject=None, 
                            reject_by_annotation=False,
                            preload=True, 
                            verbose=True)
        
        epochs.event_id = {'fixed_length':1}  # rename the event_id

        print('saving subject #', subjectID)
        epochs.save(op.join(deriv_dir, deriv_fname), overwrite=True)
        
    except:
        print(f'an error occured while reading subject # {subjectID}')
        pass