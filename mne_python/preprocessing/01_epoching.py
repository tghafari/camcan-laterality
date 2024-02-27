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

# Define where to read and write the data
preproc_dir = r'X:\camcan_bigglm\processed-data\CamCAN_firstlevel'
info_dir = r'Z:\Projects\Subcortical Structures\Resting State\CamCan\Results\Data information'
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
deriv_dir = r'X:\betabursts\derivatives'

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Set the peak-peak amplitude threshold for trial rejection.
""" subject to change based on data quality"""
reject = dict(grad=5000e-13,  # T/m (gradiometers)
              mag=5e-12,      # T (magnetometers)
              )

# Read fif files and make fixed events
for subjectID in good_subject_pd.index:

   base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
   preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
   preproc_fif = op.join(preproc_dir, base_name, preproc_name)
   deriv_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif' 
   print('epoching subject #', subjectID)
   
   raw = mne.io.read_raw_fif(preproc_fif)
    
   events = mne.make_fixed_length_events(raw, id=1, start=10, stop=550,
                                         duration=5.0, overlap=0.0)
    
   metadata, _, _ = mne.epochs.make_metadata(
                   events=events, event_id={'fixed_length':1},
                   tmin=0, tmax=5.0, 
                   sfreq=raw.info['sfreq'])
    
   epochs = mne.Epochs(raw, events, tmin=0, tmax=5.0, 
                       baseline=None, proj=True, picks='all', 
                       detrend=1, event_repeated='drop',
                       reject=reject, reject_by_annotation=True,
                       preload=True, verbose=True)
   
   epochs.save(op.join(deriv_dir,deriv_fname), overwrite=True)
