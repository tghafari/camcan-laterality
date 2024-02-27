# -*- coding: utf-8 -*-
"""
===============================================
00. Extract demographic information 

This code extracts demographic information from 
"participant_data.csv" from camcan data base

written by Tara Ghafari
adapted from https://github.com/tghafari/camcan_betaEvents_ageing/blob/master/00_demographics.py
==============================================  
ToDos:
    1) 
Issues:
    1) 
Contributions to Community:
    1) 
Questions:
    1) 
"""

# Import libraries
import csv
import os
import os.path as op
import pandas as pd
import mne

# Script to pull demographic and data-related information for participants

dem_dir = r'X:\dataman\useraccess\processed\Tara_Ghafari_1448'
dem_fname = op.join(dem_dir, 'standard_data.csv')
MEG_data_folder = r'X:\cc700\meg\pipeline\release005\BIDSsep\derivatives_rest\aa\AA_movecomp_transdef\aamod_meg_maxfilt_00003'  # RDS folder for MEG data
data_path = op.join(MEG_data_folder, 'sub-CC')
preproc_dir = r'X:\camcan_bigglm\processed-data\CamCAN_firstlevel'
output_dir = r'Z:\Projects\Subcortical Structures\Resting State\CamCan\Results\Data information'

# Do you want to quality check raw or preproc data?
raw_data = False
preproc_data = True

####################################################################
# Pull demographic information from the CamCAN supplied spreadsheet

csvfile = open(dem_fname)
reader = csv.DictReader(csvfile)
subject_data = []
subject_IDs = []
for row in reader:

    # Read demographic data
    subject_ID = row['CCID']
    age = int(round(float(row['Age']),0))
    hand = float(row['Hand'])
    gender = row['Sex']

    subject_IDs.append(int(subject_ID[2:]))
    subject_data.append({'SubjectID':subject_ID, 'Age':age, 'Hand':hand, 'Gender':gender})

# Put all this information into a pandas dataframe 
subject_data_pd = pd.DataFrame(subject_data, index=subject_IDs)

###########################################################################
# Check which subject has megtransdef and preproc resting state data

megtransdef_exists = []
megtransdef_preproc_exists = []

for subjectID in subject_data_pd.index:
    
    base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
    raw_fif = op.join(data_path + str(subjectID) , base_name + '.fif')
    preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
    preproc_fif = op.join(preproc_dir, base_name, preproc_name)
    if os.path.exists(raw_fif):
        megtransdef_exists.append(1)
    else:
        megtransdef_exists.append(0)
    if os.path.exists(preproc_fif):
        megtransdef_preproc_exists.append(1)
    else:
        megtransdef_preproc_exists.append(0)

# Add these findings to the pandas dataframe
subject_data_pd['megtransdef_exists'] = pd.Series(megtransdef_exists, index=subject_data_pd.index)
subject_data_pd['megtransdef_preproc_exists'] = pd.Series(megtransdef_preproc_exists, index=subject_data_pd.index)

###########################################################################
# Get some info for reporting

withRawData = subject_data_pd.loc[subject_data_pd['megtransdef_exists'] == 1]
withPreprocData = subject_data_pd.loc[subject_data_pd['megtransdef_preproc_exists'] == 1]
print('There is megtransdef data for ' + str(len(withRawData)) + ' participants')
print('There is preprocessed megtransdef data for ' + str(len(withPreprocData)) + ' participants')

###########################################################################
## check if raw files are readable
if raw_data:
    good_subject_data_temp = subject_data_pd.copy()
    
    data_reads = []
    
    for subjectID in good_subject_data_temp.index:
    
        if subject_data_pd['megtransdef_exists'][subjectID]:
            print(str(subjectID))
            # Get events
            base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
            raw_fif = op.join(data_path + str(subjectID) , base_name + '.fif')
            try:
                raw = mne.io.Raw(raw_fif)
            except ValueError:
                data_reads.append(0)
            else:
                data_reads.append(1)
                
    # Add results to pandas dataframe - drop subjects with no raw data file
    good_subject_data_1 = good_subject_data_temp.copy()
    good_subject_data_1 = good_subject_data_1.loc[good_subject_data_1['megtransdef_exists']==1]
    
    # Drop subjects with unreadable raw data files
    good_subject_data_1['data_reads'] = pd.Series(data_reads, index=good_subject_data_1.index)
    good_subject_data_1 = good_subject_data_1.loc[good_subject_data_1['data_reads']==1]

    # Write files
    good_subject_data_1.to_csv(op.join(output_dir, 'demographics_goodSubjects.csv'))
     
#######################################################################
## check if preproc files are readable
if preproc_data:
    good_subject_data_temp = subject_data_pd.copy()
    
    preproc_data_reads = []
    
    for subjectID in good_subject_data_temp.index:
    
        if subject_data_pd['megtransdef_preproc_exists'][subjectID]:
            print(str(subjectID))
            # Get events
            base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
            preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
            preproc_fif = op.join(preproc_dir, base_name, preproc_name)
            try:
                raw = mne.io.Raw(preproc_fif)
            except ValueError:
                preproc_data_reads.append(0)
            else:
                preproc_data_reads.append(1)
                
    # Add results to pandas dataframe - drop subjects with no preproc data file
    good_subject_data_2 = good_subject_data_temp.copy()
    good_subject_data_2 = good_subject_data_2.copy().loc[good_subject_data_2['megtransdef_preproc_exists']==1]
    
    # Drop subjects with unreadable preproc data files
    good_subject_data_2['preproc_data_reads'] = pd.Series(preproc_data_reads, index=good_subject_data_2.index)
    good_subject_data_2 = good_subject_data_2.loc[good_subject_data_2['preproc_data_reads']==1]
    
    # Write files
    good_subject_data_2.to_csv(op.join(output_dir, 'demographics_goodPreproc_subjects.csv'))

#######################################################################

# Write files and make standard output

subject_data_pd.to_csv(op.join(output_dir, 'demographics_allSubjects.csv'))

