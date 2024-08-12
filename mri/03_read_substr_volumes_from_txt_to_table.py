# -*- coding: utf-8 -*-
"""
===============================================
03. substr volumes to tabe

This code reads the volume of all subcortical
structures from a text file (output of fslstats)
and put them in a table.

written by Tara Ghafari
==============================================
"""

import numpy as np
import os.path as op
import pandas as pd

# Define where to read and write the data
subStr_segmented_dir = r'X:\derivatives\mri\subStr_segmented'
info_dir = r'X:\dataman\Data_information'
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
output_dir = r'X:\derivatives\mri\lateralized_index'
output_fname = op.join(output_dir, 'all_subs_substr_volumes.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Specify labels assigned to structures thatwere segmented by FSL
labels = [10, 11, 12, 13, 16, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58]
structures = ['L-Thal', 'L-Caud', 'L-Puta', 'L-Pall', 'BrStem /4th Ventricle',
              'L-Hipp', 'L-Amyg', 'L-Accu', 'R-Thal', 'R-Caud', 'R-Puta',
              'R-Pall', 'R-Hipp', 'R-Amyg', 'R-Accu']

all_subject_substr_volume_table = np.full((619, 15), np.nan)
sub_IDs =[]

# Read good subjects 
for i, subjectID in enumerate(good_subject_pd.index):
    base_name = 'sub-CC' + str(subjectID) + '.SubVol'
    if op.exists(op.join(subStr_segmented_dir, base_name)):
        for idx, label in enumerate(labels):
            volume_label = 'volume' + str(label)
            substr_vol_fname = op.join(subStr_segmented_dir, base_name, volume_label)
            if op.exists(substr_vol_fname):
                print(f"reading structure {structures[idx]} in subject # {subjectID}")
                # Read the text file
                with open(substr_vol_fname, "r") as file:
                    line = file.readline()
                substr_volume_array = np.fromstring(line.strip(), sep=' ')[1]     
            else:
                print(f"no volume for substructure {structures[idx]} found for subject # {subjectID}")
                substr_volume_array = np.nan  
            
            # Store the volume of each substr in one columne and data of each subject in one row  
            all_subject_substr_volume_table[i, idx] = substr_volume_array
    else:
        print('no substructures segmented by fsl for subject # ', subjectID)
        all_subject_substr_volume_table[i, :] = np.nan 
    
    sub_IDs.append(subjectID)
    
 
# Create a dataframe for all the data
columns = ['SubID'] + structures
df = pd.DataFrame(np.hstack((np.array(sub_IDs).reshape(-1, 1), all_subject_substr_volume_table)),
                  columns=columns)
df.set_index('SubID', inplace=True)

# Save 
df.to_csv(output_fname)   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    