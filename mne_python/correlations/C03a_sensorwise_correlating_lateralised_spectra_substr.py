# -*- coding: utf-8 -*-
"""
====================================
C03a_sensorwise_correlating_lateralised_spectra_substr:
    this script is in use:
    1. reads lateralised indices of one 
    pair of sensors (one csv file)
    from each frequency bin (each folder) into
    one variable called pair_lat_freq
    2. reads lateralised values of each 
    subcortical structure into hipp/thal_lat
    3. removes the participants without spectra
    or lateralised volume based on the subID 
    column
    4. calculates correlation for remaining
    participants between pair sensor lateralisation
    and lateralisation of one subcortical structure
    5. puts that one correlation value in a table
    called "lat_spectra_substr_corr"
    6. loops over all substrs and saves each
    correlation value column-wise
    7. loops over all frequencies and
    save each correlation row-wise
    8. save each table (containing correlation values
      of all substrs and all freqs )
    
    
Written by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import pandas as pd
import numpy as np
import os.path as op
import os
import scipy.stats as stats

platform = 'mac'

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    
# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
spectra_dir = op.join(deriv_dir, 'meg/sensor/lateralized_index/frequency_bins')
substr_dir = op.join(deriv_dir, 'mri/lateralized_index')
substr_sheet_fname = op.join(substr_dir, 'lateralization_volumes.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# Load substr file
substr_lat_df = pd.read_csv(substr_sheet_fname)
substr_lat_df = substr_lat_df.rename(columns={'SubID':'subject_ID'})

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

freqs = np.array([1,2])
# freqs = np.append(0.1, np.arange(1,51)) # freqs start from 0.1 (not 0)
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

def working_df_maker(freq, spectra_dir, right_sensor, left_sensor, substr_lat_df):
    # Navigate to the folder
    lat_index_dir = op.join(spectra_dir, f'{str(float(freq))}')

    # Load lateralisation index for each pair
    spectrum_pair_lat_df = pd.read_csv(op.join(lat_index_dir, f'{right_sensor}_{left_sensor}.csv'))
    print(f'creating working df for {right_sensor}_{left_sensor}')
    # Remove brackets and convert to float
    spectrum_pair_lat_df['lateralised_spec'] = spectrum_pair_lat_df['lateralised_spec'].str.strip('[]').astype(float)

    # merge and match the subject_ID column
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['subject_ID'])
    working_df = working_df.dropna()

    return working_df

def pearson_calculator(working_df, substr, ls_corrs, ls_pvals):
    print(f'calculating pearson correlation for {substr}')
    temp_corr, temp_pvalue = stats.pearsonr(working_df['lateralised_spec'].to_numpy(), working_df[substr].to_numpy()) 
    ls_corrs.append(temp_corr)
    ls_pvals.append(temp_pvalue)
    return ls_corrs, ls_pvals

for _, row in sensors_layout_names_df.head(2).iterrows():
    print(f'working on pair {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}')

    # Predefine lists
    ls_corrs = []
    ls_pvals = []
    list_of_freqs = []
    ls_corrs_arr = []
    ls_pvals_arr = []
    
    output_corr_dir = op.join(deriv_dir, 'correlations', 'sensor_pairs', f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}')
    if not op.exists(output_corr_dir):
        os.makedirs(output_corr_dir)

    pearsonrs_csv_file = op.join(output_corr_dir, 'lat_spectra_substr_pearsonr.csv')
    pvals_csv_file = op.join(output_corr_dir, 'lat_spectra_substr_pvals.csv')

    for freq in freqs:
        print(f'calculating correlations for {freq} Hz')
        list_of_freqs.append(freq)
        working_df = working_df_maker(freq, spectra_dir, row["right_sensors"][0:8], row["left_sensors"][0:8], substr_lat_df)

        # Calculate correlation with each substr
        for substr in substrs:
            ls_corrs, ls_pvals = pearson_calculator(working_df, substr, ls_corrs, ls_pvals)

    # Preparing the lists
    ls_corrs_arr = np.asarray(ls_corrs).reshape(-1, 7)
    ls_pvals_arr = np.asarray(ls_pvals).reshape(-1, 7)
    output_pearr_df = pd.DataFrame(ls_corrs_arr, columns=['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu'], index=list_of_freqs)
    output_pvalue_df = pd.DataFrame(ls_pvals_arr, columns=['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu'], index=list_of_freqs)
    
    # Save
    output_pearr_df.to_csv(pearsonrs_csv_file)
    output_pvalue_df.to_csv(pvals_csv_file)

    # Freshen the variables for the next frequency band
    del output_pearr_df
    del output_pvalue_df
