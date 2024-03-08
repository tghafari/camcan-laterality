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

def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """This definition merges the dataframes containing spectrum lateralisation values and 
    substr lateralisation values together"""

    # Navigate to the sensor_pair folder
    spec_lat_index_fname = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')

    # Load lateralisation index for each pair
    spectrum_pair_lat_df = pd.read_csv(spec_lat_index_fname)
    spectrum_pair_lat_df = spectrum_pair_lat_df.rename(columns={'Unnamed: 0':'subject_ID'})
    
    # Merge and match the subject_ID column and remove nans
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['subject_ID'])
    working_df = working_df.dropna()

    # Get the freqs of spectrum from spec_pair_lat
    freqs = spectrum_pair_lat_df.columns.values[1:]  # remove subject_ID column
    freqs = [float(freq) for freq in freqs]  # convert strings to floats
    return working_df, freqs

def pearson_calculator(working_df, freq, substr, ls_corrs_all_freqs, ls_pvals_all_freqs):
    """This definition takes working df, reads lateralised value of one freq and one substr and calculates
    pearson correlation for all subjects"""

    print(f'Calculating pearson correlation for {freq} and {substr}')
    temp_corr, temp_pvalue = stats.pearsonr(working_df[f'{freq}'].to_numpy(), working_df[substr].to_numpy()) 
    ls_corrs_all_freqs.append(temp_corr)  # try to append horizontally- all subs in one freq and one substr = 1 corr and 1 p-value
    ls_pvals_all_freqs.append(temp_pvalue)
    return ls_corrs_all_freqs, ls_pvals_all_freqs

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    
# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
spectra_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs')
substr_dir = op.join(deriv_dir, 'mri/lateralized_index')
substr_sheet_fname = op.join(substr_dir, 'lateralization_volumes.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# Load substr file
substr_lat_df = pd.read_csv(substr_sheet_fname)

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

for i, row in sensors_layout_names_df.iterrows():
    print(f'Working on pair {row["left_sensors"][0:8]}, {row["right_sensors"][0:8]}')

    # Get the frequencies of spectrum (only once enough)
    _, freqs = working_df_maker(spectra_dir, 
                                row["left_sensors"][0:8], 
                                row["right_sensors"][0:8], 
                                substr_lat_df) if i == 0 else (None, freqs)

    # Make the working df containing lateralised value of the current sensor pair
    working_df, _ = working_df_maker(spectra_dir,  # shape: #subject by #freqs + #substr + 1(for subject_ID column) = 560 * 481
                                    row["left_sensors"][0:8], 
                                    row["right_sensors"][0:8], 
                                    substr_lat_df)
    
    output_corr_dir = op.join(deriv_dir, 'correlations', 'sensor_pairs_0803_final',
                               f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}')
    if not op.exists(output_corr_dir):
        os.makedirs(output_corr_dir)
 
    # Calculate correlation in each substr
    for substr in substrs:
        print(f'Working on {substr}')

        # Predefine lists
        ls_corrs_all_freqs = []
        ls_pvals_all_freqs = []
        list_of_freqs = []
        ls_corrs_arr = []
        ls_pvals_arr = []
        
        output_corr_substr_dir = op.join(output_corr_dir, f'{substr}')
        if not op.exists(output_corr_substr_dir):
            os.makedirs(output_corr_substr_dir)
    
        # Calculate correlation with each freq 
        for freq in freqs:
            print(f'Calculating correlations for {freq} Hz')
            ls_corrs_all_freqs, ls_pvals_all_freqs = pearson_calculator(working_df, 
                                                                        freq, 
                                                                        substr, 
                                                                        ls_corrs_all_freqs, 
                                                                        ls_pvals_all_freqs)

        # Save correlation values of each sensor pair and each substr with all freqs separately
        substr_spec_corr_all_freqs_df = pd.DataFrame(ls_corrs_all_freqs, index=freqs)
        substr_spec_corr_all_freqs_df.to_csv(op.join(output_corr_substr_dir, f'{substr}_lat_spectra_substr_pearsonr.csv'))
        substr_spec_pval_all_freqs_df = pd.DataFrame(ls_pvals_all_freqs, index=freqs)
        substr_spec_pval_all_freqs_df.to_csv(op.join(output_corr_substr_dir, f'{substr}_lat_spectra_substr_pvals.csv'))
    
        # Freshen the variables for the next substr
        del substr_spec_corr_all_freqs_df
        del substr_spec_pval_all_freqs_df
