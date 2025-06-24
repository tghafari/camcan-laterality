# -*- coding: utf-8 -*-
"""
====================================
CS01_sensorwise_correlating_lateralised_spectra_substr:

Define Frequency Bands: Create ranges for delta (1-4Hz), 
theta (4-8Hz), alpha (9-14Hz), and beta (15-40Hz).
Calculate Average Power in Each Band: For each frequency 
band, calculate the average power across the corresponding frequencies.
Calculate Correlations: Use these average power values to calculate 
correlations with the subcortical structure's lateralization index.

    
Written by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import pandas as pd
import numpy as np
import os.path as op
import os
import scipy.stats as stats

def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """Merge the dataframes containing spectrum lateralization values and 
    subcortical structure lateralization values together."""

    # Navigate to the sensor_pair folder
    spec_lat_index_fname = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')

    # Load lateralization index for each pair
    spectrum_pair_lat_df = pd.read_csv(spec_lat_index_fname)
    spectrum_pair_lat_df = spectrum_pair_lat_df.rename(columns={'Unnamed: 0':'subject_ID'})
    
    # Merge and match the subject_ID column and remove nans
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['subject_ID'])
    working_df = working_df.dropna()

    # Get the freqs of spectrum from spec_pair_lat
    freqs = spectrum_pair_lat_df.columns.values[1:]  # remove subject_ID column
    freqs = [float(freq) for freq in freqs]  # convert strings to floats
    return working_df, freqs

def calculate_band_power(working_df, freqs, band):
    """Calculate the average power within a specified frequency band."""
    # Round frequencies to one decimal place to match the column names in working_df
    freqs_rounded = [round(f, 1) for f in freqs]

    # Select frequencies that fall within the band range
    band_freqs = [f for f in freqs_rounded if band[0] <= f <= band[1]]
    
    # Ensure the selected frequencies are actually in the DataFrame columns
    band_freqs = [str(f) for f in band_freqs if str(f) in working_df.columns]

    # Check if there are any valid frequencies selected
    if len(band_freqs) == 0:
        raise ValueError(f"No frequencies found in the range {band[0]}-{band[1]} Hz in the data.")
    
    # Calculate the average power across the selected frequencies
    band_power = working_df[band_freqs].mean(axis=1)
    
    return band_power

def correlation_calculator(working_df, band_power, substr, 
                           ls_corrs_all_bands, ls_pvals_all_bands):
    """Calculate Pearson and Spearman correlations between band power and subcortical structure lateralization index."""

    print(f'Calculating correlations for {substr} and frequency band')

    # Calculate Spearman correlation
    spear_temp_corr, spear_temp_pvalue = stats.spearmanr(band_power, working_df[substr].to_numpy()) 
    ls_corrs_all_bands.append(spear_temp_corr)
    ls_pvals_all_bands.append(spear_temp_pvalue)

    return ls_corrs_all_bands, ls_pvals_all_bands

# Define where to read and write the data
platform = 'bluebear'  # bluebear or mac?
rds_dir = '/rds/projects/q/quinna-camcan' if platform == 'bluebear' else '/Volumes/quinna-camcan'

# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
spectra_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_nooutliers_absolute-thresh')  #psd outliers removed
substr_dir = op.join(deriv_dir, 'mri/lateralized_index')
substr_sheet_fname = op.join(substr_dir, 'lateralization_volumes.csv')  # later we used lateralization_volumes_nooutliers where outliers where smaller than 10% centile.
lat_sheet_fname_nooutlier = op.join(substr_dir, 'lateralization_volumes_nooutliers.csv')  # vol and psd outliers removed
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# Load subcortical lateralization data
substr_lat_df = pd.read_csv(substr_sheet_fname)

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Define frequency bands
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 14),  # different than CB04 plotting script
    'Beta': (14, 40)
}

# List of subcortical structures
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

for i, row in sensors_layout_names_df.iterrows():
    print(f'Working on pair {row["left_sensors"][0:8]}, {row["right_sensors"][0:8]}')

    # Get the frequencies of spectrum (only once is enough)
    _, freqs = working_df_maker(spectra_dir, 
                                row["left_sensors"][0:8], 
                                row["right_sensors"][0:8], 
                                substr_lat_df) if i == 0 else (None, freqs)

    # Make the working df containing lateralized value of the current sensor pair
    working_df, _ = working_df_maker(spectra_dir,  
                                    row["left_sensors"][0:8], 
                                    row["right_sensors"][0:8], 
                                    substr_lat_df)
    
    output_corr_dir = op.join(deriv_dir, 'correlations/bands_sensor_pairs_subtraction_nooutlier-psd',
                               f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}')
    if not op.exists(output_corr_dir):
        os.makedirs(output_corr_dir)
 
    # Calculate correlation in each substr
    for substr in substrs:
        print(f'Working on {substr}')

        # Predefine lists
        ls_corrs_all_bands = []
        ls_pvals_all_bands = []
        
        output_corr_substr_dir = op.join(output_corr_dir, f'{substr}')
        if not op.exists(output_corr_substr_dir):
            os.makedirs(output_corr_substr_dir)
    
        # Calculate correlation with each frequency band
        for band_name, band_range in bands.items():
            print(f'Calculating correlations for {band_name} band ({band_range[0]}-{band_range[1]} Hz)')
            band_power = calculate_band_power(working_df, freqs, band_range)
            ls_corrs_all_bands, ls_pvals_all_bands = correlation_calculator(working_df, 
                                                                            band_power, 
                                                                            substr, 
                                                                            ls_corrs_all_bands, 
                                                                            ls_pvals_all_bands)

        # Save correlation values of each sensor pair and each substr with all frequency bands separately
        substr_spec_corr_all_bands_df = pd.DataFrame(ls_corrs_all_bands, index=bands.keys())
        substr_spec_corr_all_bands_df.to_csv(op.join(output_corr_substr_dir, f'{substr}_lat_spectra_substr_spearmanr.csv'))
        substr_spec_pval_all_bands_df = pd.DataFrame(ls_pvals_all_bands, index=bands.keys())
        substr_spec_pval_all_bands_df.to_csv(op.join(output_corr_substr_dir, f'{substr}_lat_spectra_substr_spearman_pvals.csv'))
    
        # Freshen the variables for the next substr
        del substr_spec_corr_all_bands_df
        del substr_spec_pval_all_bands_df
