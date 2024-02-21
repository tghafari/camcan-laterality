"""
====================================
C04_correlation_frequency_topo_plot:
    this script:
    1. navigates to correlations/sensor_pairs dir
    2. navigates to one sensor_pair folder
    3. reads csv file of correlation values
    for all frequencies and all substr
    4. for each substr plots frequencies vs.
    correlation values
    5. draws a line under the correlation values
    that are significant (from pval csv)
    6. saves the plot with the name of right sensor
    in substr directory
    6. loops over substr
    7. loops over sensor pairs
    
Written by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import pandas as pd
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt

platform = 'mac'

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    
# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs')
fig_output_dir = op.join(deriv_dir, 'correlations/figures/substr_correlation_freqs')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

for _, row in sensors_layout_names_df.head(1).iterrows():
    print(f'working on pair {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}')

    pearsonr_fname = op.join(corr_dir, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}', 
                             'lat_spectra_substr_pearsonr.csv')
    pval_fname = op.join(corr_dir, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}', 
                             'lat_spectra_substr_pvals.csv')
    pearsonr_freq_substr_df = pd.read_csv(pearsonr_fname)
    pearsonr_freq_substr_df = pearsonr_freq_substr_df.set_index('Unnamed: 0')  # set freqs as the index
    pearsonr_freq_substr_df.index.names = ['freqs']
    pval_freq_substr_df = pd.read_csv(pval_fname)
    pval_freq_substr_df = pval_freq_substr_df.set_index('Unnamed: 0')  # set freqs as the index
    pval_freq_substr_df.index.names = ['freqs']

    for substr in substrs:
        pearsonr_substr_series  = pearsonr_freq_substr_df[substr]
        pval_substr_series  = pval_freq_substr_df[substr]

        # Extract frequencies and correlation values - all arrays
        freq_substr = pearsonr_substr_series.index.values
        corr_val_substr = pearsonr_substr_series.values
        pval_substr = pval_substr_series.values

        # Plot correlation values
        plt.figure(figsize=(10, 6))
        plt.plot(freq_substr, corr_val_substr, marker='o', color='b', label='Correlation Values')

        # Highlight frequencies with p-values smaller than 0.05
        significant_freqs = freq_substr[pval_substr < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='k', linestyle='--', linewidth=1)

        plt.xlabel('Frequencies')
        plt.ylabel('Correlation Values')
        plt.title(f'Correlation Values for {substr}')
        plt.legend()
        plt.grid(True)

        # Save the figure
        fig_output_path = op.join(fig_output_dir, 
                                  f'{substr}')
        if not op.exists(fig_output_path):
            os.makedirs(fig_output_path)

        fig_output_fname = op.join(fig_output_path, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}.png')

        plt.savefig(fig_output_fname)

        plt.show()



