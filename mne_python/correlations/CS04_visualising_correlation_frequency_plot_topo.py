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
import mne

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

# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, r'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

# Create a placeholder for correlation values of all sensor pairs
freqs = np.arange(1,60.125,0.125)

for substr in substrs:
    print(f'working on {substr}')
    correlation_df = pd.DataFrame(index=freqs)  # frequencies of correlations = index of pearsonr_freq_substr_df

    for _, row in sensors_layout_names_df.head(5).iterrows():
        print(f'working on pair {row["right_sensors"][0:8]}, {row["left_sensors"][0:8]}')

        pearsonr_fname = op.join(corr_dir, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}', 
                                f'{substr}', 'lat_spectra_substr_pearsonr.csv')
        pval_fname = op.join(corr_dir, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}', 
                                f'{substr}', 'lat_spectra_substr_pvals.csv')
        pearsonr_freq_substr_df = pd.read_csv(pearsonr_fname)
        pearsonr_freq_substr_df = pearsonr_freq_substr_df.set_index('Unnamed: 0')  # set freqs as the index
        pearsonr_freq_substr_df.index.names = ['freqs']
        pval_freq_substr_df = pd.read_csv(pval_fname)
        pval_freq_substr_df = pval_freq_substr_df.set_index('Unnamed: 0')  # set freqs as the index
        pval_freq_substr_df.index.names = ['freqs']
 
        # Extract frequencies and correlation values - all arrays
        freq_substr = pearsonr_freq_substr_df.index.values
        pval_substr = pval_freq_substr_df.values.flatten()
        corr_val_substr = pearsonr_freq_substr_df.values.flatten()
        
        # Fit a quadratic polynomial (degree=2)
        poly_fit = np.polyfit(freq_substr, corr_val_substr, 4)
        poly_line_func = np.poly1d(poly_fit)

        # Evaluate polynomial function at frequencies
        poly_line = poly_line_func(freq_substr)

        # Plot correlation values
        plt.figure(figsize=(10, 6))
        plt.plot(freq_substr, corr_val_substr, marker='o', color='lightgrey', label='Correlation Values', zorder=1)
        plt.plot(freq_substr, poly_line, linewidth=2, color='darkgrey', label='Polynomial Line (degree=4)', zorder=2)
        plt.title(f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}')
        plt.legend()
        
        """
        # Highlight frequencies with p-values smaller than 0.05  - needs p-values to be calculated again
        significant_freqs = freq_substr[pval_substr < 0.05]
        for freq in significant_freqs:
            plt.axvline(x=freq, color='k', linestyle='--', linewidth=1)

        plt.xlabel('Frequencies')
        plt.ylabel('Correlation Values')
        plt.title(f'Correlation Values for {substr}')
        plt.legend()
        plt.grid(True)
        """
        # Save the figure
        fig_output_path = op.join(fig_output_dir, f'{substr}')
        if not op.exists(fig_output_path):
            os.makedirs(fig_output_path)

        fig_output_fname = op.join(fig_output_path, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}.png')
        #plt.savefig(fig_output_fname)
        plt.close()
        
        # Put all correlations of sensor pairs in one df to create evoekd object for evoked_plot_topo
        # Concatenate correlation values along columns
        correlation_df = pd.concat([correlation_df, pearsonr_freq_substr_df],axis=1)

    # Rename columns with sensor pair names
    all_right_names = [f'{row["right_sensors"][0:8]}' for _, row in sensors_layout_names_df.head(5).iterrows()]
    correlation_df.columns = all_right_names

    # Create an EvokedArray object from the DataFrame
    rightraw = raw.copy().pick(all_right_names)
    evoked = mne.EvokedArray(correlation_df.values.T, rightraw.info, tmin=0, comment=f'pearsonr')

    # Plot the correlation values with similar format as plot_topo
    evoked.plot_topo(title=f"correlation between frequency and {substr} laterality")


