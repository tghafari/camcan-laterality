# -*- coding: utf-8 -*-
"""
====================================
CS04_visualising_correlations_topoplots:
    this script:
    the goal is to creat topo plots using 
    correlation values instead of power

    1. navigates to the correlation directory
    2. then lists all the folders (with 
    sensor pair names)
    3. then navigates to each of the sensor
    pair folders 
    4. then inside each of the subcortical
    structure folders
    5. then opens the correlation table
    6. finds the correlation value for
    the frequency of interest
    7. put it in a list of all correlation
    values for that substr and all sensor pairs
    8. does the same for p-value
    

Written by Andrew Quinn
Adapted by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import pandas as pd
import numpy as np
import os.path as op
import os
import matplotlib.pyplot as plt
import mne

platform = 'bluebear'

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    
# Define directories 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise')  # containing all sensor pair folders
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/sensor-pair-freq-substr-correlations_subtraction-nonoise')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# List of the things for which you want topoplots
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
freqs = [10,10.5,11,11.5,12,12.5,60,60.5,61,61.5]
# np.arange(1,120.5,0.5)  # all frequencies available in spectra

# Initialize a dictionary to store correlation values for each sensor pair
metrics = {}

# Loop through each sensor pair folder
for sensor_pair_folder in os.listdir(corr_dir):
    sensor_pair_path = os.path.join(corr_dir, sensor_pair_folder)
    
    # Check if the current item is a directory
    if os.path.isdir(sensor_pair_path):
        for substr in substrs:
            substr_folder_path = os.path.join(sensor_pair_path, f'{substr}')
        
        # Check if the 'caudate' folder exists
        if os.path.exists(caudate_folder_path):
            # Initialize a list to store correlation values for the current sensor pair
            correlation_values = []
            
            # Loop through each frequency file in the 'caudate' folder
            for filename in os.listdir(caudate_folder_path):
                # Check if the file is a CSV file
                if filename.endswith(".csv"):
                    # Read the correlation table into a DataFrame
                    correlation_table = pd.read_csv(os.path.join(caudate_folder_path, filename))
                    
                    # Get the correlation value for the desired frequency (e.g., 10Hz)
                    # Assuming the frequency column in the correlation table is named 'Frequency'
                    correlation_value = correlation_table.loc[correlation_table['Frequency'] == '10Hz', 'Correlation'].values[0]
                    
                    # Append the correlation value to the list
                    correlation_values.append(correlation_value)
            
            # Store the correlation values for the current sensor pair in the dictionary
            metrics[sensor_pair_folder] = correlation_values

# Convert the dictionary to a DataFrame
metrics_df = pd.DataFrame(metrics)







    
# Define the directory 

# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)

magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot
raw.pick_types(meg='grad')


for f, band in enumerate(freqs):
    correlation_band_path = op.join(correlation_path, band)
    pearsonrs_csv_file = op.join(correlation_band_path, 'lat_spectra_substr_pearsonr.csv')
    pvals_csv_file = op.join(correlation_band_path, 'lat_spectra_substr_pvals.csv')

    # Load correlation files and rename first column
    
    pearson_df = pearson_df.rename(columns={'Unnamed: 0': 'ch_names'})
    pval_df = pd.read_csv(pvals_csv_file)
    pval_df = pval_df.rename(columns={'Unnamed: 0': 'ch_names'})
    print(f'opening correlations csv file for {band} band and reading channel names')

    # Here we read both channel names from each pair
    ch_names = [chs.split(' - ')[0] for chs in pearson_df['ch_names']] + [chs.split(' - ')[1] for chs in pearson_df['ch_names']]

    plt.figure(f)
    for ind, substr in enumerate(substrs):
        print(f' plotting correlation values for {substr}')

        # Read channel indices of the correlation table from loaded fif file
        metrics = np.zeros((204,))
        pvals = np.zeros((204,))
        for idx, chan in enumerate(ch_names):
            ind_chan = mne.pick_channels(raw.ch_names, [chan])
            ind_val = np.where(np.array([name.find(chan) for name in pearson_df['ch_names']]) > -1 )[0]
            metrics[ind_chan] = pearson_df[substr].values[ind_val]
            pvals[ind_chan] = pval_df[substr].values[ind_val]
        
        # Define those correlations that are significant - average across grads to halve the number of channels
        mask = pvals < 0.05
        metrics_grad = (metrics[::2] + metrics[1::2]) / 2 

        # Subplot each substr
        ax = plt.subplot(2, 4, ind+1)
        im, _ = mne.viz.plot_topomap(metrics_grad, magraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(metrics), max(metrics)), 
                            axes=ax, mask=mask[::2],
                            image_interp='nearest')
                            # names=raw.ch_names)
                            # mask_params={'marker': '*'},
        ax.set_title(substr)
        ax.set_xlim(0, )
        cbar = plt.colorbar(im, orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=5)
        if ind > 3:
            cbar.set_label('Correlation Values')
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
                        wspace=0.4, hspace=0.2)
    plt.figure(f).figsize=(12, 12)
    #plt.show()
    plt.savefig(op.join(correlation_path, f'{band}_correlations.svg'), format='svg', dpi=300)
    plt.savefig(op.join(correlation_path, f'{band}_correlations.tiff'), format='tiff', dpi=300)
    plt.savefig(op.join(correlation_path, f'{band}_correlations.jpg'), format='jpg', dpi=300)
