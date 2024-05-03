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

def freq_substr_corr_p_values(corr_dir, substr, correlation_values, p_values):    
    """this definition inputs corr_dir that contains all sensor pair folders and
    all substr folders for which we calculated correlation and pvalues.
    then creates a list of those values for each substr, each frequency,
    and all sensor pairs
        # example
    corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise') 
    substr = 'Caud'
    correlations_all_pairs = {}
    pvals_all_pairs = {}"""

    for sensor_pair_fname in os.listdir(corr_dir):
        if sensor_pair_fname.startswith('MEG'):  # check if the folder name starts with 'MEG'
            channel_index.append(sensor_pair_fname[0:7])  
            channel_drop.append(sensor_pair_fname[-7:])
            sensor_pair_path = os.path.join(corr_dir, sensor_pair_fname)
            substr_folder_path = os.path.join(sensor_pair_path, f'{substr}')
            
            # Loop through each correlation table in the substr folder
            for filename in os.listdir(substr_folder_path):
                
                if filename.endswith("spearmanr.csv") and not filename.startswith('._'):  # to ensure not reading hidden files
                    correlation_table = pd.read_csv(os.path.join(substr_folder_path, filename))
            
                if filename.endswith("pvals.csv") and not filename.startswith('._'):              
                    pvals_table = pd.read_csv(os.path.join(substr_folder_path, filename))

            # Get the correlation value for the desired frequency 
            correlation_value = correlation_table.loc[correlation_table['Unnamed: 0'] == freq, '0'].values[0]
            p_value = pvals_table.loc[pvals_table['Unnamed: 0'] == freq, '0'].values[0]

            # Append the correlation value to the list
            correlation_values.append(correlation_value)
            p_values.append(p_value)

            # Store the correlation values for the current sensor pair in the dictionary - don't think it's needed
            # correlations_all_pairs[sensor_pair_fname] = correlation_value
            # pvals_all_pairs[sensor_pair_fname] = p_value

        else:
            print(f'{sensor_pair_path} does not exist')

    return correlation_values, p_values, channel_index, channel_drop
    
platform = 'mac'

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
substrs = ['Caud']
#['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
freqs = [10,10.5]
#[10,10.5,11,11.5,12,12.5,60,60.5,61,61.5]
# np.arange(1,120.5,0.5)  # all frequencies available in spectra

# Initialize a dictionary to store correlation values for each sensor pair and dfs for all frequencies dfs
correlations_dfs = {}
pvals_dfs = {}

for freq in freqs:

    # Initialize a list to store correlation values for the current frequency
    #correlations_all_pairs = {}
    #pvals_all_pairs = {}
    correlation_values = []
    p_values = []
    channel_index = [] 
    channel_drop = []  # sensor names to drop from info object (plotting)

    #for substr in substrs:
        # Loop through each sensor pair folder


    # Convert the dictionary to a DataFrame
    correlations_df = pd.DataFrame(correlation_values, index=channel_index, columns=['Correlation'])
    pvals_df = pd.DataFrame(p_values, index=channel_index, columns=['Correlation'])

    # Name the DataFrame with the frequency value
    correlations_dfs[f'df_{freq}Hz'] = correlations_df
    pvals_dfs[f'df_{freq}Hz'] = pvals_df

    del correlations_df, pvals_df







    
# Define the directory 

# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)

halfraw = raw.copy().drop_channels(channel_drop)  # this is to be able to show negative values on topoplot
raw.pick_types(meg='grad')




for f, band in enumerate(freqs):


    # Load correlation files and rename first column
    
    pearson_df = pearson_df.rename(columns={'Unnamed: 0': 'ch_names'})
    pval_df = pd.read_csv(pvals_csv_file)
    pval_df = pval_df.rename(columns={'Unnamed: 0': 'ch_names'})
    print(f'opening correlations csv file for {band} band and reading channel names')

    # Here we read both channel names from each pair
    ch_names = channel_index
    #[chs.split(' - ')[0] for chs in pearson_df['ch_names']] + [chs.split(' - ')[1] for chs in pearson_df['ch_names']]

    plt.figure(f)
    for ind, substr in enumerate(substrs):
        print(f' plotting correlation values for {substr}')

        # Read channel indices of the correlation table from loaded fif file
        metrics = np.zeros((204,))
        pvals = np.zeros((204,))
        for idx, chan in enumerate(ch_names):
            ind_chan = mne.pick_channels(halfraw.ch_names, [chan])
            ind_val = np.where(np.array([name.find(chan) for name in pearson_df.index]) > -1 )[0]
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
