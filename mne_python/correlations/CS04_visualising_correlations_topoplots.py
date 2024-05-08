# -*- coding: utf-8 -*-
"""
def freq_substr_corr_p_values(corr_dir, substr, freq, correlation_values_grad, correlation_values_mag, p_values_grad, p_values_mag, channel_index_grad, channel_index_mag, channel_drop_grad, channel_drop_mag):    
    """"""this definition inputs corr_dir that contains all sensor pair folders and
    all substr folders for which we calculated correlation and pvalues.
    then creates a list of those values for each substr, each frequency,
    and all sensor pairs
        # example
    corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise') 
    substr = 'Caud'
    correlations_values = []
    p_values = []""""""

    for sensor_pair_fname in os.listdir(corr_dir):
        if sensor_pair_fname.startswith('MEG') and not sensor_pair_fname.endswith('1'):  # check if the folder name starts with 'MEG' and is not magnetometer
            print(f'working on {sensor_pair_fname}')
            channel_index_grad.append(sensor_pair_fname[0:7])  
            channel_drop_grad.append(sensor_pair_fname[-7:])
            sensor_pair_path = os.path.join(corr_dir, sensor_pair_fname)
            substr_folder_path = os.path.join(sensor_pair_path, f'{substr}')
            
            # Loop through each correlation table in the substr folder
            for filename in os.listdir(substr_folder_path):
                
                if filename.endswith("spearmanr.csv") and not filename.startswith('._'):  # to ensure not reading hidden files
                    correlation_table_grad = pd.read_csv(os.path.join(substr_folder_path, filename))
            
                if filename.endswith("pvals.csv") and not filename.startswith('._'):              
                    pvals_table_grad = pd.read_csv(os.path.join(substr_folder_path, filename))

            # Get the correlation value for the desired frequency 
            correlation_value_grad = correlation_table_grad.loc[correlation_table_grad['Unnamed: 0'] == freq, '0'].values[0]
            p_value_grad = pvals_table_grad.loc[pvals_table_grad['Unnamed: 0'] == freq, '0'].values[0]

            # Append the correlation value to the list
            correlation_values_grad.append(correlation_value_grad)
            p_values_grad.append(p_value_grad)
        
        if sensor_pair_fname.startswith('MEG') and sensor_pair_fname.endswith('1'):  # check if the folder name starts with 'MEG' and is magnetometer
            print(f'working on {sensor_pair_fname}')
            channel_index_mag.append(sensor_pair_fname[0:7])  
            channel_drop_mag.append(sensor_pair_fname[-7:])
            sensor_pair_path = os.path.join(corr_dir, sensor_pair_fname)
            substr_folder_path = os.path.join(sensor_pair_path, f'{substr}')
            
            # Loop through each correlation table in the substr folder
            for filename in os.listdir(substr_folder_path):
                
                if filename.endswith("spearmanr.csv") and not filename.startswith('._'):  # to ensure not reading hidden files
                    correlation_table_mag = pd.read_csv(os.path.join(substr_folder_path, filename))
            
                if filename.endswith("pvals.csv") and not filename.startswith('._'):              
                    pvals_table_mag = pd.read_csv(os.path.join(substr_folder_path, filename))

            # Get the correlation value for the desired frequency 
            correlation_value_mag = correlation_table_mag.loc[correlation_table_mag['Unnamed: 0'] == freq, '0'].values[0]
            p_value_mag = pvals_table_mag.loc[pvals_table_mag['Unnamed: 0'] == freq, '0'].values[0]

            # Append the correlation value to the list
            correlation_values_mag.append(correlation_value_mag)
            p_values_mag.append(p_value_mag)

        else:
            print(f'{sensor_pair_fname} does not exist')

    return correlation_values_grad, correlation_values_mag, p_values_grad, p_values_mag, channel_index_grad, channel_index_mag, channel_drop_grad, channel_drop_mag
"""


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


def freq_substr_corr_p_values(corr_dir, substr, freq):    
    """
    This definition inputs corr_dir that contains all sensor pair folders and
    all substr folders for which we calculated correlation and pvalues.
    then creates a list of those values for each substr, each frequency,
    and all sensor pairs
    
    Parameters:
        corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise') 
        substr(str) = 'Caud'
        freq(int) = Frequency for which to retrieve correlation and p-values
    Returns:
        tuple: Tuple containing correlation and p-values for gradiometers and magnetometers, channel indices, and channel drops.
            - correlation_values_grad (list): Correlation values for gradiometers.
            - correlation_values_mag (list): Correlation values for magnetometers.
            - p_values_grad (list): P-values for gradiometers.
            - p_values_mag (list): P-values for magnetometers.
            - channel_index_grad (list): Channel indices for gradiometers.
            - channel_index_mag (list): Channel indices for magnetometers.
            - channel_drop_grad (list): Channel drops for gradiometers.
            - channel_drop_mag (list): Channel drops for magnetometers.
    """
    correlation_values_grad = []
    correlation_values_mag = []
    p_values_grad = []
    p_values_mag = []
    channel_index_grad = []
    channel_index_mag = []
    channel_drop_grad = []
    channel_drop_mag = []
    
    for sensor_pair_fname in os.listdir(corr_dir):
        if not sensor_pair_fname.startswith('MEG'):
            continue
        
        sensor_pair_path = os.path.join(corr_dir, sensor_pair_fname)
        substr_folder_path = os.path.join(sensor_pair_path, substr)

        if not os.path.isdir(substr_folder_path):
            print(f'{substr_folder_path} does not exist')
            continue
        
        print(f'Working on {sensor_pair_fname}')
        
        # Determine whether the sensor pair is a magnetometer or a gradiometer
        is_magnetometer = sensor_pair_fname.endswith('1')

        # Update channel index and drop lists
        if is_magnetometer:
            channel_index_mag.append(sensor_pair_fname[0:7])  
            channel_drop_mag.append(sensor_pair_fname[-7:])
        else:
            channel_index_grad.append(sensor_pair_fname[0:7])  
            channel_drop_grad.append(sensor_pair_fname[-7:])
        
        # Loop through each correlation table in the substr folder
        for filename in os.listdir(substr_folder_path):
            if not filename.endswith("spearmanr.csv") or filename.startswith('._'):
                continue

            # Read correlation and p-value tables
            correlation_table = pd.read_csv(os.path.join(substr_folder_path, filename))
            pvals_table = pd.read_csv(os.path.join(substr_folder_path, filename.replace('spearmanr', 'spearman_pvals')))
            
            # Get the correlation value for the desired frequency
            correlation_value = correlation_table.loc[correlation_table['Unnamed: 0'] == freq, '0'].values[0]
            p_value = pvals_table.loc[pvals_table['Unnamed: 0'] == freq, '0'].values[0]

            # Update the corresponding lists
            if is_magnetometer:
                correlation_values_mag.append(correlation_value)
                p_values_mag.append(p_value)
            else:
                correlation_values_grad.append(correlation_value)
                p_values_grad.append(p_value)

    return (correlation_values_grad, correlation_values_mag, 
            p_values_grad, p_values_mag, 
            channel_index_grad, channel_index_mag, 
            channel_drop_grad, channel_drop_mag)

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
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction')  # containing all sensor pair folders
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/Correlation topomaps/freqs/subtraction')

# List of the things for which you want topoplots
substrs = ['Thal', 'Caud']
#['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
freqs = [10,10.5]
#[10,10.5,11,11.5,12,12.5,60,60.5,61,61.5]

# Initialize a dictionary to store correlation values for each sensor pair and dfs for all frequencies dfs
correlations_dfs_grad = {}
correlations_dfs_mag = {}
pvals_dfs_grad = {}
pvals_dfs_mag = {}

for freq in freqs:
    for substr in substrs:
        # Loop through each sensor pair folder
        (correlation_values_grad, correlation_values_mag, 
            p_values_grad, p_values_mag, 
            channel_index_grad, channel_index_mag, 
            channel_drop_grad, channel_drop_mag) = freq_substr_corr_p_values(corr_dir, substr, freq)
        
        # Convert the dictionary to a DataFrame
        correlations_df_grad = pd.DataFrame(correlation_values_grad, 
                                            index=channel_index_grad, 
                                            columns=['Correlation'])
        correlations_df_mag = pd.DataFrame(correlation_values_mag, 
                                            index=channel_index_mag, 
                                            columns=['Correlation'])
        pvals_df_grad = pd.DataFrame(p_values_grad, 
                                     index=channel_index_grad, 
                                     columns=['Correlation'])
        pvals_df_mag = pd.DataFrame(p_values_mag, 
                                    index=channel_index_mag, 
                                    columns=['Correlation'])
        # Name the DataFrame with the frequency value
        correlations_dfs_grad[f'{freq}Hz_{substr}'] = correlations_df_grad
        correlations_dfs_mag[f'{freq}Hz_{substr}'] = correlations_df_mag
        pvals_dfs_grad[f'{freq}Hz_{substr}'] = pvals_df_grad
        pvals_dfs_mag[f'{freq}Hz_{substr}'] = pvals_df_mag

    del (correlations_df_grad, correlations_df_mag, 
         pvals_df_grad, pvals_df_mag)  # refresh for next freq


# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)
halfgradraw = raw.copy().pick(channel_index_grad)

magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot for grads
halfmagraw = raw.copy().pick(channel_index_mag)  # channels for plotting grads

for substr in substrs:
    for i, freq in enumerate(freqs):
        
        # Get correlation and p-value DataFrames for grad sensors
        corr_df_grad = correlations_dfs_grad[f'{freq}Hz_{substr}'].reindex(halfgradraw.ch_names)
        corr_grad_ls = corr_df_grad['Correlation'].to_list()
        pval_df_grad = pvals_dfs_grad[f'{freq}Hz_{substr}'].reindex(halfgradraw.ch_names)
        pval_grad_ls = [float(val) for val in pval_df_grad['Correlation']]
        mask_grad = np.array([val < 0.05 for val in pval_grad_ls], dtype=bool)
        
        # Get correlation and p-value DataFrames for mag sensors
        corr_df_mag = correlations_dfs_mag[f'{freq}Hz_{substr}'].reindex(halfmagraw.ch_names)
        corr_mag_ls = corr_df_mag['Correlation'].to_list()
        pval_df_mag = pvals_dfs_mag[f'{freq}Hz_{substr}'].reindex(halfmagraw.ch_names)
        pval_mag_ls = [float(val) for val in pval_df_mag['Correlation']]
        mask_mag = np.array([val < 0.05 for val in pval_mag_ls], dtype=bool)

        ax = plt.subplot(2, 4, freq+1)
        ax = plt.subplots(len(freqs), 2, figsize=(12, 12))  # create subplots for each frequency, with 2 columns for mag and grad
        # Plot grad sensors correlation
        im,_ = mne.viz.plot_topomap(corr_grad_ls, magraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_grad_ls), max(corr_grad_ls)), 
                            mask=mask_grad, 
                            image_interp='nearest', axes=axes[i, 0])  # use the axes for grad

        axes[i, 0].set_title(f'Grad: {freq}Hz')

        # Plot mag sensors correlation
        axes[i,1] = mne.viz.plot_topomap(corr_mag_ls, halfmagraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_mag_ls), max(corr_mag_ls)), 
                            mask=mask_mag, 
                            image_interp='nearest', axes=axes[i, 1])  # use the axes for mag

        axes[i, 1].set_title(f'Mag: {freq}Hz')
        ax.set_title(substr)
        ax.set_xlim(0, )
        cbar = plt.colorbar(im, orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=5)
        if ind > 3:
            cbar.set_label('Correlation Values')
    # Adjust subplot layout
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, 
                        wspace=0.4, hspace=0.2)

    plt.show()
    
    
    if not op.exists(op.join(fig_output_dir, substr)):
        os.mkdir(op.join(fig_output_dir, substr))
    #plt.savefig(op.join(correlation_path, f'{band}_correlations.svg'), format='svg', dpi=300)
    #plt.savefig(op.join(correlation_path, f'{band}_correlations.tiff'), format='tiff', dpi=300)
    plt.savefig(op.join(fig_output_dir, substr, f'{freq}_correlations.jpg'), format='jpg', dpi=300)
