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
    the frequency band of interest
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


def band_substr_corr_p_values(corr_dir, substr, band):
    """
    This function inputs corr_dir that contains all sensor pair folders and
    all substr folders for which we calculated correlation and p-values.
    Then creates a list of those values for each substr, each frequency band,
    and all sensor pairs.

    Parameters:
        corr_dir (str): Directory containing all sensor pair folders.
        substr (str): Subcortical structure, e.g., 'Caud'.
        band (tuple): Frequency band range as a tuple, e.g., (8, 12) for alpha band.

    Returns:
        tuple: Tuple containing correlation and p-values for gradiometers and magnetometers, 
               channel indices, and channel drops.
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
            channel_index_mag.append(sensor_pair_fname[-7:])  
            channel_drop_mag.append(sensor_pair_fname[0:7])
        else:
            channel_index_grad.append(sensor_pair_fname[-7:])  
            channel_drop_grad.append(sensor_pair_fname[0:7])
        
        # Loop through each correlation table in the substr folder
        for filename in os.listdir(substr_folder_path):
            if not filename.endswith("spearmanr.csv") or filename.startswith('._'):
                continue

            # Read correlation and p-value tables
            correlation_table = pd.read_csv(os.path.join(substr_folder_path, filename))
            pvals_table = pd.read_csv(os.path.join(substr_folder_path, filename.replace('spearmanr', 'spearman_pvals')))
            
            # Filter the correlation and p-value tables for the desired frequency band
            band_corr_values = correlation_table.loc[(correlation_table['Unnamed: 0'] >= band[0]) & 
                                                     (correlation_table['Unnamed: 0'] <= band[1]), '0']
            band_pvals = pvals_table.loc[(pvals_table['Unnamed: 0'] >= band[0]) & 
                                         (pvals_table['Unnamed: 0'] <= band[1]), '0']
            
            # Calculate the mean correlation value and p-value for the band
            correlation_value = band_corr_values.mean()
            p_value = band_pvals.mean()

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
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nooutlier-psd')  # containing all sensor pair folders
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/Correlation_topomaps/bands/subtraction_nonoise_nooutliers-psd')

# List of the things for which you want topoplots
substrs = ['Thal', 'Puta', 'Pall', 'Amyg', 'Accu']

# Define frequency bands (for example: delta, theta, alpha, beta, gamma)
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 40)
}

# Initialize a dictionary to store correlation values for each sensor pair and dfs for all bands
correlations_dfs_grad = {}
correlations_dfs_mag = {}
pvals_dfs_grad = {}
pvals_dfs_mag = {}

# Create lists of correlation values and p-values for each band and substr
for band_name, band_range in bands.items():
    for substr in substrs:
        # Loop through each sensor pair folder
        (correlation_values_grad, correlation_values_mag, 
            p_values_grad, p_values_mag, 
            channel_index_grad, channel_index_mag, 
            channel_drop_grad, channel_drop_mag) = band_substr_corr_p_values(corr_dir, substr, band_range)
        
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
        # Name the DataFrame with the band name
        correlations_dfs_grad[f'{band_name}_{substr}'] = correlations_df_grad
        correlations_dfs_mag[f'{band_name}_{substr}'] = correlations_df_mag
        pvals_dfs_grad[f'{band_name}_{substr}'] = pvals_df_grad
        pvals_dfs_mag[f'{band_name}_{substr}'] = pvals_df_mag

    del (correlations_df_grad, correlations_df_mag, 
         pvals_df_grad, pvals_df_mag)  # refresh for next band


# Load one sample MEG file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)
halfgradraw = raw.copy().pick(channel_index_grad)

magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot for grads
halfmagraw = raw.copy().pick(channel_index_mag)  # channels for plotting grads

for substr in substrs:
    for band_name in bands.keys():
        fig, axes = plt.subplots(1, 2)
        print(f'Plotting correlation values for {substr} in {band_name} band')

        # Get correlation and p-value DataFrames for grad sensors
        corr_df_grad = correlations_dfs_grad[f'{band_name}_{substr}'].reindex(halfgradraw.ch_names)
        corr_grad_np = corr_df_grad['Correlation'].to_numpy()
        corr_grad_half = (corr_grad_np[::2] + corr_grad_np[1::2]) / 2  # to be able to use maginfo for displaying negative values on topoplot
        pval_df_grad = pvals_dfs_grad[f'{band_name}_{substr}'].reindex(halfgradraw.ch_names)
        pval_grad_ls = [float(val) for val in pval_df_grad['Correlation']]
        mask_grad = np.array([val < 0.05 for val in pval_grad_ls], dtype=bool)
        mask_grad_half = mask_grad[::2]  # same reason as line [195]
        
        # Get correlation and p-value DataFrames for mag sensors
        corr_df_mag = correlations_dfs_mag[f'{band_name}_{substr}'].reindex(halfmagraw.ch_names)
        corr_mag_ls = corr_df_mag['Correlation'].to_list()
        pval_df_mag = pvals_dfs_mag[f'{band_name}_{substr}'].reindex(halfmagraw.ch_names)
        pval_mag_ls = [float(val) for val in pval_df_mag['Correlation']]
        mask_mag = np.array([val < 0.05 for val in pval_mag_ls], dtype=bool)

        # Plot grad sensors correlation on left
        im,_ = mne.viz.plot_topomap(corr_grad_half, halfmagraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_grad_np), max(corr_grad_np)), 
                            mask=mask_grad_half, 
                            image_interp='nearest', 
                            axes=axes[0],
                            show=False)  # use the axes for grad

        axes[0].set_title(f'{substr} with grad: {band_name}')
        axes[0].set_xlim(0, )  # remove the left half of topoplot

        # Plot mag sensors correlation on right
        im, _ = mne.viz.plot_topomap(corr_mag_ls, halfmagraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_mag_ls), max(corr_mag_ls)), 
                            mask=mask_mag, 
                            image_interp='nearest', 
                            axes=axes[1],
                            show=False)  # use the axes for mag

        axes[1].set_title(f'{substr} with mag: {band_name}')
        axes[1].set_xlim(0, )  # remove the left half of topoplot
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label('Correlation Values')

        fig.set_size_inches(12, 12)

        if not op.exists(op.join(fig_output_dir, substr)):
            os.mkdir(op.join(fig_output_dir, substr))
        plt.savefig(op.join(fig_output_dir, substr, f'{band_name}_correlations.jpg'), format='jpg', dpi=300)
        plt.close()
