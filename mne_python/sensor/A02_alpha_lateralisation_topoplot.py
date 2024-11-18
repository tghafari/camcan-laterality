"""
====================================
A02_alpha_lateralisation_topoplot:
    the goal is to creat topo plots using 
    lateralised power of alpha
    1. first navigate to the folder containing
    the lateralisation index of interest
    2. average over the frequencies
    that constitute your band of
    interest.
    3. use the info from a raw object
    and the indices of the channels
    in the order of the lateralised
    powers to visualise the lateralised 
    power (instead of power) on the
    right half of the topoplot.


===================================
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
            channel_index_mag.append(sensor_pair_fname[-7:])  
            channel_drop_mag.append(sensor_pair_fname[0:7])
        else:
            channel_index_grad.append(sensor_pair_fname[-7:])  
            channel_drop_grad.append(sensor_pair_fname[0:7])
        
        # Loop through each correlation table in the substr folder
        for filename in os.listdir(substr_folder_path):
            if not filename.endswith("spearmanr.csv") or filename.startswith('._'):
                continue
                
            # without changing the name of correlation_table, replace this with the lateralised spec table
            # then average over the band frequencies to have 153 lateralised powers per band
            # # Read correlation and p-value tables
            # correlation_table = pd.read_csv(os.path.join(substr_folder_path, filename))
            # pvals_table = pd.read_csv(os.path.join(substr_folder_path, filename.replace('spearmanr', 'spearman_pvals')))
            
            # # Get the correlation value for the desired frequency
            # correlation_value = correlation_table.loc[correlation_table['Unnamed: 0'] == freq, '0'].values[0]
            # p_value = pvals_table.loc[pvals_table['Unnamed: 0'] == freq, '0'].values[0]

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
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/Correlation_topomaps/freqs/subtraction_nonoise_nooutliers-psd')

# List of the things for which you want topoplots
substrs = ['Thal', 'Puta', 'Pall', 'Amyg', 'Accu']
freqs = np.arange(1,40,1)

# Initialize a dictionary to store correlation values for each sensor pair and dfs for all frequencies dfs
correlations_dfs_grad = {}
correlations_dfs_mag = {}
pvals_dfs_grad = {}
pvals_dfs_mag = {}

# Create lists of correlation values and pvalues for meg and grad for each freq and substr
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
halfmagraw = raw.copy().pick(channel_index_mag)  # channels for plotting mags

for substr in substrs:
    for f, freq in enumerate(freqs):
        fig, axes = plt.subplots(1, 2)
        print(f'plotting correlations values for {substr} in {freq}Hz')

        # Get correlation and p-value DataFrames for grad sensors
        corr_df_grad = correlations_dfs_grad[f'{freq}Hz_{substr}'].reindex(halfgradraw.ch_names)
        corr_grad_np = corr_df_grad['Correlation'].to_numpy()
        corr_grad_half = (corr_grad_np[::2] + corr_grad_np[1::2]) / 2  # to be able to use maginfo for displaying negative values on topoplot
        pval_df_grad = pvals_dfs_grad[f'{freq}Hz_{substr}'].reindex(halfgradraw.ch_names)
        pval_grad_ls = [float(val) for val in pval_df_grad['Correlation']]
        mask_grad = np.array([val < 0.05 for val in pval_grad_ls], dtype=bool)
        mask_grad_half = mask_grad[::2]  # same reason as line [195]
        
        # Get correlation and p-value DataFrames for mag sensors
        corr_df_mag = correlations_dfs_mag[f'{freq}Hz_{substr}'].reindex(halfmagraw.ch_names)
        corr_mag_ls = corr_df_mag['Correlation'].to_list()
        pval_df_mag = pvals_dfs_mag[f'{freq}Hz_{substr}'].reindex(halfmagraw.ch_names)
        pval_mag_ls = [float(val) for val in pval_df_mag['Correlation']]
        mask_mag = np.array([val < 0.05 for val in pval_mag_ls], dtype=bool)

        # Plot grad sensors correlation on left
        im,_ = mne.viz.plot_topomap(corr_grad_half, halfmagraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_grad_np), max(corr_grad_np)), 
                            mask=mask_grad_half, 
                            image_interp='nearest', 
                            axes=axes[0],
                            show=False)  # use the axes for grad

        axes[0].set_title(f'{substr} with grad: {freq}Hz')
        axes[0].set_xlim(0, )  # remove the left half of topoplot

        # Plot mag sensors correlation on right
        im, _ = mne.viz.plot_topomap(corr_mag_ls, halfmagraw.info, contours=0,
                            cmap='RdBu_r', vlim=(min(corr_mag_ls), max(corr_mag_ls)), 
                            mask=mask_mag, 
                            image_interp='nearest', 
                            axes=axes[1],
                            show=False)  # use the axes for mag

        axes[1].set_title(f'{substr} with mag: {freq}Hz')
        axes[1].set_xlim(0, )  # remove the left half of topoplot
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal', location='bottom')
        cbar.ax.tick_params(labelsize=5)
        cbar.set_label('Correlation Values')

        fig.set_size_inches(12, 12)

        if not op.exists(op.join(fig_output_dir, substr)):
            os.mkdir(op.join(fig_output_dir, substr))
        #plt.savefig(op.join(correlation_path, f'{band}_correlations.svg'), format='svg', dpi=300)
        #plt.savefig(op.join(correlation_path, f'{band}_correlations.tiff'), format='tiff', dpi=300)
        plt.savefig(op.join(fig_output_dir, substr, f'{freq}_correlations.jpg'), format='jpg', dpi=300)
        plt.close()