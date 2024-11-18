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


def freq_lateralised_power(lat_dir, freq):    
    """
    This definition inputs lat_dir that contains all sensor pair folders and
    all substr folders for which we calculated lateralisation power.
    then creates a list of those values for each frequency and all sensor pairs
    
    Parameters:
        lat_dir = op.join(deriv_dir, 'lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise') 
        freq(int) = Frequency for which to retrieve lateralised power

    Returns:
        tuple: Tuple containing lateralisation power for gradiometers and magnetometers, channel indices, and channel drops.
            - lateralisation_values_grad (list): Lateralisation values for gradiometers.
            - lateralisation_values_mag (list): Lateralisation values for magnetometers.
            - channel_index_grad (list): Channel indices for gradiometers.
            - channel_index_mag (list): Channel indices for magnetometers.
            - channel_drop_grad (list): Channel drops for gradiometers.
            - channel_drop_mag (list): Channel drops for magnetometers.
    """

    lateralisation_values_mag = []
    lateralisation_values_grad = []
    channel_index_mag = []
    channel_index_grad = []
    channel_drop_mag = []
    channel_drop_grad = []

    
    for sensor_pair_fname in os.listdir(lat_dir):

        if not sensor_pair_fname.startswith('MEG') or sensor_pair_fname.startswith('._'):
            continue

        # Determine whether the sensor pair is a magnetometer or a gradiometer
        is_magnetometer = sensor_pair_fname.endswith('1.csv')

        # Update channel index and drop lists
        if is_magnetometer:
            print(f'working on magnetometer {sensor_pair_fname}')
            channel_index_mag.append(sensor_pair_fname[8:-4])  
            channel_drop_mag.append(sensor_pair_fname[0:7])
        else:
            print(f'working on gradiometer {sensor_pair_fname}')
            channel_index_grad.append(sensor_pair_fname[8:-4])  
            channel_drop_grad.append(sensor_pair_fname[0:7])
        
        # Loop through each correlation table in the substr folder
            # Read power lateralisation tables
            lat_table = pd.read_csv(os.path.join(lat_dir, sensor_pair_fname))
            
            # Get the lateralised power for the desired frequency
            lat_value = np.array(lat_table[str(float(freq))].values)

            lat_power_sensor_pair_freq = lat_table["Unnamed :0"],lat_value

            # # Update the corresponding lists
            # if is_magnetometer:
            #     lateralisation_values_mag.append(lat_value)
            # else:
            #     lateralisation_values_grad.append(lat_value)

    return (lateralisation_values_mag, 
            lateralisation_values_grad, 
            channel_index_mag, channel_index_grad,
            channel_drop_mag, channel_drop_grad)

def average_across_freqs(lateralisation_values_mag, 
                         lateralisation_values_grad,
                         freq_band):
    """
    this function averages over frequencies of the band of interes.

    freq_band(str) = the frequency band over which we average the frequencies. None if full spectrum is requiered
    """
    if freq_band == 'alpha':
        l_freq = 8
        h_freq = 12

        for idx, freq in enumerate(np.arange(l_freq, h_freq, 1)):
            filtered_values_mag = lateralisation_values_mag[str(float(freq))].value
            freq_band_values = lateralisation_values_mag.loc[:, str(l_freq):str(h_freq)]

        filtered_values_grad = lateralisation_values_grad.loc[
            (lateralisation_values_grad['Unnamed: 0'] >= l_freq) & 
            (lateralisation_values_grad['Unnamed: 0'] <= h_freq), 
            '0'
            ]
        
        avg_lat_value_mag = filtered_values_mag.mean()
        avg_lat_value_grad = filtered_values_grad.mean()

    else:  # can be modified for all frequency bands
        l_freq = 1
        h_freq = 40

        filtered_values_mag = lateralisation_values_mag.loc[
            (lateralisation_values_mag['Unnamed: 0'] >= l_freq) & 
            (lateralisation_values_mag['Unnamed: 0'] <= h_freq), 
            '0'
            ]
        
        filtered_values_grad = lateralisation_values_grad.loc[
            (lateralisation_values_grad['Unnamed: 0'] >= l_freq) & 
            (lateralisation_values_grad['Unnamed: 0'] <= h_freq), 
            '0'
            ]
        
        avg_lat_value_mag = filtered_values_mag.mean()
        avg_lat_value_grad = filtered_values_grad.mean()

    return avg_lat_value_mag, avg_lat_value_grad
            
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
deriv_dir = op.join(rds_dir, 'derivatives/meg/sensor') 
lat_dir = op.join(deriv_dir, 'lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise')  # containing all sensor pair files
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/sensor-pair-freq-substr-correlations_subtraction_nooutlier-psd/lateralised_power_plots')

# List of the things for which you want topoplots
freqs = np.arange(7,15,1)

# Initialize a dictionary to store lateralisation  values for each sensor pair and dfs for all frequencies dfs
lateralisations_dfs_mag = {}
lateralisations_dfs_grad = {}
# if one subject only
lateralisation_df_mag_all_freqs = []
lateralisation_df_grad_all_freqs = [] 
# Create lists of correlation values and pvalues for meg and grad for each freq and substr
for freq in freqs:
    # Loop through each sensor pair folder
    (lateralisation_values_mag, 
    lateralisation_values_grad, 
    channel_index_mag, channel_index_grad,
    channel_drop_mag, channel_drop_grad) = freq_lateralised_power(lat_dir, freq)

    # Convert the dictionary to a DataFrame
    lateralisations_df_mag = pd.DataFrame(lateralisation_values_mag, 
                                        index=channel_index_mag, 
                                        columns=['power_lateralisation'])
    lateralisations_df_grad = pd.DataFrame(lateralisation_values_grad, 
                                        index=channel_index_grad, 
                                        columns=['power_lateralisation'])
    # Name the DataFrame with the frequency value
    lateralisations_dfs_mag[f'{freq}Hz'] = lateralisations_df_mag
    lateralisations_dfs_grad[f'{freq}Hz'] = lateralisations_df_grad

    # for one subject only
    lateralisation_df_mag_all_freqs.append(lateralisations_df_mag) 
    lateralisation_df_grad_all_freqs.append(lateralisations_df_grad) 


    del lateralisations_df_mag, lateralisations_df_grad  # refresh for next freq

alpha_lateralisaion_mag, alpha_lateralisation_grad = average_across_freqs(lateralisation_df_mag_all_freqs, # work out the dictionary vs list vs df slicing
                                                                          lateralisation_df_grad_all_freqs,
                                                                          freq_band='alpha')

# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)
halfgradraw = raw.copy().pick(channel_index_grad)

magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot for grads
halfmagraw = raw.copy().pick(channel_index_mag)  # channels for plotting mags

for f, freq in enumerate(freqs):
    fig, axes = plt.subplots(1, 2)
    print(f'plotting lateralisation power in {freq}Hz')

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