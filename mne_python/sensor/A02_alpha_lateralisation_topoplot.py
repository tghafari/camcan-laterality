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

written by Tara Ghafari
t.ghafari@bham.ac.uk
===================================
"""

import numpy as np
import pandas as pd
import os
import os.path as op

import mne
import matplotlib.pyplot as plt

# Define frequency bands
freq_bands = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30)
}
       
platform = 'mac'
# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

# Define directories 
deriv_dir = op.join(rds_dir, 'derivatives/meg/sensor') 
lat_dir = op.join(deriv_dir, 'lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise')

# Load the first file to extract the number of subjects
sample_file = op.join(lat_dir, sorted(os.listdir(lat_dir))[10])
subject_ids = pd.read_csv(sample_file)['Unnamed: 0']

# Initialize empty DataFrames for each frequency band with sensor names as columns
band_tables = {
    band: pd.DataFrame(index=subject_ids, 
                       columns=[file_name[8:-4] for file_name in os.listdir(lat_dir) if file_name.startswith('MEG')])
        for band in freq_bands
    }

channel_index_grad = []
channel_index_mag = []
 
# Process all sensor pair CSV files
sensor_files = sorted(os.listdir(lat_dir))  # Ensure files are processed in order
for sensor_idx, file_name in enumerate(sensor_files):
    if not file_name.startswith('MEG') or file_name.startswith('._'):
        continue

    sensor_name = file_name[8:-4]
 
   # Determine whether the sensor pair is a magnetometer or a gradiometer
    is_magnetometer = sensor_name.endswith('1')

    # Update channel index and drop lists
    if is_magnetometer:
        channel_index_mag.append(sensor_name)  
    else:
        channel_index_grad.append(sensor_name)  

    file_path = op.join(lat_dir, file_name)
    # Load the table (589 rows x 240 columns) and set 'Unnamed: 0' as the index
    table = pd.read_csv(file_path)
    subject_ids = table['Unnamed: 0']  # Extract subject IDs
    table = table.drop(columns=["Unnamed: 0"])  # Drop the non-frequency column

    # Convert column names to float for proper comparison
    table.columns = table.columns.astype(float)

    # Ensure alignment of indices between `band_tables` and `table`
    if not band_tables["Delta"].index.equals(subject_ids):
        print(f"Aligning indices for {file_name}")
        subject_ids = subject_ids.reset_index(drop=True)

    # Loop over frequency bands and compute the averages
    for band, (low, high) in freq_bands.items():
        # Select columns corresponding to the frequency range
        freq_cols = table.loc[:, (table.columns >= low) & (table.columns < high)]
        # Average over the selected frequencies and assign to the corresponding column
        band_tables[band].loc[:, sensor_name] = freq_cols.mean(axis=1).values
        # table_fname = op.join(deriv_dir, f'lateralized_index/{band}_lateralised_power_allsens_subtraction_nonoise.csv')
        # band_tables[band].to_csv(table_fname, index=False)  # shape: (586,153)


# Function to plot alpha lateralized power for the right gradiometers
def plot_alpha(subject_id, band_tables, meg_fpath, channel_index_grad, channel_index_mag, band):
    """
    Plot the lateralized power for alpha band on right gradiometers.

    Parameters:
    - subject_id (int): Index of the subject.
    - band_table (pd.DataFrame): Table of input band lateralized power (subjects x sensors). e.g. band_tables["Alpha"]
    - raw_data_path (str): Path to the sample MEG .fif file.
    - channel_index_grad (list): Indices for gradiometer channels.
    - channel_index_mag (list): Indices for magnetometer channels.
    """
    # Load MEG channel names
    raw = mne.io.read_raw_fif(meg_fpath)
    halfgradraw = raw.copy().pick(channel_index_grad)
    magraw = raw.copy().pick('mag')
    halfmagraw = magraw.copy().pick(channel_index_mag)
    
    # Get the alpha power for the given subject
    band_values_mag_ls = band_tables[band].loc[subject_id].reindex(halfmagraw.ch_names).to_list()
    band_values_grad_ls = band_tables[band].loc[subject_id].reindex(halfgradraw.ch_names).to_list()
    
    # Create a single figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

    # Plot mags topomap on the right subplot
    im_mag, _ = mne.viz.plot_topomap(
        band_values_mag_ls, 
        halfmagraw.info, 
        contours=0,
        cmap='RdBu_r',
        vlim=(min(band_values_mag_ls), max(band_values_mag_ls)), 
        image_interp='nearest', 
        axes=axes[1],
        show=False)  

    axes[1].set_xlim(0, )  # remove the left half of topoplot
    axes[1].set_title(f"{band} Lateralised Power (Mags) - Subject {subject_id}")

    # Plot grads topomap on the left subplot
    im_grad, _ = mne.viz.plot_topomap(
        band_values_grad_ls, 
        halfgradraw.info, 
        contours=0,
        cmap='RdBu_r',
        vlim=(min(band_values_grad_ls), max(band_values_grad_ls)), 
        image_interp='nearest', 
        axes=axes[0],
        show=False)  
    
    axes[0].set_xlim(0, )  # remove the left half of topoplot
    axes[0].set_title(f"{band} Lateralised Power (Grads) - Subject {subject_id}")

    # Add colorbars below each subplot
    fig.colorbar(im_mag, ax=axes[1], orientation='horizontal', label='Alpha Lateralised Power')
    fig.colorbar(im_grad, ax=axes[0], orientation='horizontal', label='Alpha Laterliased Power')


    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


subject_id = 120469  # Replace with the actual subject ID
meg_fpath =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')

plot_alpha(subject_id, band_tables, meg_fpath, channel_index_grad, channel_index_mag, band="Alpha")