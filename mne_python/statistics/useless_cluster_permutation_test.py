"""
===============================================
S01. Cluster permutation test

This script corrects for multiple comparisons
in the correlation df with cluster permutation 
test.
steps:
    1. Load the correlation values from the 
    CSV file into a numpy array.
    2. Create a connectivity matrix to define 
    the neighborhood relationships between sensors.
    3. Perform the cluster permutation test.
    4. Plot the results.

written by Tara Ghafari
==============================================
"""

import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_test, permutation_t_test
import matplotlib.pyplot as plt
import os.path as op

# Define the directory  
base_deriv_dir = r'/rds/projects/q/quinna-camcan/derivatives'
correlation_path = op.join(base_deriv_dir, 'correlations')

# Create a sample info for connectivity matrix
meg_fname =  r'/rds/projects/q/quinna-camcan/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'
raw = mne.io.read_raw_fif(meg_fname)
magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot
raw.pick_types(meg='grad')  # this has to be here for channel indices

# Load correlation values from CSV file into a numpy array
bands = ['delta', 'theta', 'alpha', 'beta']
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

for f, band in enumerate(bands):
    correlation_band_path = op.join(correlation_path, band)
    pearsonrs_csv_file = op.join(correlation_band_path, 'lat_spectra_substr_pearsonr.csv')

    # Read correlation files
    pearson_df = pd.read_csv(pearsonrs_csv_file)
    pearson_df = pearson_df.rename(columns={'Unnamed: 0': 'ch_names'})
    print(f'opening correlations csv file for {band} band and reading channel names')

    # Here we read both channel names from each pair
    ch_names = [chs.split(' - ')[0] for chs in pearson_df['ch_names']] + [chs.split(' - ')[1] for chs in pearson_df['ch_names']]

    plt.figure(f)
    for ind, substr in enumerate(substrs):

        print(f'running cluster permutation tests on {substr}')

        # Read channel indices of the correlation table from loaded fif file
        metrics = np.zeros((204,))
        for idx, chan in enumerate(ch_names):
            ind_chan = mne.pick_channels(raw.ch_names, [chan])
            ind_val = np.where(np.array([name.find(chan) for name in pearson_df['ch_names']]) > -1 )[0]
            metrics[ind_chan] = pearson_df[substr].values[ind_val]

        # Average across grads to halve the number of channels
        metrics_grad = (metrics[::2] + metrics[1::2]) / 2 

        # Create an adjacency matrix defining neighborhood relationships between sensors
        adjacency = mne.channels.find_ch_adjacency(magraw.info, 'grad')

        # Define the clustering parameters
        threshold = None  # 'an F-threshold will be chosen automatically 
                        # that corresponds to a p-value of 0.05 for 
                        # the given number of observations' 
        T_obs, clusters, p_values, _ = permutation_t_test(
                metrics_grad,  # The data for the test
                threshold=threshold,  # Threshold for cluster formation
                n_permutations=1000,  # Number of permutations
                adjacency=adjacency,  # adjacency matrix
                verbose=True)  # Print progress

        # Try permutation_t_test
        T_obs, p_values, H0 = permutation_t_test(
                metrics_grad,
                n_permutations=10000, 
                tail=0, 
                verbose=True)

        # Plot the results for each substr
        print(f'plotting cluster tests and p values')
        plt.figure(figsize=(8,4))
        plt.suptitle(f'cluster permutation test for {substr} in {band}', fontsize=16)

        # Plot p-values of the test
        plt.subplot(1, 2, 1)
        plt.title('Cluster p-values')
        plt.hist(p_values, bins=20)
        plt.xlabel('p-values')
        plt.ylabel('Frequency')

        # Plot t-statistics observed from the test
        plt.subplot(1, 2, 2)
        plt.title('Cluster permutation t-statistics')
        plt.imshow(T_obs, cmap='coolwarm', origin='lower', 
                    extent=[0, len(metrics_grad), 0, len(metrics_grad)],
                    aspect='auto')
        plt.colorbar()
        plt.xlabel('channels')
        plt.ylabel('channels')






















