# -*- coding: utf-8 -*-
"""
====================================
CS02_correlation_frequency_plot_topo:
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
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    
# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nooutlier-psd')
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/sensor-pair-freq-substr-correlations_subtraction_nooutlier-psd/average-sensors')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')

# Load one sample meg file for channel names
meg_fname =  op.join(rds_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif')
raw = mne.io.read_raw_fif(meg_fname)

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Subcortical structure and their associated sensor clusters
substr_sensclusters = {
    'Thal': [['0431','1141'], ['1821','2211'], ['1841', '2231']], 
    'Puta': [['0531', '0941'], ['0541', '0931'], ['0511', '0921'],
             ['0321', '1231'], ['0341', '1221']], 
    'Hipp': [['0532','0942'], ['0611','1021'], ['0541','0931'],
             ['0331','1241'], ['0321','1231'], ['0341','1221'], 
             ['0121','1411']]
}
# colormap = ['#FFD700', '#191970', '#6B8E23']  # structure color map from fslanat
colormap = ['darkred', 'darkred', 'darkblue']

# Frequencies of interest
freqs = np.arange(1, 41, 1)

for idx, substr in enumerate(substr_sensclusters.keys()):
    print(f'Working on {substr}')
    
    # Placeholder for averaged correlation values and p-values
    avg_corr_vals = np.zeros(len(freqs))
    avg_pvals = np.zeros(len(freqs))
    num_sensors = len(substr_sensclusters[substr])

    for sensor_id in substr_sensclusters[substr]:
        print(f'Processing sensor {sensor_id}')
        
        # Build file paths
        pearsonr_fname = op.join(corr_dir, f'MEG{sensor_id[0]}_MEG{sensor_id[1]}', f'{substr}', f'{substr}_lat_spectra_substr_spearmanr.csv')
        pval_fname = op.join(corr_dir, f'MEG{sensor_id[0]}_MEG{sensor_id[1]}', f'{substr}', f'{substr}_lat_spectra_substr_spearman_pvals.csv')
        
        if not op.exists(pearsonr_fname) or not op.exists(pval_fname):
            print(f'Files for sensor {sensor_id} not found. Skipping.')
            continue
        
        # Load correlation and p-value data
        corr_df = pd.read_csv(pearsonr_fname, index_col=0)
        #pval_df = pd.read_csv(pval_fname, index_col=0)

        # Ensure data is aligned with the frequencies
        corr_vals = corr_df.loc[freqs, '0'].values
        #pvals = pval_df.loc[freqs, '0'].values

        # Accumulate correlation values and p-values
        avg_corr_vals += corr_vals
        #avg_pvals += pvals

    # Average the correlation values and p-values across sensors
    avg_corr_vals /= num_sensors
    #avg_pvals /= num_sensors

    # Plot averaged correlation values
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, avg_corr_vals, marker='o', color=colormap[idx], zorder=1)
    
    # Highlight frequencies where p-values < 0.05  -- find a reasonable way to find significant p-values. average doesn't make sense
    # significant_freqs = freqs[avg_pvals < 0.05]
    # for freq in significant_freqs:
    #     plt.axvline(x=freq, color='red', alpha=0.5, linestyle='--', linewidth=1, label='p < 0.05')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Averaged Correlation Values')
    plt.title(f'Averaged Correlation Values for {substr} Across Sensors')
    plt.legend()
    plt.grid(True)

    # Save the figure
    fig_output_path = op.join(fig_output_dir, f'{substr}')
    if not op.exists(fig_output_path):
        os.makedirs(fig_output_path)
    
    fig_output_fname = op.join(fig_output_path, f'{substr}_avg_corr_values.png')
    plt.savefig(fig_output_fname, dpi=500)
    plt.close()

print('Processing complete.')
