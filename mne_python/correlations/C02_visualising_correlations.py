# -*- coding: utf-8 -*-
"""
====================================
C02_visualising_correlations:
    this script:
    1. reads the correlation csv file
     from each frequency band
    2. creates 7(substr) by 4(frequency
     bands) topoplots

Written by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import numpy as np
import os.path as op
import pandas as pd

import mne
from mne.channels import read_layout
import matplotlib.pyplot as plt

# Define the directory  
base_deriv_dir = r'/rds/projects/q/quinna-camcan/derivatives'
correlation_path = op.join(base_deriv_dir, 'correlations')

# Load one sample meg file for channel names
meg_fname =  r'/rds/projects/q/quinna-camcan/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'
raw = mne.io.read_raw_fif(meg_fname)
magraw = raw.copy().pick_types(meg='mag')  # this is to be able to show negative values on topoplot
raw.pick_types(meg='grad')

bands = ['delta', 'theta', 'alpha', 'beta']
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

for f, band in enumerate(bands):
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
                            #names=raw.ch_names)
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
