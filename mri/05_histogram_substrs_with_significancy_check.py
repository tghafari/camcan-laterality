# -*- coding: utf-8 -*-
"""
===============================================
05. histogram substrs

This code read the lateralized volumes from a 
csv file, plots a histogram for each substr and
checks for significant differences with normal 
distribution

written by Tara Ghafari
==============================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as op
from scipy import stats
from scipy.stats import shapiro

# Load the lateralization index sheet
volume_sheet_dir = r'X:\derivatives\mri\lateralized_index'
lat_sheet_fname = op.join(volume_sheet_dir, 'lateralization_volumes.csv')
df = pd.read_csv(lat_sheet_fname)
lateralization_volume = df.iloc[:,1:8].to_numpy()

colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
p_values = []
p_values_shapiro =[]

# null hypothesis (H0) mean value
throw_out_outliers = False
null_hypothesis_mean = 0.0
t_stats = []
t_p_vals = []

# wilcoxon p-vals
null_hypothesis_median = 0.0
wilcox_p_vals = []

fig, axs = plt.subplots(2, 4)
fig.set_figheight(6)
fig.set_figwidth(10)

for his in range(7):       
    # Define plot settings
    ax = axs[his // 4, his % 4]
    ax.set_title(structures[his], fontsize=12, fontname='Calibri')
    ax.set_xlabel('Lateralization Volume', fontsize=12, fontname='Calibri')
    ax.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
    # ax.set_ylim([1, 28])
    ax.axvline(x=0, color='k', linewidth=0.25, linestyle=':')
    
    # Remove nans and plot normalized (z-scored) distributions
    valid_lateralization_volume = lateralization_volume[~np.isnan(lateralization_volume[:, his]), his]
    lateralization_volume_hist = np.histogram(valid_lateralization_volume[:260], bins=10, density=False)
    
    # Throw out the outliers
    if throw_out_outliers:
        mean_lateralization_volume = np.nanmean(valid_lateralization_volume)
        std_lateralization_volume = np.nanstd(valid_lateralization_volume)
        threshold = mean_lateralization_volume - (2.5 * std_lateralization_volume)
        valid_lateralization_volume[:][valid_lateralization_volume[:] <= threshold] = np.nan
    
    # Perform the ranksum test
    k2, p = stats.normaltest(valid_lateralization_volume, nan_policy='omit')
    p_values.append(p)
    stat, shapiro_p = shapiro(valid_lateralization_volume)
    p_values_shapiro.append(shapiro_p)
    
    # 1 sample t-test for left/right lateralisation
    t_statistic, t_p_value = stats.ttest_1samp(valid_lateralization_volume, null_hypothesis_mean)
    t_stats.append(t_statistic)
    t_p_vals.append(t_p_value)
    
    # one sample wilcoxon signed rank (for non normal distributions)
    _, wilcox_p = stats.wilcoxon(valid_lateralization_volume - null_hypothesis_median,
                                 zero_method='wilcox', correction=False)
    wilcox_p_vals.append(wilcox_p)

    # plot histogram
    x = lateralization_volume_hist[1]
    y = lateralization_volume_hist[0]
    ax.bar(x[:-1], y, width=np.diff(x), color=colormap[his])
    
    # plot a normal density function
    mu, std = stats.norm.fit(valid_lateralization_volume)
    pdf = stats.norm.pdf(x, mu, std)
    ax.plot(x, pdf, 'r-', label='Normal Fit', linewidth=0.5)        
    
    txt = r'$p = {:.2f}$'.format(p)
    ax.text(min(ax.get_xlim()) + 0.2 * max(ax.get_xlim()), max(ax.get_ylim()) - 40, txt,
            fontsize=10, fontname='Calibri')
    txt = r'$1samp_p = {:.2f}$'.format(t_p_value)
    ax.text(min(ax.get_xlim()) + 0.2 * max(ax.get_xlim()), max(ax.get_ylim()) - 50, txt,
            fontsize=10, fontname='Calibri')
    txt = r'$wilcox_p = {:.2f}$'.format(wilcox_p)
    ax.text(min(ax.get_xlim()) + 0.2 * max(ax.get_xlim()), max(ax.get_ylim()) - 60, txt,
            fontsize=10, fontname='Calibri')
    
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_axisbelow(True)
    # ax.legend()
    
plt.tight_layout()
plt.show()
