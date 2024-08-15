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


import os.path as op

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro

# Define paths
platform = 'mac'  # 'bluebear' or 'mac'?

if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

volume_sheet_dir = 'derivatives/mri/lateralized_index'
lat_sheet_fname = op.join(rds_dir, volume_sheet_dir, 'lateralization_volumes.csv')
substr_vol_sheet_fname = op.join(rds_dir, volume_sheet_dir, 'all_subs_substr_volumes.csv')

# Define colormap and structures
colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']


# Function to filter subjects and plot lateralization histograms
def filter_and_plot_volumes(lat_sheet_fname, good_sub_sheet, structures, colormap):
    """
    Filters subjects based on available lateralization volume measurements and good subject list,
    then plots histograms.
    
    Parameters:
    - lat_sheet_fname: str, path to the lateralization volumes CSV file
    - good_sub_sheet: str, path to the good subject CSV file
    - structures: list, names of the subcortical structures
    - colormap: list, color map for plotting histograms
    
    This function removes subjects with missing volume measurements and ensures only subjects from the
    good subject list are included in the analysis. Histograms are plotted for the lateralised volume of
    each subcortical structure.
    """

    # Load data
    df = pd.read_csv(lat_sheet_fname)
    good_subjects_df = pd.read_csv(good_sub_sheet)

    # Remove CC prefix from good subjects and convert to strings for comparison
    good_subjects_df['SubjectID'] = good_subjects_df['SubjectID'].str.replace('CC', '', regex=False).astype(str)
    df['subject_ID'] = df['subject_ID'].astype(str)
    
    # Filter out subjects with missing volume measurements
    df = df.dropna()
    
    # Filter to include only good subjects
    df = df[df['subject_ID'].isin(good_subjects_df['SubjectID'])]
   
    # Extract lateralization volumes
    lateralization_volume = df.iloc[:,1:8].to_numpy()

    # Preallocation
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

    # Plotting histograms
    fig, axs = plt.subplots(2, 4)
    fig.set_figheight(6)
    fig.set_figwidth(10)

    for idx, structure in enumerate(structures):
        ax = axs[idx // 4, idx % 4]
        ax.set_title(structure, fontsize=12, fontname='Calibri')
        ax.set_xlabel('Lateralization Volume', fontsize=12, fontname='Calibri')
        ax.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
        ax.axvline(x=0, color='k', linewidth=0.25, linestyle=':')
        
        valid_lateralization_volume = lateralization_volume[:, idx]
        lateralization_volume_hist = np.histogram(valid_lateralization_volume, bins=10, density=False)
        
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

        # Plot histogram with colors
        x = lateralization_volume_hist[1]
        y = lateralization_volume_hist[0]
        ax.bar(x[:-1], y, width=np.diff(x), color=colormap[idx])
        
        # Fit and plot a normal density function
        mu, std = stats.norm.fit(valid_lateralization_volume)
        pdf = stats.norm.pdf(x, mu, std)
        ax.plot(x, pdf, 'r-', label='Normal Fit', linewidth=0.5)        
        
        txt = r'$ranksum_p = {:.2f}$'.format(p)
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
        
    plt.tight_layout()
    plt.show()


# Function to plot subcortical volumes with left (L) and right (R) distinction
def plot_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet, structures, colormap):
    """
    Plots histograms of subcortical volumes, with darker shades of the colormap for left (L) structures
    and lighter shades for right (R) structures.
    
    Parameters:
    - substr_vol_sheet_fname: str, path to the subcortical volumes CSV file
    - good_sub_sheet: str, path to the good subject CSV file
    - structures: list, names of the subcortical structures
    - colormap: list, color map for plotting histograms
    
    This function reads the subcortical volumes from the provided CSV file, removes subjects with missing
    volume measurements, and ensures only subjects from the good subject list are included in the analysis.
    Histograms are plotted for each subcortical structure with left (L) and right (R) sides distinguished
    by shades of the colormap.
    """
    
    # Load data
    df = pd.read_csv(substr_vol_sheet_fname)
    good_subjects_df = pd.read_csv(good_sub_sheet)

    # Remove CC prefix from good subjects and convert to strings for comparison
    good_subjects_df['SubjectID'] = good_subjects_df['SubjectID'].str.replace('CC', '', regex=False).astype(str)
    df['subject_ID'] = df['subject_ID'].astype(str)
    
    # Filter out subjects with missing volume measurements
    df = df.dropna()
    
    # Filter to include only good subjects
    df = df[df['subject_ID'].isin(good_subjects_df['SubjectID'])]
    
    # Extract substr volumes
    substr_volume = df.iloc[:,1:8].to_numpy()

    # Plotting histograms
    fig, axs = plt.subplots(2, 4)
    fig.set_figheight(6)
    fig.set_figwidth(10)
    
    for idx, structure in enumerate(structures):
        ax = axs[idx // 4, idx % 4]
        ax.set_title(structure, fontsize=12, fontname='Calibri')
        ax.set_xlabel('Volume', fontsize=12, fontname='Calibri')
        ax.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
        
        # Extract L and R volumes
        volume_L = df[f'L-{structure}'].values
        volume_R = df[f'R-{structure}'].values
        
        # Plot L side histogram with darker color
        volume_L_hist = np.histogram(volume_L, bins=10, density=False)
        x_L = volume_L_hist[1]
        y_L = volume_L_hist[0]
        ax.bar(x_L[:-1], y_L, width=np.diff(x_L), color=colormap[idx], alpha=0.7, label=f'Left {structure}')
        
        # Plot R side histogram with lighter color
        volume_R_hist = np.histogram(volume_R, bins=10, density=False)
        x_R = volume_R_hist[1]
        y_R = volume_R_hist[0]
        ax.bar(x_R[:-1], y_R, width=np.diff(x_R), color=colormap[idx], alpha=0.3, label=f'Right {structure}')
        
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_axisbelow(True)
        ax.legend()
        
    plt.tight_layout()
    plt.show()

# Call the function to filter and plot lateralization volumes
filter_and_plot_volumes(lat_sheet_fname, good_sub_sheet, structures, colormap)

# Call the function to plot subcortical volumes
plot_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet, structures, colormap)









    
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
