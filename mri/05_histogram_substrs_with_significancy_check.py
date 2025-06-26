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
outlier_subjectID_psd_csv = op.join(info_dir, 'outlier_subjectID_psd_df.csv')
final_sub_list_path = op.join(info_dir, 'FINAL_sublist-LV-LI-outliers-removed')  # list of subjects after removing size below 10th percentile and lateralised power nonoise_subtraction_abs_thresh

volume_sheet_dir = 'derivatives/mri/lateralized_index'
lat_sheet_fname = op.join(rds_dir, volume_sheet_dir, 'lateralization_volumes.csv')
lat_sheet_fname_nooutlier = op.join(rds_dir, volume_sheet_dir, 'lateralization_volumes_nooutliers.csv')
lat_sheet_fname_final_subs = op.join(rds_dir, volume_sheet_dir, 'lateralization_volumes_final_subs.csv')
substr_vol_sheet_fname = op.join(rds_dir, volume_sheet_dir, 'all_subs_substr_volumes.csv')
output_plot_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/volume-plots')

# Define colormap and structures
colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']


def find_outliers(substr_vol_df, outlier_subjectID_vol_df, q1, q2):
    # Load data
    quantiles_df = substr_vol_df.iloc[:, 1:].quantile([q1, q2])
    quantiles_df.to_csv(op.join(info_dir, f"substr_volumes_{q1}-{q2}_quantiles.csv"))

    # Remove CC prefix from subject IDs and convert to strings for comparison
    substr_vol_df['subject_ID'] = substr_vol_df['subject_ID'].astype(str)

    # Iterate through each subject
    for index, row in substr_vol_df.iterrows():
        subjectID = row['subject_ID']
        print(subjectID)
        # Compare each subcortical structure's volume with the quantile values
        for structure in substr_vol_df.columns[1:]:  # Skip the first column (subject_ID)
            volume = row[structure]
            print(volume)
            # Ensure that q1 and q2 are used as float values for indexing
            quant1 = quantiles_df.loc[q1, structure]
            quant2 = quantiles_df.loc[q2, structure]
            print([quant1, quant2])
            # Check if the volume is an outlier
            if volume < quant1 or volume > quant2:
                print(f'{structure} in {subjectID} is an outlier') 
                temp_outlier_df = pd.DataFrame({'SubjectID': subjectID, 
                                                'outlier_structure': structure},
                                                 index=([0]))
                # Append the subject ID and structure name to the outlier DataFrame
                outlier_subjectID_vol_df = pd.concat([outlier_subjectID_vol_df, temp_outlier_df], ignore_index=True)
                del temp_outlier_df  # cleanup before moving on

    return outlier_subjectID_vol_df


# Function to plot subcortical volumes with left (L) and right (R) distinction
def preprocess_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet,
                                    outlier_subjectID_psd_csv, lat_sheet_fname, final_sub_list_path, q1=0.01, q2=1, old_outliers=False):
    """
    Plots histograms of subcortical volumes, with darker shades of the colormap for left (L) structures
    and lighter shades for right (R) structures.
    
    Parameters:
    - substr_vol_sheet_fname: str, path to the subcortical volumes CSV file
    - good_sub_sheet: str, path to the good subject CSV file
    - structures: list, names of the subcortical structures
    - colormap: list, color map for plotting histograms
    - q1 and q2: quantiles to be calculated for identifying outliers. Default q1=0.01, q2=1 (only 
    ignore smaller volumes)
    
    This function reads the subcortical volumes from the provided CSV file, removes subjects with missing
    volume measurements, and ensures only subjects from the good subject list are included in the analysis.
    Histograms are plotted for each subcortical structure with left (L) and right (R) sides distinguished
    by shades of the colormap.
    """
    
    if old_outliers:
        # Load volume data
        substr_vol_df = pd.read_csv(substr_vol_sheet_fname)
        good_subjects_df = pd.read_csv(good_sub_sheet)
        outlier_subjectID_psd_df = pd.read_csv(outlier_subjectID_psd_csv)
        outlier_subjectID_vol_df = pd.DataFrame(columns=['SubjectID', 'structure'])  # initialise a dataframe for volume outliers

        outlier_subjectID_vol_df = find_outliers(substr_vol_df, outlier_subjectID_vol_df, q1, q2)

        # Remove CC prefix from good subjects and convert to strings for comparison
        good_subjects_df['SubjectID'] = good_subjects_df['SubjectID'].str.replace('CC', '', regex=False).astype(str)
        substr_vol_df['subject_ID'] = substr_vol_df['subject_ID'].astype(str)
        
        # Filter out subjects with missing volume measurements and find and save q1 and q2 
        substr_vol_df = substr_vol_df.dropna()
        
        # Filter to include only good subjects, exclude psd and volume outliers
        substr_vol_df = substr_vol_df[substr_vol_df['subject_ID'].isin(good_subjects_df['SubjectID'])]
        substr_vol_df = substr_vol_df[~substr_vol_df['subject_ID'].isin(outlier_subjectID_psd_df['SubjectID'])]
        substr_vol_df = substr_vol_df[~substr_vol_df['subject_ID'].isin(outlier_subjectID_vol_df['SubjectID'])]

        # Load lateralisation volume data and preprocess
        substr_lat_df = pd.read_csv(lat_sheet_fname)
        substr_lat_df['subject_ID'] = substr_lat_df['subject_ID'].astype(str)

        # Filter out subjects with missing volume measurements and find and save q1 and q2 
        substr_lat_df = substr_lat_df.dropna()
        
        # Filter to include only good subjects, exclude psd and volume outliers
        substr_lat_df = substr_lat_df[substr_lat_df['subject_ID'].isin(good_subjects_df['SubjectID'])]
        substr_lat_df = substr_lat_df[~substr_lat_df['subject_ID'].isin(outlier_subjectID_psd_df['SubjectID'])]
        substr_lat_df = substr_lat_df[~substr_lat_df['subject_ID'].isin(outlier_subjectID_vol_df['SubjectID'])]

        # Save lateralisation index df without outliers
        substr_lat_df.to_csv(lat_sheet_fname_nooutlier)
    else:
        # this is with psd_vol new outliers
        
        # Load FINAL subject list
        final_sub_df = pd.read_csv(final_sub_list_path)  # Should contain column 'subject_ID'
        final_sub_df['subject_ID'] = final_sub_df['subject_ID'].astype(str)
        final_sub_ids = final_sub_df['subject_ID'].tolist()

        # Load volume data
        substr_vol_df = pd.read_csv(substr_vol_sheet_fname)
        substr_vol_df['subject_ID'] = substr_vol_df['subject_ID'].astype(str)  # ensure subject_IDs are strings
        substr_vol_df = substr_vol_df.dropna()

        # Filter to FINAL subject list 
        substr_vol_df = substr_vol_df[
                         substr_vol_df['subject_ID'].isin(final_sub_ids)]
        
        # Load lateralisation index data
        substr_lat_df = pd.read_csv(lat_sheet_fname)
        substr_lat_df['subject_ID'] = substr_lat_df['subject_ID'].astype(str)
        substr_lat_df = substr_lat_df.dropna()

        # Filter to FINAL subject list 
        substr_lat_df = substr_lat_df[
            substr_lat_df['subject_ID'].isin(final_sub_ids)]
        
        # Save filtered lateralisation index without outliers
        substr_lat_df.to_csv(lat_sheet_fname_final_subs, index=False)

    return substr_vol_df, substr_lat_df, outlier_subjectID_vol_df

def plot_volume_histograms(structures, substr_vol_df, colormap):
    # Plotting histograms
    fig, axs = plt.subplots(len(structures), 2, figsize=(12, 14))
    
    for idx, structure in enumerate(structures):
        # Extract L and R volumes as numpy arrays
        volume_L = substr_vol_df[f'L-{structure}'].values
        volume_R = substr_vol_df[f'R-{structure}'].values
        
        # Plot L side histogram with darker color on the left subplot
        ax_L = axs[idx, 0]
        volume_L_hist = np.histogram(volume_L, bins=10, density=False)
        x_L = volume_L_hist[1]
        y_L = volume_L_hist[0]
        ax_L.bar(x_L[:-1], y_L, width=np.diff(x_L), color=colormap[idx], alpha=0.7, label=f'L-{structure}')
        ax_L.set_title(f'L-{structure}', fontsize=12, fontname='Calibri')
        ax_L.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
        ax_L.tick_params(axis='both', which='both', length=0)
        ax_L.set_axisbelow(True)

        # Fit and plot a normal density function
        mu_L, std_L = stats.norm.fit(volume_L)
        pdf_L = stats.norm.pdf(x_L, mu_L, std_L)
        ax_L.plot(x_L, pdf_L * max(y_L) / max(pdf_L), 'r-', label='Normal Fit', linewidth=0.5)  # Scaled for comparison

        if idx == 6:
            ax_L.set_xlabel('Volume', fontsize=12, fontname='Calibri') 
        
        # Plot R side histogram with lighter color on the right subplot
        ax_R = axs[idx, 1]
        volume_R_hist = np.histogram(volume_R, bins=10, density=False)
        x_R = volume_R_hist[1]
        y_R = volume_R_hist[0]
        ax_R.bar(x_R[:-1], y_R, width=np.diff(x_R), color=colormap[idx], alpha=0.3, label=f'R-{structure}')
        ax_R.set_title(f'R-{structure}', fontsize=12, fontname='Calibri')
        ax_R.set_ylabel('# Subjects', fontsize=12, fontname='Calibri')
        ax_R.tick_params(axis='both', which='both', length=0)
        ax_R.set_axisbelow(True)

        # Fit a normal distribution and plot the PDF
        mu_R, std_R = stats.norm.fit(volume_R)
        pdf_R = stats.norm.pdf(x_R, mu_R, std_R)
        ax_R.plot(x_R, pdf_R * max(y_R) / max(pdf_R), 'r-', label='Normal Fit', linewidth=0.5)  # Scaled for comparison

        if idx == 6:
            ax_R.set_xlabel('Volume', fontsize=12, fontname='Calibri')
    
    plt.tight_layout()
    plt.savefig(op.join(output_plot_dir, 'substr-histograms-nooutliers.png'), dpi=300)

# Function to filter subjects and plot lateralization histograms
def plot_lateralisation_volumes(substr_lat_df, structures, colormap):
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

    # Extract lateralization volumes
    lateralization_volume = substr_lat_df.iloc[:,1:8].to_numpy()

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
        ax.set_xlabel('Lateralisation Volume', fontsize=12, fontname='Calibri')
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
    plt.savefig(op.join(output_plot_dir, 'lateralisation-histograms-nooutliers-nonormalcurve.png'), dpi=300)

(substr_vol_df, substr_lat_df, 
   outlier_subjectID_vol_df) = preprocess_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet, 
                                                              outlier_subjectID_psd_csv,lat_sheet_fname, 
                                                              q1=0.01, q2=1)
# Save outlier dataframe
outlier_subjectID_vol_df.to_csv(op.join(info_dir,'outlier_subjectID_vol_df.csv'))

# plot_volume_histograms(structures, substr_vol_df, colormap)
plot_lateralisation_volumes(substr_lat_df, structures, colormap)


