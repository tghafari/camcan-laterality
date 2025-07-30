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
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')  # list of all subjects at the beginning
final_sub_list_path = op.join(info_dir, 'FINAL_sublist-vol-psd-outliers-removed.csv')  # list of subjects after removing size below 10th percentile and power nonoise_subtraction_abs_thresh

volume_sheet_dir = 'derivatives/mri/lateralized_index'
substr_vol_sheet_fname = op.join(sub2ctx_dir, volume_sheet_dir, 'all_subs_substr_volumes.csv')
lat_sheet_fname = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes.csv')  # this is lateralisation vol for all subjects that will be filtered to exclude vol outliers in this script
lat_sheet_fname_nooutlier = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes_no-vol-outliers.csv')  # only volume outlier
outlier_subjectID_vol_fname = op.join(info_dir, 'outlier_subjectID_vol.csv')  # this only contains vol outliers
subject_list_no_vol_outliers = op.join(info_dir, 'FINAL_sublist-vol-outliers-removed.csv')  # in case we only wanted to exclude vol outliers
output_plot_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/volume-plots')
# outlier_subjectID_psd_csv = op.join(info_dir, 'outlier_subjectID_psd_df.csv')  # don't remember how the outlier was calculated for psd here
# lat_sheet_fname_final_subs = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes_final_subs.csv')  # this is not used with old final list of subs

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
            if structure != 'BrStem /4th Ventricle':
                print(structure)
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
                    outlier_subjectID_vol_df.to_csv(outlier_subjectID_vol_fname)

    return outlier_subjectID_vol_df


# Function to plot subcortical volumes with left (L) and right (R) distinction
def preprocess_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet,
                                    lat_sheet_fname, q1=0.01, q2=1):
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
    
    # this is removing volume outliers (<0.01 quantile)
    # Load volume data
    substr_vol_df = pd.read_csv(substr_vol_sheet_fname)
    good_subjects_df = pd.read_csv(good_sub_sheet)
    # outlier_subjectID_psd_df = pd.read_csv(outlier_subjectID_psd_csv)
    outlier_subjectID_vol_df = pd.DataFrame(columns=['SubjectID', 'outlier_structure'])  # initialise a dataframe for volume outliers

    outlier_subjectID_vol_df = find_outliers(substr_vol_df, outlier_subjectID_vol_df, q1, q2)

    # Remove CC prefix from good subjects and convert to strings for comparison
    good_subjects_df['SubjectID'] = good_subjects_df['SubjectID'].str.replace('CC', '', regex=False).astype(str)
    substr_vol_df['subject_ID'] = substr_vol_df['subject_ID'].astype(str)
    
    # Filter out subjects with missing volume measurements and find and save q1 and q2 
    substr_vol_df = substr_vol_df.dropna()
    
    # Filter to include only good subjects, exclude psd and volume outliers
    substr_vol_df = substr_vol_df[substr_vol_df['subject_ID'].isin(good_subjects_df['SubjectID'])]
    # substr_vol_df = substr_vol_df[~substr_vol_df['subject_ID'].isin(outlier_subjectID_psd_df['SubjectID'])]
    substr_vol_df = substr_vol_df[~substr_vol_df['subject_ID'].isin(outlier_subjectID_vol_df['SubjectID'])]

    # Load lateralisation volume data and preprocess
    substr_lat_df = pd.read_csv(lat_sheet_fname)
    substr_lat_df['subject_ID'] = substr_lat_df['subject_ID'].astype(str)

    # Filter out subjects with missing volume measurements and find and save q1 and q2 
    substr_lat_df = substr_lat_df.dropna()
    
    # Filter to include only good subjects, exclude psd and volume outliers
    substr_lat_df = substr_lat_df[substr_lat_df['subject_ID'].isin(good_subjects_df['SubjectID'])]
    # substr_lat_df = substr_lat_df[~substr_lat_df['subject_ID'].isin(outlier_subjectID_psd_df['SubjectID'])]  # by removing these two lines I"m only removing volume outliers- will do the psd outliers in later stages
    substr_lat_df = substr_lat_df[~substr_lat_df['subject_ID'].isin(outlier_subjectID_vol_df['SubjectID'])]

    # Save lateralisation index df without outliers
    substr_lat_df.to_csv(lat_sheet_fname_nooutlier)

    # Save subject list with no volume outliers
    sublist = pd.DataFrame(columns=['SubjectID'])
    sublist = substr_lat_df['subject_ID']
    sublist.to_csv(subject_list_no_vol_outliers)

    return substr_vol_df, substr_lat_df, outlier_subjectID_vol_df, sublist

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
        ax_L.plot(x_L, pdf_L * max(y_L) / max(pdf_L), 'k-', label='Normal Fit', linewidth=0.5)  # Scaled for comparison

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
        ax_R.plot(x_R, pdf_R * max(y_R) / max(pdf_R), 'k-', label='Normal Fit', linewidth=0.5)  # Scaled for comparison

        if idx == 6:
            ax_R.set_xlabel('Volume', fontsize=12, fontname='Calibri')
    
    plt.tight_layout()
    plt.savefig(op.join(output_plot_dir, 'substr-histograms-subs_no-vol-outlier.png'), dpi=3000)

# Checking the normality of the distributions
def test_normality(structures, lateralization_volume):
    # Perform the normality test analytically and graphically:
    """ p > 0.05 is normal distribution""" 

    n_structures = len(structures)
    n_cols = 4
    n_rows = int(np.ceil(n_structures / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = axs.flatten()

    p_values_shapiro = []

    for idx, structure in enumerate(structures):
        ax = axs[idx]

        # Extract and clean data
        valid_lateralization_volume = lateralization_volume[:, idx].copy()
        clean_data = valid_lateralization_volume[~np.isnan(valid_lateralization_volume)]

        # Shapiro-Wilk test
        stat, p_shapiro = shapiro(clean_data)
        p_values_shapiro.append(p_shapiro)

        # Q-Q plot
        stats.probplot(clean_data, dist="norm", plot=ax)
        ax.set_xlabel('Theoretical Quantiles', fontsize=10, fontname='Calibri')
        ax.set_ylabel('Ordered Values', fontsize=10, fontname='Calibri')
        ax.set_title(structure, fontsize=12, fontname='Calibri')


        # Annotate p-value
        ax.text(0.05, 0.95,
                f'Shapiro p = {p_shapiro:.3f}',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))

        # Add large NORMAL / NOT NORMAL label
        label = 'NORMAL' if p_shapiro > 0.05 else 'NOT NORMAL'
        color = 'green' if label == 'NORMAL' else 'red'
        ax.text(0.5, 0.5,
                label,
                transform=ax.transAxes,
                fontsize=20,
                color=color,
                alpha=0.4,
                ha='center', va='center',
                fontweight='bold')

    # Remove empty axes
    for i in range(len(structures), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()

    return p_values_shapiro

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

    # null hypothesis (H0) mean value
    throw_out_outliers = False
    null_hypothesis_mean = 0.0
    p_vals = []  # is used for both wilcox and ttest

    # wilcoxon p-vals
    null_hypothesis_median = 0.0

    # Perform the normality test analytically and graphically:
    """ p > 0.05 is normal distribution""" 
    p_values_shapiro = test_normality(structures, lateralization_volume)
        
    # Plotting histograms
    n_structures = len(structures)
    n_cols = 4
    n_rows = int(np.ceil(n_structures / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axs = axs.flatten()
    fig.set_figheight(6)
    fig.set_figwidth(10)

    for idx, structure in enumerate(structures):
        # ax = axs[idx // 4, idx % 4]
        ax = axs[idx]
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
        
        if p_values_shapiro[idx] > 0.05:
            # 1 sample t-test for left/right lateralisation (for normal distributions)
            _, t_p_value = stats.ttest_1samp(valid_lateralization_volume, null_hypothesis_mean)
            p_vals.append(t_p_value)

        elif p_values_shapiro[idx] < 0.05:
            # one sample wilcoxon signed rank (for non normal distributions)
            _, wilcox_p = stats.wilcoxon(valid_lateralization_volume - null_hypothesis_median,
                                            zero_method='wilcox', correction=False)
            p_vals.append(wilcox_p)

        # Plot histogram with colors
        x = lateralization_volume_hist[1]
        y = lateralization_volume_hist[0]
        ax.bar(x[:-1], y, width=np.diff(x), color=colormap[idx])
        
        # Fit and plot a normal density function
        mu, std = stats.norm.fit(valid_lateralization_volume)
        pdf = stats.norm.pdf(x, mu, std)
        ax.plot(x, pdf, 'k-', label='Normal Fit', linewidth=0.5)       

        # Annotate p-value
        ax.text(0.05, 0.95,
                f'p-value = {p_vals[idx]:.3f}',
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.6)) 
        
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_axisbelow(True)
    
    # Remove empty axes
    for i in range(len(structures), len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.savefig(op.join(output_plot_dir, 'lateralisation-histograms-final_subs_no-vol-outliers_nonormalcurve.png'), dpi=3000)

(substr_vol_df, substr_lat_df, 
 outlier_subjectID_vol_df, sublist) = preprocess_subcortical_volumes(substr_vol_sheet_fname, good_sub_sheet,
                                                                        lat_sheet_fname, q1=0.01, q2=1)

plot_volume_histograms(structures, substr_vol_df, colormap)
plot_lateralisation_volumes(substr_lat_df, structures, colormap)


