
# this code needs editing. use last_FINAL_sublist-vol-outliers-removed.csv 
# to plot histograms. remove the pdf.


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