# -*- coding: utf-8 -*-
"""
Plot lateralisation volume indices across participants for 7 subcortical structures.
Saves figure as TIFF/PNG/SVG (dpi=800). Each subplot shows a Wilcoxon p-value vs 0.
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------- Config / Paths ----------------------- #
platform = 'mac'  # 'mac' or 'bluebear'

if platform == 'bluebear':
    quinna_dir = '/rds/projects/q/quinna-camcan'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    fig_output_root = '/rds/projects/j/jenseno-avtemporal-attention/Manuscript/Figures'
elif platform == 'mac':
    quinna_dir = '/Volumes/quinna-camcan'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    fig_output_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures'
else:
    raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

# Data files (adjust if your layout differs)
volume_sheet_dir = 'derivatives/mri/lateralized_index'
lat_csv_all        = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes.csv')
lat_csv_no_outlier = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes_no-vol-outliers.csv')
final_sub_list     = op.join(quinna_dir, 'dataman/data_information', 'last_FINAL_sublist-vol-outliers-removed.csv')

# ---------------- Structures & Colors ---------------- #
structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
# You can use your custom palette (7 colors):
colormap = ['#FFD700', '#8A2BE2', '#191970', '#8B0000', '#6B8E23', '#4B0082', '#ADD8E6']

# ----------------------- Utilities ----------------------- #
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_figure_all_formats(fig: plt.Figure, out_dir: str, basename: str, dpi: int = 800) -> None:
    ensure_dir(out_dir)
    base = basename.replace(' ', '_')
    fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.svg"),              format='svg', bbox_inches='tight')

# ----------------------- Loading ------------------------ #
def load_lateralisation_dataframe() -> pd.DataFrame:
    """
    Load the lateralisation volumes CSV (prefer no-outliers file if present).
    Assumes the first column is subject ID, and the next 7 columns correspond to
    Thal, Caud, Puta, Pall, Hipp, Amyg, Accu (in that order).
    Applies an optional filter to a final subject list if available.
    """
    csv_path = lat_csv_no_outlier if op.exists(lat_csv_no_outlier) else lat_csv_all
    if not op.exists(csv_path):
        raise FileNotFoundError(f"Lateralisation volumes CSV not found:\n{csv_path}")

    df = pd.read_csv(csv_path)
    if df.shape[1] < 8:
        raise RuntimeError("Expected at least 8 columns: SubjectID + 7 structures.")

    # Standardize column names: keep first col as 'Subject' and next 7 as structures
    cols = list(df.columns)
    df = df.rename(columns={cols[1]: 'subjectID'})

    # Load final subject list
    if not op.exists(final_sub_list):
        raise FileNotFoundError(f"Final subject list not found:\n{final_sub_list}")
    sub_df = pd.read_csv(final_sub_list)

    # Expect subjectID column in sub_df
    if 'subjectID' not in sub_df.columns:
        raise RuntimeError(f"'subjectID' column not found in {final_sub_list}")

    # Filter to subjects present in sub_df
    keep_ids = set(sub_df['subjectID'].astype(str))
    df = df[df['subjectID'].astype(str).isin(keep_ids)].reset_index(drop=True)
    
    return df

# ----------------------- Plotting ----------------------- #
def plot_lateralisation_volumes(df: pd.DataFrame,
                                bins: int = 10,
                                title: str = 'Lateralisation Volume Indices (no-vol-outliers)'):
    """
    Plot histograms of lateralisation volume indices for each subcortical structure,
    annotate with Wilcoxon signed-rank p-value vs 0, and save at 800 dpi.
    """
    # Use Arial globally
    plt.rcParams['font.family'] = 'Arial'

    n_structures = len(structures)
    n_cols = 4
    n_rows = int(np.ceil(n_structures / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.8 * n_rows))
    axs = axs.flatten()

    # Styling for p-value box (as requested)
    box_props = dict(facecolor='oldlace', alpha=0.8, edgecolor='darkgoldenrod', boxstyle='round')

    for idx, structure in enumerate(structures):
        ax = axs[idx]
        vals = pd.to_numeric(df[structure], errors='coerce').dropna().values
        if vals.size == 0:
            ax.set_visible(False)
            continue

        # Histogram
        ax.hist(vals, bins=bins, color=colormap[idx], edgecolor='white')

        # Zero reference line
        ax.axvline(x=0.0, color='k', linewidth=0.8, linestyle=':')

        # Wilcoxon signed-rank test vs 0 (median)
        # scipy.stats.wilcoxon requires non-zero differences for the default zero_method
        # Add a tiny jitter to zero entries to avoid zero-sum issues (optional, conservative)
        diffs = vals - 0.0
        if np.allclose(diffs, 0.0):
            wilcox_p = 1.0  # all zeros → no evidence against 0
        else:
            # Use 'wilcox' zero_method (drop zeros)
            _, wilcox_p = stats.wilcoxon(diffs, zero_method='wilcox', correction=False, alternative='two-sided')

        # p-value text
        txt_wilx = f"Wilcoxon p = {wilcox_p:.3f}" if wilcox_p >= 0.001 else "Wilcoxon p < 0.001"
        ax.text(0.05, 0.95,
                txt_wilx,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=box_props,
                style='italic')

        # Axis labels & title (Arial, bold)
        ax.set_title(structure, fontsize=12, fontweight='bold')
        ax.set_xlabel('Lateralisation Volume', fontsize=12, fontweight='bold')
        ax.set_ylabel('# Subjects', fontsize=12, fontweight='bold')

        # Ticks styling
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_axisbelow(True)
        ax.grid(True, axis='y', alpha=0.25)

    # Remove any unused axes
    for ax in axs[len(structures):]:
        fig.delaxes(ax)

    # Also remove empty ones defensively
    for ax in axs:
        if not ax.has_data():
            fig.delaxes(ax)

    # Super-title
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])

    # Save
    out_dir = op.join(fig_output_root, 'Lateralisation_Volume_Histograms')
    save_figure_all_formats(fig, out_dir, 'lateralisation-histograms-final_subs_no-vol-outliers_nonormalcurve', dpi=800)
    plt.show()
    plt.close(fig)

# ----------------------- Run ----------------------- #
if __name__ == '__main__':
    df_lat = load_lateralisation_dataframe()
    plot_lateralisation_volumes(df_lat)



# this code needs editing. use last_FINAL_sublist-vol-outliers-removed.csv 
# to plot histograms. remove the pdf.


import os.path as op
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro

# Define paths
platform = 'mac'  # 'bluebear' or 'mac'?

if platform == 'bluebear':
    quinna_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
elif platform == 'mac':
    quinna_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'

epoched_dir = op.join(quinna_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(quinna_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')  # list of all subjects at the beginning
final_sub_list_path = op.join(info_dir, 'FINAL_sublist-vol-psd-outliers-removed.csv')  # list of subjects after removing size below 10th percentile and power nonoise_subtraction_abs_thresh

volume_sheet_dir = 'derivatives/mri/lateralized_index'
sub_list= op.join(quinna_dir, 'dataman/data_information/dblCheck_last_FINAL_sublist-vol-outliers-removed.csv'),
substr_vol_sheet_fname = op.join(sub2ctx_dir, volume_sheet_dir, 'all_subs_substr_volumes.csv')
lat_sheet_fname = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes.csv')  # this is lateralisation vol for all subjects that will be filtered to exclude vol outliers in this script
lat_sheet_fname_nooutlier = op.join(sub2ctx_dir, volume_sheet_dir, 'lateralization_volumes_no-vol-outliers.csv')  # only volume outlier
outlier_subjectID_vol_fname = op.join(info_dir, 'outlier_subjectID_vol.csv')  # this only contains vol outliers
subject_list_no_vol_outliers = op.join(info_dir, 'FINAL_sublist-vol-outliers-removed.csv')  # in case we only wanted to exclude vol outliers
fig_output_root= '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures',

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

    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def save_figure_all_formats(fig, out_dir, basename, dpi=800):
        ensure_dir(out_dir)
        base = basename.replace(' ', '_')
        fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
        fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
        fig.savefig(op.join(out_dir, f"{base}.svg"), format='svg', bbox_inches='tight')

    # Extract lateralization volumes
    lateralization_volume = substr_lat_df.iloc[:,1:8].to_numpy()

    # null hypothesis (H0) mean value
    null_hypothesis_mean = 0.0
    p_vals = []  # is used for both wilcox and ttest

    # wilcoxon p-vals
    null_hypothesis_median = 0.0

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

        # one sample wilcoxon signed rank (for non normal distributions)
        _, wilcox_p = stats.wilcoxon(valid_lateralization_volume - null_hypothesis_median,
                                        zero_method='wilcox', correction=False)
        p_vals.append(wilcox_p)

        # Plot histogram with colors
        x = lateralization_volume_hist[1]
        y = lateralization_volume_hist[0]
        ax.bar(x[:-1], y, width=np.diff(x), color=colormap[idx])

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

    basename = 'lateralisation-histograms-final_subs_no-vol-outliers_nonormalcurve'
    save_figure_all_formats(fig, fig_output_root, basename)
                            