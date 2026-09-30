# -*- coding: utf-8 -*-
"""
==============================================
figure1_substr_histograms

This script plots lateralisation volume indices 
across participants for 7 subcortical structures.
Saves figure as TIFF/PNG/SVG (dpi=800). 
Each subplot shows a Wilcoxon p-value vs 0.
The output figure is to be used as Figure 1 in
the CamCAN paper.

Written by Tara Ghafari
tara.ghafari@gmail.com
19/08/2025
==============================================
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
    fig_output_root = '/Users/taraghafari/Desktop/Desktop - Tara’s MacBook Pro/BEAR_outage/CamCAN/manuscript/Figures'
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
def plot_lateralisation_volumes(
    df: pd.DataFrame,
    bins: int = 10,
    title: str | None = None,
    fig_output_root: str = fig_output_root,
    font_family: str = 'Arial'
):
    """
    Plot lateralisation volume histograms for the seven subcortical structures.

    Caudate, putamen, pallidum, and hippocampus are saved as separate figures.
    Thalamus, amygdala, and nucleus accumbens are saved together in one figure.

    Every histogram uses an x-axis symmetric around zero. Each panel also
    reports a two-sided one-sample Wilcoxon signed-rank p-value against zero.

    Figures are saved as TIFF, PNG, and SVG at 800 dpi.
    """
    plt.rcParams['font.family'] = font_family

    box_props = dict(
        facecolor='oldlace',
        alpha=0.8,
        edgecolor='darkgoldenrod',
        boxstyle='round'
    )

    individual_structures = ['Caud', 'Puta', 'Pall', 'Hipp']
    combined_structures = ['Thal', 'Amyg', 'Accu']
    null_hypothesis_median = 0.0

    structure_to_idx = {
        structure: structures.index(structure)
        for structure in structures
    }

    def get_symmetric_xlim(values):
        """Return symmetric x-axis limits around zero with a 5% margin."""
        max_abs = np.nanmax(np.abs(values))
        if max_abs == 0:
            max_abs = 1.0
        max_abs *= 1.05
        return -max_abs, max_abs

    def get_wilcoxon_p(values):
        """Compute a two-sided one-sample Wilcoxon test against zero."""
        diffs = values - null_hypothesis_median

        if np.allclose(diffs, 0.0):
            return 1.0

        _, p_value = stats.wilcoxon(
            diffs,
            zero_method='wilcox',
            correction=False,
            alternative='two-sided'
        )
        return p_value

    def format_p_value(p_value):
        """Format Wilcoxon p-values for figure annotation."""
        return (
            f"Wilcoxon p = {p_value:.3f}"
            if p_value >= 0.001
            else "Wilcoxon p < 0.001"
        )

    def style_axis(ax, structure, values, show_ylabel=True):
        """Apply consistent publication-style formatting."""
        x_min, x_max = get_symmetric_xlim(values)
        ax.set_xlim(x_min, x_max)

        ax.axvline(
            0.0,
            color='dimgray',
            linewidth=0.8,
            linestyle='-'
        )

        ax.set_title(
            structure,
            fontsize=16,
            fontweight='bold'
        )

        ax.set_xlabel(
            'Lateralisation Volume',
            fontsize=14,
            fontweight='bold',
            labelpad=5
        )

        if show_ylabel:
            ax.set_ylabel(
                '# Subjects',
                fontsize=14,
                fontweight='bold',
                labelpad=5
            )

        ax.tick_params(
            axis='both',
            which='both',
            length=0,
            labelsize=12
        )

        ax.set_axisbelow(True)

        ax.grid(
            True,
            axis='y',
            alpha=0.25
        )

    def annotate_p_value(ax, p_value):
        """Add the Wilcoxon p-value annotation box."""
        ax.text(
            0.05,
            0.95,
            format_p_value(p_value),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=box_props,
            style='italic'
        )

    def plot_single_structure(structure):
        """Create and save a standalone histogram."""
        idx = structure_to_idx[structure]

        values = pd.to_numeric(
            df[structure],
            errors='coerce'
        ).dropna().values

        if values.size == 0:
            print(f"[WARNING] No valid data available for {structure}.")
            return

        wilcox_p = get_wilcoxon_p(values)

        fig, ax = plt.subplots(figsize=(5, 4.5))

        ax.hist(
            values,
            bins=bins,
            color=colormap[idx],
            edgecolor='white'
        )

        style_axis(ax, structure, values)
        annotate_p_value(ax, wilcox_p)

        fig.tight_layout()

        out_dir = op.join(
            fig_output_root,
            'Lateralisation_Volume_Histograms'
        )
        ensure_dir(out_dir)

        save_figure_all_formats(
            fig,
            out_dir,
            f'{structure}_lateralisation_volume_histogram',
            dpi=800
        )

        plt.show()
        plt.close(fig)

        print(
            f"[DONE] {structure}: N = {len(values)}, "
            f"Wilcoxon p = {wilcox_p:.4g}"
        )

    def plot_combined_structures():
        """Create and save the combined Thal/Amyg/Accu figure."""
        fig, axs = plt.subplots(
            1,
            len(combined_structures),
            figsize=(15, 4.5)
        )

        for plot_idx, structure in enumerate(combined_structures):
            ax = axs[plot_idx]
            idx = structure_to_idx[structure]

            values = pd.to_numeric(
                df[structure],
                errors='coerce'
            ).dropna().values

            if values.size == 0:
                ax.set_visible(False)
                continue

            wilcox_p = get_wilcoxon_p(values)

            ax.hist(
                values,
                bins=bins,
                color=colormap[idx],
                edgecolor='white'
            )

            style_axis(
                ax,
                structure,
                values,
                show_ylabel=(plot_idx == 0)
            )
            annotate_p_value(ax, wilcox_p)

        fig.tight_layout()

        out_dir = op.join(
            fig_output_root,
            'Lateralisation_Volume_Histograms'
        )
        ensure_dir(out_dir)

        save_figure_all_formats(
            fig,
            out_dir,
            'Thal_Amyg_Accu_lateralisation_volume_histograms',
            dpi=800
        )

        plt.show()
        plt.close(fig)

        print("[DONE] Combined figure: Thal, Amyg, Accu")

    # Four individual figures.
    for structure in individual_structures:
        plot_single_structure(structure)

    # One combined figure.
    plot_combined_structures()


# ----------------------- Run ----------------------- #
if __name__ == '__main__':
    df_lat = load_lateralisation_dataframe()
    plot_lateralisation_volumes(df_lat)
