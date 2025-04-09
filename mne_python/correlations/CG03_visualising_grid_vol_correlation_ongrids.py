"""
========================================================
CG03_visualising_grid_vol_correlation_ongrids

Script for analyzing and visualizing the correlation 
between lateralized MEG source power and subcortical volumes.

This script performs the following steps:
    1. Reads user input for frequency, sensor type, and subcortical structure.
    2. Loads Spearman correlation results and significance values.
    3. Computes right hemisphere grid positions from the fsaverage source space.
    4. Plots a 3D scatter plot of correlation values with significant points highlighted.
    5. Creates a volumetric source estimate (stc) based on correlation values.
    6. Visualizes the source estimate on MRI and optionally in 3D.

Functions:
- setup_paths: Defines file paths based on platform.
- compute_hemispheric_index: Extracts right hemisphere grid positions.
- plot_scatter: Plots 3D scatter of correlation values.
- create_volume_estimate: Creates a volumetric source estimate.
- plot_volume_estimate: Plots source estimate on MRI and in 3D.
- main: Orchestrates the entire workflow.

Author: Tara Ghafari
tara.ghafari@gmail.com
Date: 03/04/2025
============================================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from mpl_toolkits.mplot3d import Axes3D

def setup_paths(platform='mac'):
    """Set up file paths for different platforms."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
        output_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
        output_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")
    
    paths = {
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/src_lat_grid_vol_correlation_nooutliers'),
        'output_base': op.join(output_dir,'subcortical-structures/resting-state/results/CamCan/Results/src-grid-pair-freq-vol-correlation'),
        'fs_sub_dir': op.join(rds_dir,'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
    }

    return paths

def compute_hemispheric_index(src_fs):
    """
    Extract right hemisphere grid positions from fsaverage source space.
    
    Parameters:
    -----------
    src_fs : list of dicts
        Source space read using mne.read_source_spaces.
    
    Returns:
    --------
    right_positions : np.ndarray, shape (n_right, 3)
        3D positions of grid points in the right hemisphere.
    right_indices : list of int
        Indices in the full source space corresponding to right hemisphere grid points.
    """

    grid_positions = [s['rr'] for s in src_fs]  # Extract all dipole positions. this is the same as src_fs[0]['rr] as len(src_fs)=1
    grid_indices = [s['vertno'] for s in src_fs] # Get active dipole indices
    
    right_positions, right_indices = [], []
    for region_idx, indices in enumerate(grid_indices[0]):
        pos = grid_positions[0][indices]  # only select in-use positions in the source model
        if pos[0] > 0:  # x > 0 is right hemisphere
            right_positions.append(pos)
            right_indices.append(indices)
    
    return np.array(right_positions), right_indices

def plot_scatter(grid_positions, correlation_values, significant_mask, output_path, save=False):
    """
    Plot a 3D scatter of correlation values on right hemisphere grid points.
    
    Parameters:
    -----------
    grid_positions : np.ndarray, shape (n_points, 3)
        XYZ coordinates of right hemisphere grid points.
    correlation_values : np.ndarray, shape (n_points,)
        Spearman correlation values for each grid point.
    significant_mask : np.ndarray, shape (n_points,)
        Boolean array indicating significant correlations (p < 0.05).
    output_path : str
        File path to save the scatter plot.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(
        grid_positions[:, 0],
        grid_positions[:, 1],
        grid_positions[:, 2],
        c=correlation_values,
        cmap='RdBu_r',
        alpha=0.8,
        s=50
    )
    
    # Highlight significant correlations in black
    sig_positions = grid_positions[significant_mask]
    ax.scatter(sig_positions[:, 0], 
               sig_positions[:, 1], 
               sig_positions[:, 2], 
               c='k', 
               s=50, 
               label='p<0.05')
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Spearman r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Src-Vol Spearman Correlation on Right Hemisphere Grids')
    plt.legend()
    
    plt.show()
    if save:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()


def create_volume_estimate(correlation_values, significant_mask, src_fs, right_indices):
    """
    Create a volumetric source estimate using correlation values.
    
    Parameters:
    -----------
    correlation_values : np.ndarray, shape (n_right,)
        Correlation values for right hemisphere grid points.
    significant_mask : np.ndarray, shape (n_right,)
        Boolean mask for significant correlation values.
    src_fs : list of dicts
        Source space read using mne.read_source_spaces.
    right_indices : list of int
        Indices (in the full source space) corresponding to right hemisphere.
    
    Returns:
    --------
    stc : mne.VolSourceEstimate        
        The volumetric source estimate with correlation data.
    vol_mask : np.ndarray    
        Boolean mask of the same shape as stc.data indicating significant regions.
    """
    # For fsaverage volume, there's one source space element
    full_vertno = src_fs[0]['vertno']  # array of vertex numbers (may be non-contiguous)
    n_dipoles_in_src = len(full_vertno)
    n_times = 1  # Single time point
    vol_data = np.zeros((n_dipoles_in_src, n_times))
    vol_mask = np.zeros((n_dipoles_in_src, n_times), dtype=bool)
    
    # For each right hemisphere vertex, find its index within full_vertno
    """I don't know how the other method in S04 works! spent hours to figure it out!"""
    for i, vertex in enumerate(right_indices):
        # Find the position in full_vertno where the vertex is located
        pos_index = np.where(full_vertno == vertex)[0][0]
        vol_data[pos_index, 0] = correlation_values[i]
        vol_mask[pos_index, 0] = significant_mask[i]

    vertices = [full_vertno]
    stc = mne.VolSourceEstimate(
        data=vol_data, 
        vertices=vertices, 
        tmin=0, 
        tstep=1, 
        subject='fsaverage'
    )
    return stc, vol_mask

def plot_volume_estimate(stc, vol_mask, src_fs, paths, freq, sensor, structure, do_plot_3d=True, save=False, volume_masked=False):
    """
    Plot the volumetric source estimate on MRI and in 3D, highlighting significant regions.
    
    Parameters:
    -----------
    stc : mne.VolSourceEstimate
        The volumetric source estimate with correlation data.
    vol_mask : np.ndarray
        Boolean mask indicating significant regions.
    paths : dict
        Dictionary containing file paths.
    freq : str
        Frequency (e.g., '5.0').
    sensor : str
        Sensor type (e.g., 'grad' or 'mag').
    structure : str
        Subcortical structure name.
    do_plot_3d : bool
        If True, plot the 3D visualization.
    """
    initial_pos = np.array([19, -50, 29]) * 0.001

    if volume_masked:  # plot significant masks on volume?
        # Apply the mask to set non-significant values to NaN or 0 and only show significant clusters
        stc_data = stc.data.copy()  # Make a copy of the original data
        stc_data[~vol_mask] = np.mean([np.min(stc_data),np.max(stc_data)]) # Set non-significant regions to 0 (or NaN)
        
        # Create a new stc with masked data
        stc_masked = mne.VolSourceEstimate(stc_data, 
                                            stc.vertices, 
                                            stc.tmin, 
                                            stc.tstep, 
                                            subject=stc.subject)

        # Plot on MRI using the masked stc
        fig = stc_masked.plot(
            src=src_fs,  # use default source
            subject='fsaverage',
            subjects_dir=paths['fs_sub_dir'],
            mode='stat_map',
            colorbar=True,
            initial_pos=initial_pos,
            verbose=True
        )
        if save:
            mri_output = op.join(paths['output_base'], structure, f'src-substr-correlation_{sensor}_{freq}_mri_sig-only.png')
            fig.savefig(mri_output)

    # Plot on MRI without significant masks
    fig2 = stc.plot(
        src=src_fs,  # use default source
        subject='fsaverage',
        subjects_dir=paths['fs_sub_dir'],
        mode='stat_map',
        colorbar=True,
        initial_pos=initial_pos,
        verbose=True
    )
    if save:
        mri_output = op.join(paths['output_base'], structure, f'src-substr-correlation_{sensor}_{freq}_mri.png')
        fig2.savefig(mri_output)
    
    if do_plot_3d:
        kwargs = dict(
            subjects_dir=paths['fs_sub_dir'],
            hemi='both',
            size=(600, 600),
            views='sagittal',
            brain_kwargs=dict(silhouette=True),
            initial_time=0.087,
            verbose=True,
        )
        stc.plot_3d(
            src=src_fs,
            **kwargs)

def main():
    paths = setup_paths(platform='mac')
    # Prompt user for input
    freq_input = input("Enter frequency (e.g., 5.0 or Alpha): (make sure input a float number or band name)").strip()
    structure = input("Enter subcortical structure (e.g., Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    sensor = input("Enter sensor type (grad or mag): ").strip()
    do_plot_3d_input = input("Plot 3D visualization? (y/n): ").strip().lower()
    do_plot_3d = do_plot_3d_input == 'y'

    corr_file = op.join(paths['correlation_dir'], f'spearman-r_src_lat_power_vol_{sensor}_{freq_input}.csv')
    pval_file = op.join(paths['correlation_dir'], f'spearman-pval_src_lat_power_vol_{sensor}_{freq_input}.csv')
    
    if not op.exists(corr_file) or not op.exists(pval_file):
        print(f"Files not found for frequency {freq_input} and sensor {sensor}.")
        return
    
    df_corr = pd.read_csv(corr_file, index_col=None)
    df_pval = pd.read_csv(pval_file, index_col=None)
    
    if structure not in df_corr.columns:
        print(f"Structure {structure} not found in correlation file.")
        return
    
    correlation_values = df_corr[structure].values
    p_values = df_pval[structure].values
    significant_mask = p_values < 0.05
    
    # Read fsaverage source space
    fname_fsaverage_src = op.join(paths['fs_sub_dir'], "fsaverage", "bem", "fsaverage-vol-5-src.fif")
    src_fs = mne.read_source_spaces(fname_fsaverage_src)
    
    # Compute right hemisphere grid positions and indices
    grid_positions, right_indices = compute_hemispheric_index(src_fs)
    
    # Plot scatter of correlation values
    scatter_output = op.join(paths['output_base'], structure, f'src-correlation_{freq_input}.png')
    os.makedirs(op.dirname(scatter_output), exist_ok=True)
    plot_scatter(grid_positions, correlation_values, significant_mask, scatter_output)
    
    # Create a volume estimate using the correlation data
    stc, vol_mask = create_volume_estimate(correlation_values, significant_mask, src_fs, right_indices)
    
    # Plot volume estimate on MRI and optionally in 3D
    plot_volume_estimate(stc, vol_mask, src_fs, paths, freq_input, sensor, structure, do_plot_3d)

if __name__ == "__main__":
    main()
