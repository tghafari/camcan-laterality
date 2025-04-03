

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.datasets import fetch_fsaverage

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
    """
    grid_positions = [s['rr'] for s in src_fs][0]  # Extract all dipole positions
    grid_indices = [s['vertno'] for s in src_fs][0]  # Get active dipole indices
    
    right_positions, right_indices = [], []
    for idx in grid_indices:
        pos = grid_positions[idx]
        if pos[0] > 0:  # x > 0 is right hemisphere
            right_positions.append(pos)
            right_indices.append(idx)
    
    return np.array(right_positions), right_indices

def plot_correlation(grid_positions, correlation_values, significant_mask, output_path):
    """Plots correlation values on the right hemisphere grid points."""
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
    ax.scatter(sig_positions[:, 0], sig_positions[:, 1], sig_positions[:, 2], c='k', s=50, label='p<0.05')
    
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Spearman r')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Spearman Correlation on Right Hemisphere Grid')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def main():
    paths = setup_paths()

    fetch_fsaverage(paths["fs_sub_dir"])  # ensure fsaverage src exists
    fname_fsaverage_src = op.join(paths["fs_sub_dir"], "fsaverage", "bem", "fsaverage-vol-5-src.fif")
    src_fs = mne.read_source_spaces(fname_fsaverage_src)

    grid_positions, grid_indices = compute_hemispheric_index(src_fs)
    
    structures = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']
    frequencies = [f'{freq:.1f}' for freq in np.arange(1.5, 60, 0.5)]
    
    for structure in structures:
        for freq in frequencies:
            corr_file = os.path.join(paths['correlation_dir'], f'spearman_src_lat_power_vol_grad_{freq}.csv')
            pval_file = os.path.join(paths['correlation_dir'], f'spearman_pvals_{freq}.csv')
            if not os.path.exists(corr_file) or not os.path.exists(pval_file):
                continue
            
            df_corr = pd.read_csv(corr_file, index_col=0)
            df_pval = pd.read_csv(pval_file, index_col=0)
            
            if structure not in df_corr.columns:
                continue
            
            correlation_values = df_corr[structure].values
            significant_mask = df_pval[structure].values < 0.05  # Identify significant values
            output_path = os.path.join(paths['output_base'], structure, f'src-correlation_{freq}.png')
            
            plot_correlation(grid_positions, correlation_values, significant_mask, output_path)

if __name__ == "__main__":
    main()
