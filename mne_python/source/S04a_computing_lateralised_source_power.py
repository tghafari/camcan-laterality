"""
===============================================
S04a. Calculate Source Lateralisation

This script computes source lateralisation indices using the formula:
    (right_stc - left_stc) / (right_stc + left_stc)

It runs for all subjects with good preprocessing and all frequency bands.
Plots and results are saved in the appropriate derivative folders.

Written by Tara Ghafari
t.ghafari@bham.ac.uk
===============================================
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Functions
# ============================================

def setup_paths(platform='mac'):
    """Sets up directory paths based on the platform."""
    if platform == 'bluebear':
        base_dir = '/rds/projects/q/quinna-camcan'
    elif platform == 'mac':
        base_dir = '/Volumes/quinna-camcan'
    else:
        raise ValueError("Unknown platform. Choose 'mac' or 'bluebear'.")
    
    deriv_folder = op.join(base_dir, 'derivatives/meg/source/freesurfer')
    sensor_folder = op.join(base_dir, 'derivatives/meg/sensor/epoched-1sec')
    info_dir = op.join(base_dir, 'dataman/data_information')
    fs_sub_dir = op.join(base_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat')
    
    return deriv_folder, sensor_folder, info_dir, fs_sub_dir

def load_subjects(info_dir):
    """Loads the list of subjects with good preprocessing."""
    good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
    good_subjects = pd.read_csv(good_sub_sheet)
    return good_subjects.index.tolist()

def construct_paths(subject, deriv_folder, sensor_folder, freq_band):
    """Constructs all required file paths for a given subject and frequency band."""
    subject_prefix = f'sub-CC{subject}'
    paths = {
        'mag_epoched': op.join(sensor_folder, f'{subject_prefix}_mag_epod-epo.fif'),
        'grad_epoched': op.join(sensor_folder, f'{subject_prefix}_grad_epod-epo.fif'),
        'mag_stc': op.join(deriv_folder, f'{subject_prefix}_mag_stc_multitaper_{freq_band}-vl.stc'),
        'grad_stc': op.join(deriv_folder, f'{subject_prefix}_grad_stc_multitaper_{freq_band}-vl.stc'),
        'fwd_vol': op.join(deriv_folder, f'{subject_prefix}_fwd-vol.fif'),
        'output_dir': op.join(deriv_folder, subject_prefix)
    }
    os.makedirs(paths['output_dir'], exist_ok=True)
    return paths

def check_existing(paths):
    """Checks whether required files exist for the given subject."""
    for key, path in paths.items():
        if 'output_dir' in key:  # Skip output directory
            continue
        if not op.exists(path):
            print(f"Missing file: {key} -> {path}")
            return False
    return True

def lateralisation_index(stc_grad, src):
    """Computes lateralisation index from the source time courses."""
    grid_positions = src[0]['rr']
    grid_indices = src[0]['vertno']
    
    left_tc, right_tc, left_pos, right_pos = [], [], [], []
    for idx, vertex in enumerate(grid_indices):
        pos = grid_positions[vertex]
        if pos[0] < 0:  # Left hemisphere
            left_tc.append(stc_grad.data[idx, :])
            left_pos.append(pos)
        elif pos[0] > 0:  # Right hemisphere
            right_tc.append(stc_grad.data[idx, :])
            right_pos.append(pos)
    
    left_tc = np.array(left_tc)
    right_tc = np.array(right_tc)
    
    lateralised_power = (right_tc - left_tc) / (right_tc + left_tc)
    return lateralised_power, np.array(right_pos), np.array(left_pos)

def plot_lateralisation(right_pos, left_pos, lateralised_power, output_path):
    """Plots lateralised power on 3D brain grid positions."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2], 
                     c=lateralised_power, cmap='coolwarm', s=50, alpha=0.6)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='Lateralised Power')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Lateralised Power in Right Hemisphere")
    plt.savefig(output_path)
    plt.close()

def process_subject(subject, freq_band, deriv_folder, sensor_folder, fs_sub_dir):
    """Processes a single subject for a specific frequency band."""
    paths = construct_paths(subject, deriv_folder, sensor_folder, freq_band)
    if not check_existing(paths):
        print(f"Skipping subject {subject}, missing files.")
        return

    # Read data
    stc_grad = mne.read_source_estimate(paths['grad_stc'])
    forward = mne.read_forward_solution(paths['fwd_vol'])
    src = forward['src']

    # Compute lateralisation
    lateralised_power, right_pos, left_pos = lateralisation_index(stc_grad, src)
    output_plot_path = op.join(paths['output_dir'], f'lateralised_power_{freq_band}.png')
    plot_lateralisation(right_pos, left_pos, np.squeeze(lateralised_power), output_plot_path)
    print(f"Processed subject {subject}, freq_band {freq_band}.")

# ============================================
# Main script
# ============================================
if __name__ == "__main__":
    deriv_folder, sensor_folder, info_dir, fs_sub_dir = setup_paths(platform='mac')
    subjects = load_subjects(info_dir)
    frequency_bands = ['alpha', 'beta', 'gamma']

    for subject in subjects:
        for freq_band in frequency_bands:
            process_subject(subject, freq_band, deriv_folder, sensor_folder, fs_sub_dir)

    print("Processing complete for all subjects and frequency bands.")
