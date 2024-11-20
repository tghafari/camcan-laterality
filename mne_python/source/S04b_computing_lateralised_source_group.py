"""
===============================================
S04b. computing lateralised source in all subjects

This script calculates the lateralized source power 
index for MEG source time courses. The lateralized 
index is calculated using the formula:


right hemisphere source power - left hemisphere source power /
right hemisphere source power + left hemisphere source power
 
The code works for volumetric source spaces and 
is optimized to handle CamCAN participants. 
The main steps are as follows:

Data Loading: Read forward models, source estimates, 
    and grid information.
Hemisphere Separation: Separate grid positions 
    and source time courses into left and right 
    hemispheres.
Matching Grids: Match right hemisphere grid 
    positions to their corresponding left hemisphere 
    counterparts.
Lateralized Power Calculation: Compute the lateralized
     source power index for each matched grid.
Output: Save the results and optionally visualize 
    the findings.

written by Tara Ghafari
(with help from chatGPT)
===============================================  
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne import read_source_estimate
from mne.vol_source_estimate import VolSourceEstimate
from tqdm import tqdm

# ============================================
# SCRIPT CONFIGURATION
# ============================================

# Input variables
subjects_dir = "/path/to/freesurfer/subjects"  # Path to FreeSurfer subjects directory
derivatives_dir = "/path/to/derivatives"      # Path to MEG derivatives directory
frequency_band = "alpha"                      # Frequency band (e.g., "alpha", "beta")
space = "volume"                              # Source space type: "volume" or "surface"

# Files and extensions
meg_extension = ".fif"
csd_extension = ".h5"
stc_extension = "-vl.stc"

# Output directory
output_dir = "/path/to/output"
os.makedirs(output_dir, exist_ok=True)

# Load subject list
subject_list = pd.read_csv("/path/to/subject_list.csv")["subject_id"].tolist()

# Initialize an empty DataFrame for lateralized source power results
lateralized_power_all = []

# ============================================
# PROCESS EACH SUBJECT
# ============================================

for subject_id in tqdm(subject_list, desc="Processing subjects"):

    try:
        # Setup paths for subject-specific files
        fs_subject = f"sub-{subject_id}_T1w"
        deriv_folder = op.join(derivatives_dir, f"sub-{subject_id}")

        # Forward model
        if space == "volume":
            forward_fname = op.join(deriv_folder, f"{subject_id}_fwd-vol{meg_extension}")
        elif space == "surface":
            forward_fname = op.join(deriv_folder, f"{subject_id}_fwd-surf{meg_extension}")
        forward = mne.read_forward_solution(forward_fname)

        # Source estimate (STC)
        stc_fname = op.join(deriv_folder, f"{subject_id}_stc_multitaper_{frequency_band}{stc_extension}")
        stc = read_source_estimate(stc_fname)

        # Extract source space information
        src = forward["src"]
        all_coordinates = np.vstack([s["rr"] for s in src])
        all_vertices = np.concatenate([s["vertno"] for s in src])

        # Separate left and right hemisphere grid positions
        left_mask = all_coordinates[:, 0] < 0
        right_mask = all_coordinates[:, 0] > 0

        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]

        left_positions = all_coordinates[left_indices]
        right_positions = all_coordinates[right_indices]

        left_time_courses = stc.data[left_indices, :]
        right_time_courses = stc.data[right_indices, :]

        # Match left and right grid positions
        matched_left_indices = []
        matched_right_indices = []
        matched_distances = []

        for right_idx, right_pos in zip(right_indices, right_positions):
            mirrored_pos = [-right_pos[0], right_pos[1], right_pos[2]]
            distances = np.linalg.norm(left_positions - mirrored_pos, axis=1)
            min_idx = np.argmin(distances)
            if distances[min_idx] < 0.01:  # Threshold for matching
                matched_left_indices.append(left_indices[min_idx])
                matched_right_indices.append(right_idx)
                matched_distances.append(distances[min_idx])

        # Calculate lateralized power index
        matched_left_time_courses = stc.data[matched_left_indices, :]
        matched_right_time_courses = stc.data[matched_right_indices, :]

        lateralized_power = (matched_right_time_courses - matched_left_time_courses) / \
                            (matched_right_time_courses + matched_left_time_courses)
        lateralized_power = np.nan_to_num(lateralized_power)  # Handle NaN values

        # Save lateralized power to a CSV for this subject
        lateralized_power_fname = op.join(output_dir, f"sub-{subject_id}_lateralized_power.csv")
        pd.DataFrame(lateralized_power).to_csv(lateralized_power_fname, index=False)

        # Save results for group-level analysis
        lateralized_power_all.append(lateralized_power)

    except Exception as e:
        print(f"Error processing subject {subject_id}: {e}")
        continue

# ============================================
# GROUP-LEVEL OUTPUT
# ============================================

# Combine lateralized power across subjects
lateralized_power_all = np.concatenate(lateralized_power_all, axis=1)
group_lateralized_power_fname = op.join(output_dir, "group_lateralized_power.csv")
pd.DataFrame(lateralized_power_all).to_csv(group_lateralized_power_fname, index=False)

print("Lateralized source power calculation complete.")
