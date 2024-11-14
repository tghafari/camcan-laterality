"""
===============================================
S04. Calculate source lateralisation

This script will read the source time
courses (calculated in S02 or S03) and
using freesurfer labels computes source
lateralised index using this formula:

right_stc - left_stc / 
right_stc + left_stc

written by Tara Ghafari
==============================================
"""


import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import mne
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import read_csd

# subject info 
subjectID = '120469'  # FreeSurfer subject name - will go in the below for loop
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

space = 'volume' # which space to use, surface or volume?
fr_band = 'alpha'  # over which frequency band you'd like to run the inverse model?

meg_extension = '.fif'
meg_suffix = 'meg'
surf_suffix = 'surf-src'
vol_suffix = 'vol-src'
fwd_vol_suffix = 'fwd-vol'
fwd_surf_suffix = 'fwd-surf'
mag_epoched_extension = 'mag_epod-epo'
grad_epoched_extension = 'grad_epod-epo'
csd_extension = '.h5'
stc_extension = '-vl.stc' # or '-vl.w'
mag_csd_extension = f'mag_csd_multitaper_{fr_band}'
grad_csd_extension = f'grad_csd_multitaper_{fr_band}'
mag_stc_extension = f'mag_stc_multitaper_{fr_band}'
grad_stc_extension = f'grad_stc_multitaper_{fr_band}'
label_fname = 'aparc+aseg.mgz'

platform = 'mac'  # are you running on bluebear or mac?
# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx'

info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])
deriv_folder_sensor = op.join(rds_dir, 'derivatives/meg/sensor/epoched-1sec')
fwd_vol_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_vol_suffix + meg_extension)
fwd_surf_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + fwd_surf_suffix + meg_extension)
label_fpath = op.join(fs_sub_dir, f'{fs_sub}/mri', label_fname)
grid_positions_csv = op.join(deriv_folder, 'grid_positions.csv')
grid_indices_csv = op.join(deriv_folder, 'grid_indices.csv')
lateralised_src_power_csv = op.join(deriv_folder, 'lateralised_src_power.csv')

# Read epoched data + baseline correction + define frequency bands
# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
    # try:
    #     print(f'Reading subject # {i}')

mag_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + mag_epoched_extension + meg_extension)
grad_epoched_fname = op.join(deriv_folder_sensor, f'{fs_sub[:-4]}_' + grad_epoched_extension + meg_extension)
mag_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_csd_extension + csd_extension)
grad_csd_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_csd_extension + csd_extension)
mag_stc_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + mag_stc_extension + stc_extension)
grad_stc_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + grad_stc_extension + stc_extension)

print('Read source time courses')
stc_mag = mne.read_source_estimate(mag_stc_fname)
stc_grad = mne.read_source_estimate(grad_stc_fname)

print('Reading forward model')
if space == 'surface':
    forward = mne.read_forward_solution(fwd_surf_fname)
elif space == 'volume':
    forward = mne.read_forward_solution(fwd_vol_fname)

src = forward["src"]

# Get the indices of the sources in the grid and the positions in 3D space
grid_positions = [s['rr'] for s in src]  # 3D coordinates of the source points in the grid => list of one array of (37696, 3) 
grid_indices = [s['vertno'] for s in src]  # indices of the dipoles in the grid => list of one array of (12183,)
                                           # this shows the active grides that were used for forward model, 
                                           # which is not necessarily equal to all the grids whose positions 
                                           # are in grid_positions 

# Step 3: Separate sources into left and right hemisphere based on x-coordinate
left_hemisphere_time_courses = []
right_hemisphere_time_courses = []
left_positions = []
right_positions = []
left_indices = []
right_indices = []

for region_idx, indices in enumerate(grid_indices[0]):
    print(f'{region_idx}')
    pos = grid_positions[0][indices]  # only select in-use positions in the source model
    print(f'{pos}')
    if pos[0] < 0:  # x < 0 is left hemisphere
        left_hemisphere_time_courses.append(stc_grad.data[region_idx, :])
        left_positions.append(pos)
        left_indices.append(indices)
    elif pos[0] > 0:  # x > 0 is right hemisphere
        right_hemisphere_time_courses.append(stc_grad.data[region_idx, :])
        right_positions.append(pos)
        right_indices.append(indices)

# Plot the in-use grid positions
# Convert lists to numpy arrays for easy manipulation
right_positions = np.array(right_positions)
left_positions = np.array(left_positions)

# Plot left and right positions in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot right hemisphere positions in red
ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
           color='red', label='Right Hemisphere', alpha=0.6)

# Plot left hemisphere positions in blue
ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
           color='blue', label='Left Hemisphere', alpha=0.6)

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.title("3D Positions of Left and Right Hemisphere Grid Positions")
plt.legend()

plt.show()

# Find the right order of grid positions
"""
    To match the positions in left_positions and right_positions by aligning the 
    x, y, and z coordinates such that each 
    (x,y,z) position in right_positions corresponds to a 
    (-x,y,z) position in left_positions, you can use a 
    sorting approach. 
    Once we find the correct order, we'll 
    reorder left_positions and right_positions along with 
    their respective left_indices and right_indices.
"""

# Step 1: Create a dictionary to easily access left positions by their coordinates
left_pos_dict = {tuple(pos): idx for idx, pos in enumerate(left_positions)}

min_distance_accepted = 0.01  # the minimum euclidian distance that's accepted from corresponding points

# Prepare lists for ordered positions and indices
ordered_right_positions = []
ordered_left_positions = []
ordered_right_indices = []
ordered_left_indices = []
distances = []

# List to keep track of which indices have already been selected in left_positions
used_left_indices = set()

# Step 2: Match each right position to the closest corresponding left position
for i, right_pos in enumerate(right_positions):
    # Flip the x-coordinate to find the corresponding left position
    corresponding_left_pos = (-right_pos[0], right_pos[1], right_pos[2])

    # Calculate the distance between corresponding_left_pos and all left positions
    min_distance = float('inf')
    closest_left_index = None

    # Iterate over all left positions to find the closest match
    for j, left_pos in enumerate(left_positions):
        # if j in used_left_indices:
        #     continue  # Skip if the left position has already been assigned - not necessary

        # Calculate Euclidean distance
        distance = np.sqrt((left_pos[0] - corresponding_left_pos[0])**2 
                           + (left_pos[1] - corresponding_left_pos[1])**2
                           + (left_pos[2] - corresponding_left_pos[2])**2)

        # Check if this is the closest match so far
        if distance < min_distance:
            min_distance = distance
            closest_left_index = j

    # If a matching left position is found, add it to the ordered lists
    if closest_left_index is not None and min_distance <= min_distance_accepted:
        # Append positions, indices, and distance in the correct order
        ordered_right_positions.append(right_pos)
        ordered_left_positions.append(left_positions[closest_left_index])
        ordered_right_indices.append(right_indices[i])
        ordered_left_indices.append(left_indices[closest_left_index])
        distances.append(min_distance)

        # Mark this left position as used
        used_left_indices.add(closest_left_index)

    # Break if there are no more available left or right positions to assign
    if len(used_left_indices) >= len(left_positions) or len(used_left_indices) >= len(right_positions):
        break

# Convert ordered lists back to numpy arrays for easier manipulation
ordered_right_positions = np.array(ordered_right_positions)
ordered_left_positions = np.array(ordered_left_positions)
ordered_right_indices = np.array(ordered_right_indices)
ordered_left_indices = np.array(ordered_left_indices)

# Step 3: Reorder time courses based on the ordered indices
ordered_right_time_courses = [right_hemisphere_time_courses[right_indices.index(idx)] for idx in ordered_right_indices]
ordered_left_time_courses = [left_hemisphere_time_courses[left_indices.index(idx)] for idx in ordered_left_indices]

# Step 4: Create tables for grid positions, indices, and distances
# Create a DataFrame for positions, including the distance between each corresponding pair
positions_table = pd.DataFrame({
    'Right Hemisphere X': [pos[0] for pos in ordered_right_positions],
    'Right Hemisphere Y': [pos[1] for pos in ordered_right_positions],
    'Right Hemisphere Z': [pos[2] for pos in ordered_right_positions],
    'Left Hemisphere X': [pos[0] for pos in ordered_left_positions],
    'Left Hemisphere Y': [pos[1] for pos in ordered_left_positions],
    'Left Hemisphere Z': [pos[2] for pos in ordered_left_positions],
    'Distance': distances
})

# Create a DataFrame for indices
indices_table = pd.DataFrame({
    'Right Hemisphere Index': ordered_right_indices,
    'Left Hemisphere Index': ordered_left_indices
})

# Step 5: Calculate lateralised source power
lateralised_power = []

for right_tc, left_tc  in zip(ordered_right_time_courses, ordered_left_time_courses):
    lateral_power_index = (right_tc - left_tc) / (right_tc + left_tc)
    lateralised_power.append(lateral_power_index)

lateralised_power_arr = np.squeeze(np.array(lateralised_power)) 
lateralised_power_df = pd.DataFrame(lateralised_power_arr, columns=['Lateralised Source Power Index'])

# Save all dataframes to disk
positions_table.to_csv(grid_positions_csv)
indices_table.to_csv(grid_indices_csv)
lateralised_power_df.to_csv(lateralised_src_power_csv)

# Plot findings in grid positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of grid positions with color representing lateralised power
sc = ax.scatter(
    ordered_right_positions[:, 0],  # x-coordinates
    ordered_right_positions[:, 1],  # y-coordinates
    ordered_right_positions[:, 2],  # z-coordinates
    c=lateralised_power_arr,        # color by lateralised power
    cmap='coolwarm',                # color map
    label='Right Hemisphere', 
    alpha=0.6,
    s=50                            # size of points
)

# Add a colorbar to show the range of lateralised power values
cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Lateralised Power')

# Set plot labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Lateralised Power on Right Hemisphere Grid Points')
plt.legend()
plt.show()

# Step 6: Plot time courses for each hemisphere
stc_lateral_power = mne.VolSourceEstimate(
    data=lateralised_power_arr[:, np.newaxis],  # Shape (n_sources, n_times); here, n_times=1 for static plot
    vertices=[ordered_right_indices],  # Left empty for left hemisphere
    tmin=0,
    tstep=1,
    subject=fs_sub
)

# Step 6: Plot the lateralized power on the brain's right hemisphere - BaseSourceSpace can only be initiated by end user (not VolSourceSpace)
stc_lateral_power.plot(
    src=forward["src"],
    subject=fs_sub,
    subjects_dir=fs_sub_dir,
    mode='stat_map',
    # hemi='rh',  # Right hemisphere
    colorbar=True
)


