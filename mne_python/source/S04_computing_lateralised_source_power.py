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

# Plot left hemisphere positions in blue
ax.scatter(left_positions[:, 0], left_positions[:, 1], left_positions[:, 2], 
           color='blue', label='Left Hemisphere', alpha=0.6)

# Plot right hemisphere positions in red
ax.scatter(right_positions[:, 0], right_positions[:, 1], right_positions[:, 2], 
           color='red', label='Right Hemisphere', alpha=0.6)

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

# Step 1: Create a dictionary to map left position to index for ordering, rounding to three decimal places
# because position numbers do not match to 4th decimal precision
left_pos_dict = {tuple(round(coord, 3) for coord in pos): i for i, pos in enumerate(left_positions)}

# Step 2: Prepare lists for ordered left and right positions/indices
ordered_right_positions = []
ordered_left_positions = []
ordered_right_indices = []
ordered_left_indices = []

left_index_list = []

# Step 3: Match each left position to the corresponding right position
for i, right_pos in enumerate(right_positions):
    # Find the right position that corresponds to this left position by flipping the x-coordinate
    corresponding_left_pos = (round(-(right_pos[0]), 3), round(right_pos[1], 3), round(right_pos[2], 3))
# distance between this point and all the points on the left hemisphere and choose the closest.
# sqrt sum x**2,...
    # Check if the corresponding right position exists in the right positions dictionary
    if corresponding_left_pos in left_pos_dict:
        # Get the index of this left position
        left_index = left_pos_dict[corresponding_left_pos]
        
        # Append positions and indices in the correct order
        ordered_right_positions.append(right_pos)
        ordered_left_positions.append(left_positions[left_index])
        ordered_right_indices.append(right_indices[i])
        ordered_left_indices.append(left_indices[left_index])

        # Append left_index for later use in indexing
        left_index_list.append(left_index)

# Convert ordered lists back to numpy arrays 
ordered_right_positions = np.array(ordered_right_positions)
ordered_left_positions = np.array(ordered_left_positions)
ordered_right_indices = np.array(ordered_right_indices)
ordered_left_indices = np.array(ordered_left_indices)

# Reorder time courses according to the ordered indices
ordered_right_time_courses = [right_hemisphere_time_courses[right_indices.index(i)] for i in ordered_right_indices]
ordered_left_time_courses = [left_hemisphere_time_courses[left_indices.index(i)] for i in ordered_left_indices]

# Step 4: Create tables for grid positions and indices
# For positions, separate the x, y, and z coordinates into different columns for each hemisphere
positions_table = pd.DataFrame({
    'Right Hemisphere X': [pos[0] for pos in ordered_right_positions],
    'Right Hemisphere Y': [pos[1] for pos in ordered_right_positions],
    'Right Hemisphere Z': [pos[2] for pos in ordered_right_positions],
    'Left Hemisphere X': [pos[0] for pos in ordered_left_positions],
    'Left Hemisphere Y': [pos[1] for pos in ordered_left_positions],
    'Left Hemisphere Z': [pos[2] for pos in ordered_left_positions]
})

# For indices, they should already be one-dimensional, so we can directly assign them to the DataFrame
indices_table = pd.DataFrame({
    'Right Hemisphere': ordered_right_indices,
    'Left Hemisphere': ordered_left_indices
})

# Step 5: Calculate lateralised source power
lateralised_power = []

for right_tc, left_tc  in zip(ordered_right_time_courses, ordered_left_time_courses):
    lateral_power_index = (right_tc - left_tc) / (right_tc + left_tc)
    lateralised_power.append(lateral_power_index)

lateralised_power_arr = np.squeeze(np.array(lateralised_power)) # shape into (198,) 
lateralised_power_df = pd.DataFrame(lateralised_power_arr, columns=['Lateralised Source Power Index'])

# Step 6: Plot time courses for each hemisphere

stc_lateral_power = mne.SourceEstimate(
    data=lateralised_power_arr[:, np.newaxis],  # Shape (n_sources, n_times); here, n_times=1 for static plot
    vertices=[[], ordered_right_indices],  # Left empty for left hemisphere
    tmin=0,
    tstep=1,
    subject=fs_sub
)

# Step 6: Plot the lateralized power on the brain's right hemisphere - BaseSourceSpace can only be initiated by end user (not VolSourceSpace)
stc_lateral_power.plot(
    src=forward["src"],
    subject=fs_sub,
    subjects_dir=fs_sub_dir,
    # mode='stat_map',
    hemi='rh',  # Right hemisphere
    title='Lateralized Power',
    colorbar=True
)


