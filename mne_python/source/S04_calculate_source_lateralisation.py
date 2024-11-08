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

platform = 'bluebear'  # are you running on bluebear or mac?
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

# chatGPT's solution
src = forward["src"]

# Get the indices of the sources in the grid and the positions in 3D space
grid_positions = [s['rr'] for s in src]  # 3D coordinates of the source points in the grid
grid_indices = [s['vertno'] for s in src]  # indices of the dipoles in the grid

# Step 3: Separate sources into left and right hemisphere based on x-coordinate
left_hemisphere_time_courses = []
right_hemisphere_time_courses = []
left_positions = []
right_positions = []
left_indices = []
right_indices = []

for pos_idx, pos in enumerate(grid_positions[0]):
    print(f'pos {pos_idx}')
    for ind_idx, indices in enumerate(grid_indices[0]):
        print(f'index {ind_idx}')
        if pos[0] < 0:  # Assuming x < 0 is left hemisphere
            print(f'NEGATIVE {indices} - left stc')
            left_hemisphere_time_courses.append(stc_grad.data[indices, :])
            left_positions.append(pos)
            left_indices.append(indices)
        else:  # x > 0 is right hemisphere
            print(f'POSITIVE {indices} - right stc')
            right_hemisphere_time_courses.append(stc_grad.data[indices, :])
            right_positions.append(pos)
            right_indices.append(indices)
i=0
for pos, indices in zip(grid_positions[0], grid_indices[0]):
    i+=1
    print(f'pos: {pos} {i}')
    if pos[0] < 0:  # Assuming x < 0 is left hemisphere
        print(f'NEGATIVE {indices} - left stc')
    else:  # x > 0 is right hemisphere
        print(f'POSITIVE {indices} - right stc')
 

# Step 4: Create tables for grid positions and indices
positions_table = pd.DataFrame({'Left Hemisphere': left_positions, 'Right Hemisphere': right_positions})
indices_table = pd.DataFrame({'Left Hemisphere': left_indices, 'Right Hemisphere': right_indices})

# Step 5: Calculate lateralised source power
lateralised_power = []
for left_tc, right_tc in zip(left_hemisphere_time_courses, right_hemisphere_time_courses):
    left_power = np.sum(left_tc ** 2, axis=1)  # Power = sum of squared amplitudes - why?
    right_power = np.sum(right_tc ** 2, axis=1)
    lateral_power_index = (right_power - left_power) / (right_power + left_power)
    lateralised_power.append(lateral_power_index)

lateralised_power = np.array(lateralised_power).mean(axis=1)  # Averaging across time points
lateralised_power_df = pd.DataFrame(lateralised_power, columns=['Lateralised Power Index'])

# Step 6: Plot time courses for each hemisphere
# Plot left hemisphere
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
for tc in left_hemisphere_time_courses:
    plt.plot(tc.T, color='blue', alpha=0.5)
plt.title('Left Hemisphere Source Time Courses')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')

# Plot right hemisphere
plt.subplot(2, 1, 2)
for tc in right_hemisphere_time_courses:
    plt.plot(tc.T, color='red', alpha=0.5)
plt.title('Right Hemisphere Source Time Courses')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Step 7: Visualize lateralised source power in the right hemisphere
plt.figure(figsize=(8, 6))
plt.plot(lateralised_power, 'o-', color='purple')
plt.title('Lateralised Source Power Indices in Right Hemisphere')
plt.xlabel('Grid Index')
plt.ylabel('Lateralised Power Index')
plt.grid()
plt.show()

# Display tables of positions and indices
print("Grid Positions Table:")
print(positions_table)

print("\nGrid Indices Table:")
print(indices_table)

print("\nLateralised Power Table:")
print(lateralised_power_df)


# what I dug out from mne website

label_names = mne.get_volume_labels_from_aseg(label_fpath)
label_tc = stc_grad.extract_label_time_course(label_fpath, src=forward["src"])

print('Extracting source time course from freesurfer label')
mne.extract_label_time_course(stcs, 
                              labels, 
                              forward["src"], 
                              mode='auto', 
                              allow_empty=False, 
                              return_generator=False,
                              mri_resolution=True, 
                              verbose=None)



labels = mne.read_labels_from_annot("sample", subjects_dir=subjects_dir)
label_names = [label.name for label in labels]
n_labels = len(labels)