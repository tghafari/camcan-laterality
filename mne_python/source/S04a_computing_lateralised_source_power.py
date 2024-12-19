"""
===============================================
S04a. Calculate Source Lateralisation

This script computes source lateralisation indices 
using the formula:
    (right_stc - left_stc) / (right_stc + left_stc)

It runs for all subjects with good preprocessing 
and all frequency bands.

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
from mne.datasets import fetch_fsaverage
from mpl_toolkits.mplot3d import Axes3D

# ============================================
# Functions
# ============================================

def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'rds_dir': rds_dir,
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'meg_source_dir': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer'),
        'meg_sensor_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-1to8sec'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
    }
    return paths

def load_subjects(good_sub_sheet):
    """Loads the list of subjects with good preprocessing."""
    good_subjects = pd.read_csv(good_sub_sheet)
    return good_subjects.index.tolist()

def construct_paths(subjectID, paths, sensortype, space, fr_band):
    """
    Construct required file paths for a given subject and frequency band.

    Parameters:
    - subjectID (str): Subject ID.
    - paths (dict): Dictionary of data paths.
    - sensortype (str): 'grad' or 'mag'.
    - space (str): 'vol' or 'surf'.
    - fr_band (str): Frequency band.

    Returns:
    - dict: File paths for the subject and frequency band.
    """

    fs_sub = f'sub-CC{subjectID}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    file_paths = {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        f'{sensortype}_stc': op.join(deriv_folder, f'{fs_sub[:-4]}_{sensortype}_stc_multitaper_{fr_band}-vl.stc'),
        f'fsmorph_{sensortype}_stc_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fsmorph_{sensortype}_stc_{fr_band}-vl.stc'),
        f'fwd_{space}': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-{space}.fif'),
        'grid_positions_csv': op.join(deriv_folder, 'grid_positions.csv'),
        'grid_indices_csv': op.join(deriv_folder, 'grid_indices.csv'),
        'lateralised_src_power_csv': op.join(deriv_folder, 'lateralised_src_power.csv'),
        'lateralised_grid_figname': op.join(deriv_folder, 'lateralised_grid.png'),
        'stc_VolEst_lateral_power_figname': op.join(deriv_folder, f'stc_VolEst_lateral_power_{sensortype}_{fr_band}.png'),
        'stc_fsmorphd_lateral_power_figname': op.join(deriv_folder, f'stc_morphd_lateral_power_{sensortype}_{fr_band}.png'),
    }
    return file_paths

def check_existing(file_paths):
    """Checks whether output files already exist for the given subject."""
    if op.exists(file_paths['grid_positions_csv']) or \
    op.exists(file_paths['grid_indices_csv']) or \
    op.exists(file_paths['lateralised_src_power_csv']):
        print(f"source lateralisation results already exist for {file_paths['fs_sub']}. Skipping...")
        return True
    return False

def morph_subject_to_fsaverage(file_paths, src, sensortype):
    """
    Morph subject data to fsaverage space for more 
    reliable comparisons later.

    Parameters:
    - file_paths (dict): Dictionary of file paths.
    - src: Source space object.
    - sensortype (str): 'grad' or 'mag'.

    Returns:
    - mne.SourceEstimate: Morphed source estimate.
    """

    fetch_fsaverage(file_paths["fs_sub_dir"])  # ensure fsaverage src exists
    fname_fsaverage_src = op.join(file_paths["fs_sub_dir"], "fsaverage", "bem", "fsaverage-vol-5-src.fif")

    src_fs = mne.read_source_spaces(fname_fsaverage_src)
    morph = mne.compute_source_morph(
        src,
        subject_from=file_paths["fs_sub"],
        src_to=src_fs,
        subjects_dir=file_paths["fs_sub_dir"],
        niter_sdr=[5, 5, 2],
        niter_affine=[5, 5, 2],
        zooms='auto',  # just for speed
        verbose=True,
    )
    stc_fs = morph.apply(file_paths[f'{sensortype}_stc'])
    stc_fs.save(file_paths[f'fsmorph_{sensortype}_stc_fname'], overwrite=True)

    lims = [0.3, 0.45, 0.6]
    stc_fs.plot(
        src=src_fs,
        mode="stat_map",
        initial_time=0.085,
        subjects_dir=file_paths["fs_sub_dir"],
        clim=dict(kind="value", pos_lims=lims),
        verbose=True,
    ).savefig(file_paths['stc_fsmorphd_lateral_power_figname'])

    return stc_fs

def compute_hemispheric_index(stc_fs, src):
    """
    Compute the hemispheric lateralisation index from source estimates.

    Parameters:
    -----------
    stc_fs : mne.SourceMorph
        Morphed source estimate from this subject to fsaverage.
        this is calculated based on sensortype
    src : instance of mne.VolSourceEstimate
        Original source space from this subject.
        note that this is NOT fs_src.

    Returns:
    --------
    tuple
        Data for left and right hemisphere time courses, positions, and indices.
    """
    grid_positions = [s['rr'] for s in src]
    grid_indices = [s['vertno'] for s in src]

    # Separate sources into left and right hemisphere based on x-coordinate
    right_hemisphere_time_courses, left_hemisphere_time_courses = [], []
    right_positions, left_positions = [], []
    right_indices, left_indices = [], []
    right_reg_indices, left_reg_indices = [], []
    

    for region_idx, indices in enumerate(grid_indices[0]):
        pos = grid_positions[0][indices]  # only select in-use positions in the source model
        if pos[0] < 0:  # x < 0 is left hemisphere
            left_hemisphere_time_courses.append(stc_fs.data[region_idx, :])
            left_positions.append(pos)
            left_indices.append(indices)
            left_reg_indices.append(region_idx)
        elif pos[0] > 0:  # x > 0 is right hemisphere
            right_hemisphere_time_courses.append(stc_fs.data[region_idx, :])
            right_positions.append(pos)
            right_indices.append(indices)
            right_reg_indices.append(region_idx)

        # Convert lists to numpy arrays for easy manipulation
        right_positions = np.array(right_positions)
        left_positions = np.array(left_positions)

    return (right_hemisphere_time_courses, left_hemisphere_time_courses,
            right_positions, left_positions,
            right_indices, left_indices,
            right_reg_indices, left_reg_indices)


def order_grid_positions(right_positions, left_positions, 
                         right_indices, left_indices, 
                         right_reg_indices, left_reg_indices,
                         right_hemisphere_time_courses, left_hemisphere_time_courses,
                         file_paths):
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
    # Minimum euclidian distance accepted from corresponding points
    min_distance_accepted = 0.01 

    # Prepare lists for ordered positions and indices
    ordered_right_positions = []
    ordered_left_positions = []
    ordered_right_indices = []
    ordered_left_indices = []
    ordered_right_region_indices = []  # for later creating the volume source estimate
    ordered_left_region_indices = []
    distances = []

    # List to keep track of which indices have already been selected in left_positions
    used_left_indices = set()

    # Step 2: Match each right position to the closest corresponding left position
    for i, right_pos in enumerate(right_positions):
        # Flip the x-coordinate to find the corresponding left position
        print(f'Reading {i}th grid position')
        corresponding_left_pos = (-right_pos[0], right_pos[1], right_pos[2])

        # Calculate the distance between corresponding_left_pos and all left positions
        min_distance = float('inf')
        closest_left_index = None

        # Iterate over all left positions to find the closest match
        for j, left_pos in enumerate(left_positions):

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
            ordered_right_region_indices.append(right_reg_indices[i])
            ordered_left_indices.append(left_indices[closest_left_index])
            ordered_left_region_indices.append(left_reg_indices[closest_left_index])
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
    ordered_right_region_indices = np.array(ordered_right_region_indices)
    ordered_left_region_indices = np.array(ordered_left_region_indices)

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

    positions_table.to_csv(file_paths["grid_positions_csv"])
    indices_table.to_csv(file_paths["grid_indices_csv"])

    return (ordered_right_positions, ordered_left_positions,
            ordered_right_indices, ordered_left_indices,
            ordered_right_region_indices, ordered_left_region_indices,
            ordered_right_time_courses, ordered_left_time_courses)

def calculate_grid_lateralisation(ordered_right_time_courses, ordered_left_time_courses, file_paths):
    # Calculate lateralised source power
    lateralised_power = []

    for right_tc, left_tc  in zip(ordered_right_time_courses, ordered_left_time_courses):
        lateral_power_index = (right_tc - left_tc) / (right_tc + left_tc)
        lateralised_power.append(lateral_power_index)

    lateralised_power_arr = np.squeeze(np.array(lateralised_power)) 
    lateralised_power_df = pd.DataFrame(lateralised_power_arr, columns=['Lateralised Source Power Index'])

    lateralised_power_df.to_csv(file_paths["lateralised_src_power_csv"])

    return lateralised_power_arr

def plot_lateralisation(ordered_right_positions, lateralised_power_arr, 
                        ordered_right_region_indices,
                        forward, file_paths):
    """ 
    Plot findings in grid positions and 
    on a VolumeEstimate.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of grid positions with color representing lateralised power
    sc = ax.scatter(
        ordered_right_positions[:, 0],  # x-coordinates
        ordered_right_positions[:, 1],  # y-coordinates
        ordered_right_positions[:, 2],  # z-coordinates
        c=lateralised_power_arr,        # color by lateralised power
        cmap='coolwarm',                # color map
        label='Source lateralised power', 
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
    plt.savefig(file_paths["lateralised_grid_figname"])
    plt.close()

    # Create a volume estimate 
    """Create an mne.VolSourceEstimate object for lateralised_power_arr, 
    ensuring the data structure is correctly formatted"""

    # Step 1: Prepare the data
    # Initialize an empty array with zeros for all dipoles in the source space
    n_dipoles_in_src = sum([len(s['vertno']) for s in forward['src']])  # Total in-use dipoles
    n_times = 1  # Single time point for static data
    lateralised_power_full = np.zeros((n_dipoles_in_src, n_times))

    # Fill the right side of the vol estimate with lateralised powers (left side is all zero)
    for i, index in enumerate(ordered_right_region_indices):
        lateralised_power_full[index, 0] = lateralised_power_arr[i]

    # Step 2: Create the VolSourceEstimate object
    vertices = [np.array(forward['src'][0]['vertno'])]

    stc_lateral_power = mne.VolSourceEstimate(
        data=lateralised_power_full,
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject=file_paths["fs_sub"]
    )

    # Step 3: Plot the lateralized power on the brain
    stc_lateral_power.plot(
        src=forward["src"],
        subject=file_paths["fs_sub"],
        subjects_dir=file_paths["fs_sub_dir"],
        mode='stat_map',
        colorbar=True,
        verbose=True
    ).savefig(file_paths['stc_VolEst_lateral_power_figname'])

def process_subject(subjectID, paths, sensortype, space, fr_band):
    """Processes a single subject for a specific frequency band.
    sensortyep= 'grad' or 'mag' 
    space= 'vol or 'surf' """
    file_paths = construct_paths(subjectID, paths, sensortype, space, fr_band)

    if check_existing(file_paths):
        return

    # Read data - add mags
    forward = mne.read_forward_solution(file_paths[f'fwd_{space}'])
    src = forward['src']

    # Compute lateralisation
    stc_fs = morph_subject_to_fsaverage(file_paths, src, sensortype)
    (left_hemisphere_time_courses, right_hemisphere_time_courses,
            left_positions, right_positions,
            left_indices, right_indices,
            left_reg_indices, right_reg_indices) = compute_hemispheric_index(stc_fs, src)
    (ordered_right_positions, _,
    _, _, ordered_right_region_indices, _,
    ordered_right_time_courses, ordered_left_time_courses) = order_grid_positions(right_positions, 
                                                        left_positions, 
                                                        right_indices, left_indices, 
                                                        right_reg_indices, left_reg_indices,
                                                        right_hemisphere_time_courses, left_hemisphere_time_courses,
                                                        file_paths)
    lateralised_power_arr = calculate_grid_lateralisation(ordered_right_time_courses, 
                                                          ordered_left_time_courses, 
                                                          file_paths)
    plot_lateralisation(ordered_right_positions, lateralised_power_arr, 
                        ordered_right_region_indices,
                        forward, file_paths)
    print(f"Processed subject {subjectID}, freq_band {fr_band}.")


# ============================================
# Main script
# ============================================

def main():

    platform = 'mac'  # Set platform: 'mac' or 'bluebear'
    sensortypes = ['grad', 'mag']
    fr_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']  # Frequency bands to process
    space = 'vol'  # Space type: 'surface' or 'volume'

    paths = setup_paths(platform)
    good_subjects = load_subjects(paths['good_sub_sheet'])

    for subjectID in good_subjects.index:
        for sensortype in sensortypes:
            for fr_band in fr_bands:
                process_subject(subjectID, paths, sensortype, space, fr_band)

    print("Processing complete for all subjects and frequency bands.")

if __name__ == "__main__":
    main()
