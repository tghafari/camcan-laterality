# -*- coding: utf-8 -*-
"""
===============================================
S00. Coregistration and preparing trans file

This script coregisteres the MEG file with MRI
and generates the trans file (which is necessary
                              for BIDS conversion)

written by Tara Ghafari
adapted from Oscar Ferrante
==============================================
ToDos:
    1) 
    
Issues/ contributions to community:
    1) 
    
Questions:
    1)
Notes:
    Step 1: Reconstructing MRI using FreeSurfer
    Step 2: Reconstructing the scalp surface
    Step 3: Getting Boundary Element Model (BEM)
    Step 4: Getting BEM solution
    Step 5: Coregistration (Manual prefered)
    
    Run recon_all on freesurfer before this script.
    Steps 1, 2, and 3 are also included in the my_recon.sh bash script
"""

import numpy as np
import os.path as op
import os
import pandas as pd

import mne
import matplotlib.pyplot as plt

platform = 'mac'  # are you running on bluebear or windows or mac?

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# subject info 
meg_extension = '.fif'
meg_suffix = 'meg'
trans_suffix = 'coreg-trans'
bem_suffix = 'bem-sol' 
subjectID = '121795'  # FreeSurfer subject name - 120469  120462  120309
fs_sub = f'sub-CC{subjectID}_T1w'  # name of fs folder for each subject

# Specify specific file names
fs_sub_dir = op.join(rds_dir, f'cc700/mri/pipeline/release004/BIDS_20190411/anat')  # FreeSurfer directory (after running recon all)
deriv_folder = op.join(rds_dir, 'derivatives/meg/source/freesurfer', fs_sub[:-4])

if not os.path.exists(deriv_folder):
    os.makedirs(deriv_folder)
trans_fname = op.join(deriv_folder, f'{fs_sub[:-4]}_' + trans_suffix + meg_extension)
bem_fname = trans_fname.replace(trans_suffix, bem_suffix)  
bem_figname = bem_fname.replace(meg_extension, '.png')
coreg_figname = bem_figname.replace(bem_suffix, 'final_coreg')

check_dig_points_csv_fname = op.join(deriv_folder, 'info_dig_head_points_fit2.csv')
check_dig_dict_points_csv_fname = op.join(deriv_folder, 'dig_dict_head_points_fit2.csv')
check_trans_points_csv_fname = op.join(deriv_folder, 'trans_head_points_fit2.csv')

# for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one 
epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
epoched_fif = op.join(epoched_dir, epoched_fname)

    # try:
    #     print(f'Reading subject # {i}')
                    
info = mne.read_epochs(epoched_fif, preload=True, verbose=True).info  # one 7min50sec epochs

# # First inspect the surface reconstruction
# Brain = mne.viz.get_brain_class()

# brain = Brain(subject=fs_sub, 
#               hemi='lh', 
#               surf='pial',
#               subjects_dir=fs_sub_dir, 
#               size=(800, 600))

# brain.add_annotation('aparc.a2009s', borders=False)

# # Get Boundary Element model (BEM) solution
# """ run this section after the watershed_bem surfaces are read in freesurfer,
# (using my_recon.sh batch script)"""

# # Creat BEM model
conductivity = (.3,)  # for single layer
model = mne.make_bem_model(subject=fs_sub, 
                           subjects_dir=fs_sub_dir,
                           ico=4, 
                           conductivity=conductivity)

# BEM solution is derived from the BEM model
bem = mne.make_bem_solution(model)
mne.write_bem_solution(bem_fname, 
                       bem, 
                       overwrite=True, 
                       verbose=True)

# # Visualize the BEM
# fig = mne.viz.plot_bem(subject=fs_sub, 
#                        subjects_dir=fs_sub_dir,
#                        orientation='coronal', 
#                        brain_surfaces='white')
# fig.savefig(bem_figname)

# Coregistration
""" trans file is created here for later use in bids and then
the source-base analysis.
1) save the trans file in the MRI folder
2) rename and move the transfile to bids structure using
    01_bids_conversion... script
"""

# ## AUTOMATED COREGISTRATION ## 
plot_kwargs = dict(subject=fs_sub, 
                   subjects_dir=fs_sub_dir,
                   surfaces="head-dense", 
                   dig=True,
                   eeg=[], 
                   meg='sensors', 
                   show_axes=True,
                   coord_frame='meg')
view_kwargs = dict(azimuth=45, 
                   elevation=90, 
                   distance=.6,
                   focalpoint=(0.,0.,0.,))

# Set up the coregistration model
fiducials = "estimated"  # gets fiducials from fsaverage
coreg = mne.coreg.Coregistration(info, 
                                 subject=fs_sub, 
                                 subjects_dir=fs_sub_dir,
                                 fiducials=fiducials)
dig_info_before = coreg._info["dig"]
dig_dict_before = coreg._dig_dict  # save the list of digitalised points in coreg
trans_before = coreg.trans

fig = mne.viz.plot_alignment(info, 
                             trans=coreg.trans, 
                             **plot_kwargs)

# Initial fit with fiducials
""" firstly fit with 3 fiducial points. This allows to find a good
initial solution before optimization using head shape points"""
coreg.fit_fiducials(verbose=True)
dig_info_after_fit_fiducials = coreg._info["dig"]
dig_dict_after_fit_fiducials = coreg._dig_dict   # save the list of digitalised points in coreg
trans_after_fit_fiducials = coreg.trans

fig = mne.viz.plot_alignment(info, 
                             trans=coreg.trans, 
                             **plot_kwargs)

# Refining with ICP
""" secondly we refine the transformation using a few iterations of the
Iterative Closest Point (ICP) algorithm."""
coreg.fit_icp(n_iterations=20, 
              nasion_weight=1., 
              verbose=True)
dig_info_after_fit_icp = coreg._info["dig"]
dig_dict_after_fit_icp = coreg._dig_dict  # save the list of digitalised points in coreg
trans_after_fit_icp = coreg.trans

fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)

# Omitting bad points
""" we now remove the points that are not on the scalp"""
coreg.omit_head_shape_points(distance=5/1000)  # distance is in meters- try smaller distances
dig_info_after_omit_head_points = coreg._info["dig"]
dig_dict_after_omit_head_points = coreg._dig_dict   # save the list of digitalised points in coreg
trans_after_omit_head_points = coreg.trans
fig = mne.viz.plot_alignment(info, 
                            trans=coreg.trans, 
                            **plot_kwargs)

# Final coregistration fit
coreg.fit_icp(n_iterations=20, 
              nasion_weight=10., 
              verbose=True)
dig_info_after_final_fit_icp = coreg._info["dig"]
dig_dict_after_final_fit_icp = coreg._dig_dict  # save the list of digitalised points in coreg
trans_after_final_fit_icp = coreg.trans

coreg_fig = mne.viz.plot_alignment(info, 
                                   trans=coreg.trans, 
                                   **plot_kwargs)
# mne.viz.set_3d_view(coreg_fig, **view_kwargs)

# To save the fig above, take a screenshot of the 3D scene
# screenshot = coreg_fig.plotter.screenshot()

# The screenshot is just a NumPy array, so we can display it via imshow()
# and then save it to a file.
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(screenshot, origin='upper')
# ax.set_axis_off()  # Disable axis labels and ticks
# fig.tight_layout()
# fig.savefig(coreg_figname, dpi=150)

# Write trans if you're happy with the coregistration
# mne.write_trans(trans_fname, 
#                 coreg.trans,
#                 overwrite=True)

# # Compute distance between MRI and HSP
# dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
# print(f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
#       f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm")

### MANUAL COREGISTRATION ##
""" manually pick the fiducials and coregister MEG with MRI.
for instructions check out:https://www.youtube.com/watch?v=ALV5qqMHLlQ""" 
mne.gui.coregistration(subject=fs_sub, subjects_dir=fs_sub_dir)

# Use this for info path in the gui
info_fname = '/Volumes/quinna-camcan/derivatives/meg/sensor/epoched-7min50/sub-CC121795_ses-rest_task-rest_megtransdef_epo.fif'
trans_fname = '/Volumes/quinna-camcan/derivatives/meg/source/freesurfer/sub-CC121795/sub-CC121795_coreg-trans.fif'

# Save them manually in the gui
fiducials_fname = op.join(fs_sub_dir, fs_sub, 'bem', fs_sub + '-fiducials.fif')



"""
double_check_headmodel

the code below reads the head point from coreg in every 
step up until final fit and saves a csv file to 
compare them together.

process_digpoint_list: A helper function that takes in a 
        ist of DigPoint objects, processes each element, and extracts 
        the label and coordinates.
Regex: The code uses regular expressions to extract the 
        label (before the :) and the coordinates (between parentheses).
Convert to DataFrame: Each processed list is converted 
        into a Pandas DataFrame with two columns (label and coordinates).
Concatenate DataFrames: The DataFrames are concatenated side by side, 
        resulting in a final DataFrame with 130 rows and 8 columns.
Save as CSV: The resulting DataFrame is saved as a CSV file called 
        digpoint_data.csv.
This script creates the required table and saves it as a CSV file.
"""

import pandas as pd
import re


# Helper function to process the DigPoint strings
def process_digpoint_list(dig_list):
    processed_data = []
    
    for digpoint in dig_list:
        # Convert DigPoint object to string
        dig_str = str(digpoint)
        
        # Extract the label (everything before the ":")
        label_match = re.search(r'<DigPoint \|(.+?):', dig_str)
        label = label_match.group(1).strip() if label_match else None
        
        # Extract the coordinates (everything between parentheses and "mm")
        coord_match = re.search(r'\((.+?)\) mm', dig_str)
        coords = coord_match.group(1).strip() if coord_match else None
        
        # Add the processed data
        processed_data.append([label, coords])
    
    return processed_data

# Process each list
before_processed = process_digpoint_list(dig_info_before)
after_fit_fiducials_processed = process_digpoint_list(dig_info_after_fit_fiducials)
after_fit_icp_processed = process_digpoint_list(dig_info_after_fit_icp)
after_omit_head_point_processed = process_digpoint_list(dig_info_after_omit_head_points)
after_final_icp_processed = process_digpoint_list(dig_info_after_final_fit_icp)

# Convert the lists into DataFrames
before_df = pd.DataFrame(before_processed, columns=["Label_Before", "Coords_Before"])
after_fit_fiducials_df = pd.DataFrame(after_fit_fiducials_processed, columns=["Label_After_Fit_Fiducials", "Coords_After_Fit_Fiducials"])
after_fit_icp_df = pd.DataFrame(after_fit_icp_processed, columns=["Label_After_Fit_ICP", "Coords_After_Fit_ICP"])
after_omit_head_point_df = pd.DataFrame(after_omit_head_point_processed, columns=["Label_After_Omit_Head_Point", "Coords_After_Omit_Head_Point"])
after_final_icp_df = pd.DataFrame(after_final_icp_processed, columns=["Label_After_Final_ICP", "Coords_After_Final_ICP"])

# Concatenate the DataFrames horizontally (side by side)
final_df = pd.concat([before_df, 
                      after_fit_fiducials_df, 
                      after_fit_icp_df, 
                      after_omit_head_point_df, 
                      after_final_icp_df], axis=1)

# Save to CSV
final_df.to_csv(check_dig_points_csv_fname, index=False)

print(f"Data saved to {check_dig_points_csv_fname}")



"""
double_check_headmodel

the code below reads the head point from coreg in every 
step up until final fit and saves a csv file to 
compare them together.

Dictionaries: The dictionaries are assumed to be before, 
        after_fit_fiducials, after_fit_icp, after_omit_head_point, 
        and after_final_icp. These contain the coordinates under 
        keys like 'nasion', 'lpa', 'rpa', 'hsp', 'hpi', and 'elp'.

CSV Writing: The code uses Python's built-in csv module to write 
        the extracted data into a CSV file called check_dig_points.csv. 
        Each row in the CSV will contain:

Label: The label of the point (nasion, lpa, rpa, etc.).
Coordinates: The x, y, z coordinates.
Set: The dictionary/set it belongs to (e.g., before, 
        after_fit_fiducials, etc.).
Coordinate Handling: The coordinates are converted to a 
        2D array even if there's only one set of coordinates. 
        This ensures that the code can handle both single points 
        and multiple points (like for hsp and hpi).

Output: The CSV file will contain the concatenated tables, 
        with rows corresponding to points and columns for label, 
        coordinates, and set.

Once you run the code, it will create the check_dig_points.csv file with all the requested data.

"""

import csv
import numpy as np

# Sample dictionaries
dictionaries = {
    'dig_info_before': dig_dict_before,
    'dig_info_after_fit_fiducials': dig_dict_after_fit_fiducials,
    'dig_info_after_fit_icp': dig_dict_after_fit_icp,
    'dig_info_after_omit_head_point': dig_dict_after_omit_head_points,
    'dig_info_after_final_icp': dig_dict_after_final_fit_icp
}

# List of point labels to extract
point_labels = ['nasion', 'lpa', 'rpa', 'hsp', 'hpi', 'elp']

# Prepare CSV file to write
with open(check_dig_dict_points_csv_fname, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Label', 'X', 'Y', 'Z', 'Set'])
    
    # Iterate over the dictionaries
    for set_name, data in dictionaries.items():
        # Iterate over the labels
        for label in point_labels:
            # Check if the label exists in the dictionary
            if label in data:
                # Get the coordinates array for the label
                coordinates = np.array(data[label])
                
                # If only one set of coordinates is present, make it a 2D array
                if coordinates.ndim == 1:
                    coordinates = coordinates[np.newaxis, :]
                
                # Write each coordinate set to the CSV file
                for coord in coordinates:
                    writer.writerow([label, coord[0], coord[1], coord[2], set_name])

print(f"Data has been saved to {check_dig_dict_points_csv_fname}")


"""
double_check_trans

the code below reads the trans matrix in every 
step up until final fit and saves a csv file to 
compare them together.

extract_transform_matrix function: This function 
        takes a transformation object, converts it 
        into a string, and splits it into lines. It 
        then removes the first line (the header) and 
        processes the remaining lines as the transformation 
        matrix.

Loop through transformations: It extracts the matrices 
        from each transformation object and stores them 
        in a list.

Concatenate the matrices: All extracted matrices are 
        concatenated into a single array using np.vstack().

Save to CSV: The combined matrix is saved as a CSV file using pandas.
"""

import numpy as np
import pandas as pd

# Function to extract the matrix and ignore the header
def extract_transform_matrix(transform):
    # Convert the transformation object to string and split by lines
    lines = str(transform).split('\n')
    
    # Ignore the first line (the descriptive header)
    matrix_lines = lines[1:]
  
    # Clean up any extra characters (like '[[' or ']]') and extract valid numeric content
    cleaned_lines = []
    for line in matrix_lines:
        # Remove brackets and extra characters, keep only the numeric content
        cleaned_line = line.replace('[', '').replace(']', '').strip()
        if cleaned_line:  # Skip any empty or invalid lines
            cleaned_lines.append(cleaned_line)
    
    # Convert the cleaned lines into a NumPy array
    matrix = np.array([list(map(float, line.split())) for line in cleaned_lines])
    
    
    return matrix

transforms = {
    'trans_before': trans_before,
    'trans_after_fit_fiducials': trans_after_fit_fiducials,
    'trans_after_fit_icp': trans_after_fit_icp,
    'trans_after_omit_head_points': trans_after_omit_head_points,
    'trans_after_final_fit_icp': trans_after_final_fit_icp
}

# Initialize an empty list to collect all matrices
all_matrices = []

# Loop through the transforms, extract the matrix, and append to the list
for name, transform in transforms.items():
    matrix = extract_transform_matrix(transform)
    all_matrices.append(matrix)

# Concatenate all matrices vertically into a single array
combined_matrix = np.hstack(all_matrices)

# Save the combined matrix into a CSV file
df = pd.DataFrame(combined_matrix)
df.to_csv(check_trans_points_csv_fname, header=False, index=False)

print("Transformations saved to check_trans_points_csv_fname")
