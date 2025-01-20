"""
====================================
CG01_gridwise_correlating_lateralised_spectra_substr:

this code will:
    1. read lateralised source power perHz for
    each subject
    2. puts lateralised source power of all
    subjects and all grids perHz in one 
    6287 (#grid) by 620 (#subjects) csv file
    3. calculates the correlation across all 
    subjects between lateralised volume of
    all subcortical regions + age and lateralised
    power of grids
    4. saves the those correlation values perHz
    in a folder. This results in number of freqs
    folders.

written by Tara Ghafaru
tara.ghafari@gmail.com
"""

import os.path as op
import pandas as pd
import numpy as np

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
        'epoched_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-7min50'),
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'good_sub_sheet': op.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv'),
        'meg_sensor_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-2sec'),
        'meg_source_dir': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer')
    }
    return paths

def load_subjects(good_sub_sheet):
    """Load subject IDs from a CSV file."""
    good_subject_pd = pd.read_csv(good_sub_sheet)
    good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # Set subject IDs as the index
    return good_subject_pd


platform = 'bluebear'  # Set platform: 'mac' or 'bluebear'
sensortypes = ['grad', 'mag']
freqs = np.arange(10, 11, 0.5)  # range of frequencies for dics
space = 'vol'  # Space type: 'surface' or 'volume'
csd_method = 'multitaper'  # or 'fourier'
paths = setup_paths(platform)
good_subjects = load_subjects(paths['good_sub_sheet'])


# Initialize the list to hold data for all subjects
all_subs_lat_grid_perHz = []

def create_all_subs_lateralised_grid(good_subjects, paths, freq, all_subs_lat_grid_perHz):
    """
    Reads, transposes, and appends lateralised data for all subjects,
    ensuring shape consistency, and saves the final DataFrame to CSV.
    
    Args:
    - good_subjects (DataFrame): DataFrame containing valid subject IDs.
    - paths (dict): Dictionary containing the directory paths.
    - freq (str): Frequency information for the filename.
    - all_subs_lat_grid_perHz (list): List to collect data for all subjects.
    
    Returns:
    - DataFrame: Combined DataFrame with all subjects' data.
    """

    # Track shapes to ensure consistency
    shapes = []
    freq = float(freq)

    for subjectID in good_subjects.index[1:3]:
        # Construct the file path for the current subject
        lat_source_perHz_fname = op.join(
            paths['meg_source_dir'], 
            f'sub-CC{subjectID}', 
            'lat_soure_perHz', 
            f'lateralised_src_power_grad_multitaper_{freq}'
        )

        # Read the subject's DataFrame
        sub_lat_grid_pd = pd.read_csv(lat_source_perHz_fname)

        # Record the shape of the DataFrame
        shapes.append(sub_lat_grid_pd.shape)

        # Transpose the DataFrame
        sub_lat_grid_pd_transposed = sub_lat_grid_pd.T
        
        # Append the transposed DataFrame to the list
        all_subs_lat_grid_perHz.append(sub_lat_grid_pd_transposed)






# Navigate to lat_source_perHz for each subject
all_subs_lat_grid_perHz = []
def create_all_subs_lateralised_grid(good_subjects, paths, freq, all_subs_lat_grid_perHz):

    for subjectID in good_subjects.index: 
        lat_source_perHz_fname = op.join(paths['meg_source_dir'], f'sub-CC{subjectID}', 'lat_soure_perHz')
        , f'lateralised_src_power_grad_multitaper_{freq}.csv')
        sub_lat_grid_pd = pd.read_csv(lat_source_perHz_fname)

        all_subs_lat_grid_perHz.append(sub_lat_grid_pd)


    
    return all_subs_lat_grid_perHz

