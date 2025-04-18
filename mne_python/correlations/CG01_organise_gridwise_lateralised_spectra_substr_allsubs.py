"""
=========================================================
CG01_organise_gridwise_lateralised_spectra_substr_allsubs:

This script processes lateralized MEG source power data from multiple subjects.

Steps:
    1. Sets up file paths based on the platform ('mac' or 'bluebear').
    2. Loads a list of valid subjects from a demographics file.
    3. Iterates over two sensor types ('grad' and 'mag').
    4. For each frequency (1.5Hz to 59.5Hz), it:
        - Reads the corresponding CSV file for each subject.
        - Rejects missing subjects.
        - Merges all subjects' data into a single DataFrame.
        - Saves the compiled data as a CSV in the appropriate directory.

Output:
- 235 CSV files (one per 0.5Hz step) for each sensor type.
- Each CSV contains 7011 rows (source grids) x #subjects.

Usage:
Run the script directly to process all data.

written by Tara Ghafari
tara.ghafari@gmail.com
=========================================================
"""

import os
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
        'meg_source_dir': os.path.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer'),
        'meg_source_all_subs_dir': os.path.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer/all_subs'),
        'good_sub_sheet': os.path.join(rds_dir, 'dataman/data_information/demographics_goodPreproc_subjects.csv')
    }
    return paths

def load_subjects(good_sub_sheet):
    """Load subject IDs from a CSV file."""
    good_subjects = pd.read_csv(good_sub_sheet)
    return good_subjects.iloc[:, 0].astype(str).tolist()  # Extract subject IDs

def process_meg_data(platform='bluebear', freqs=np.arange(1.5, 11, 0.5), sensortypes=['grad', 'mag']):
    """Process MEG data for all subjects and frequencies, saving compiled CSVs."""
    paths = setup_paths(platform)
    subjects = load_subjects(paths['good_sub_sheet'])
    
    for sensor in sensortypes:
        print(f'processing {sensor}')
        for freq in freqs:
            print(f'Processing {freq} Hz')
            all_data = []
            
            for subj in subjects:
                file_path = os.path.join(paths['meg_source_dir'], f'sub-CC{subj}', 'lat_source_perHz',
                                         f'lateralised_src_power_{sensor}_multitaper_{freq}.csv')
                if os.path.exists(file_path):
                    # print(f'Reading {subj}')
                    data = pd.read_csv(file_path, header=None)  # Read CSV without headers
                    data = data[1]  # only keep the lateralised source power
                    if data.isnull().values.any():
                        print(f'{subj} has a NaN')
                    data[0] = str(subj)  # Add subject code as column header
                    all_data.append(data)
                else:
                    print(f"Missing: {file_path}, moving on to the next subj")
                    # nan_df = pd.DataFrame(np.nan, index=range(7011), columns=[subj])
                    # all_data.append(nan_df)
                    
            # Combine all subjects' data into one DataFrame
            combined_df = pd.concat(all_data, axis=1)
            print(f'{sum(sum(combined_df.isnull().values))}')
            # Define output filename
            output_file = os.path.join(paths['meg_source_all_subs_dir'],
                                       f'all_subs_lateralised_src_power_{sensor}_{freq}.csv')
            combined_df.to_csv(output_file, index=False, header=False)
            print(f"Saved: {output_file}")

if __name__ == "__main__":
    process_meg_data()
