"""
===============================================
S02. lateralised index dataframe organiser

This code will:
    1. read in all_subs_all_sensor_pairs_all_freqs_lat_index.csv
    from disk the structure of which is:
    - index, subject_ID, sensor_pair, frequency, lateralisation_index 
    2. sorts the big df first based on freq, then sensor_pair and 
    lastly subject_ID
    2. reorganises the csv file info 
    - 102 folders (freqs)
    - 153 csv files (sensor pairs) in each folder
    - 619 rows (# subjects) in each csv file
    each showing one lateralised index

written by Tara Ghafari
==============================================
"""

# import libraries
import os.path as op
import os
import pandas as pd

# Define where read the data from
rds_dir = '/rds/projects/q/quinna-camcan'
main_csv_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/frequency_bins')
csv_fname = op.join(main_csv_dir, 'all_subs_all_sensor_pairs_all_freqs_lat_index.csv')

# read the csv file back to df
df = pd.read_csv(csv_fname)

# Sort the DataFrame based on freqs, sensor_pairs, and subject_ID
print('sorting the big df')
df_sorted = df.sort_values(by=['freqs', 'sensor_pairs', 'subject_ID'])

# Group the DataFrame by 'freqs' and 'sensor_pairs' 
# for later slicing the big dataframe to smaller ones
print('grouping the sorted df by freq and sensor pair')
grouped = df_sorted.groupby(['freqs', 'sensor_pairs'])

# Create a dictionary to store the individual DataFrames
data_frames_dict = {}

# Iterate through the groups
print('making a dictionary of dataframes')
for (freq, sensor_pair), group_df in grouped:
    # Create a key for the dictionary based on freq and sensor_pair
    key = f'freq_{freq}_{sensor_pair}'
    
    # Store the DataFrame in the dictionary
    data_frames_dict[key] = group_df

# Access individual DataFrames from the dictionary
''' for example:
# df_freq_1_MEG1422_MEG0112 = data_frames_dict['freq_0.1_MEG1422 _ MEG0112']
'''

# Iterate through the unique freqs
print('saving dataframes into separate csv files')
for freq in df_sorted['freqs'].unique():
    # Create a directory for each freq
    freq_dir = op.join(main_csv_dir, f'{freq}')
    os.makedirs(freq_dir, exist_ok=True)

    # Get DataFrames for the current freq
    freq_data_frames = {key: df for key, df in data_frames_dict.items() if f'freq_{freq}' in key}

    # Iterate through the unique sensor_pairs
    for sensor_pair, df_sensor_pair in freq_data_frames.items():
        # Save the DataFrame for the current sensor_pair to a CSV file
        output_filename = op.join(freq_dir, f'{sensor_pair}.csv')
        df_sensor_pair.to_csv(output_filename, index=False)
