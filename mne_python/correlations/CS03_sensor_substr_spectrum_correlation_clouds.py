
"""
===============================================
CS03. Sensor pair lateralisation index vs substr
laterlisation (cloud plots)

the aim of this code is to plot lateralised power
of each sensor pair vs lateralised index of each 
subcortical structure.
this is ensure moving from one sensor to another 
sensor in one participant does not change the power 
drastically.


This code will:
    1. make a df for each pair of sensors and all 
    substr laterality indices
    2. plots value of lateralised power for each
    sensor pair in each frequency bin and all 
    participants vs the value of volume 
    lateralisation index for all substr 


written by Tara Ghafari
==============================================
ToDos:

Issues/ contributions to community:
  
Questions:
"""

import pandas as pd
import numpy as np
import os.path as op
import os
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def working_df_maker(spectra_dir, left_sensor, right_sensor, substr_lat_df):
    """This definition merges the dataframes containing spectrum lateralisation values and 
    substr lateralisation values together"""

    # Navigate to the sensor_pair folder
    spec_lat_index_fname = op.join(spectra_dir, f'{left_sensor}_{right_sensor}.csv')

    # Load lateralisation index for each pair
    spectrum_pair_lat_df = pd.read_csv(spec_lat_index_fname)
    spectrum_pair_lat_df = spectrum_pair_lat_df.rename(columns={'Unnamed: 0':'subject_ID'})
    
    # Merge and match the subject_ID column and remove nans
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['subject_ID'])
    working_df = working_df.dropna()
    working_df = working_df.reset_index(drop=True)  # reset the index to have continued indexing

    # Get the freqs of spectrum from spec_pair_lat
    freqs = spectrum_pair_lat_df.columns.values[1:]  # remove subject_ID column
    freqs = [float(freq) for freq in freqs]  # convert strings to floats
    return working_df, freqs

platform = 'bluebear'
all_sensors = False  # do you want to plot all sensors?
random_selection = False  # do you want to plot only a subgroup of participants?
correlation_method = 'linear_regression'  # linear_regression of spearman?

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    
# Define the directory 
info_dir = op.join(rds_dir, 'dataman/data_information')
deriv_dir = op.join(rds_dir, 'derivatives') 
spectra_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_nooutliers_absolute-thresh')
substr_dir = op.join(deriv_dir, 'mri/lateralized_index')
substr_sheet_fname = op.join(substr_dir, 'lateralization_volumes.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')
fig_output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/cloud-substr-freq-subtraction-nooutlier-psd')

# Load substr file
substr_lat_df = pd.read_csv(substr_sheet_fname)

sensor_pairs_to_plot = [   # first is left sensor, second is right sensor in each pair
    ['MEG0233','MEG1343'], 
    ['MEG0422','MEG1112'],
    ['MEG1933','MEG2333'],
    ['MEG1941','MEG2321']
] 
random_subject_num = 20  # if random_selection == True, how many participants you want in the subgroup?


# Read sensor layout sheet
if all_sensors:
    sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)
else:
    sensors_layout_names_df = pd.DataFrame(sensor_pairs_to_plot,
                                           columns=['left_sensors', 'right_sensors'])  # first is left sensor, second is right sensor in each pair

substrs = ['Caud']
#['Thal', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']


for i, row in sensors_layout_names_df.iterrows():
    print(f'Working on pair {row["left_sensors"][0:8]}, {row["right_sensors"][0:8]}')

    # Get the frequencies of spectrum (only once enough)
    _, freqs = working_df_maker(spectra_dir, 
                                row["left_sensors"][0:8], 
                                row["right_sensors"][0:8], 
                                substr_lat_df) if i == 0 else (None, freqs)

    # Make the working df containing lateralised value of the current sensor pair
    working_df, _ = working_df_maker(spectra_dir,  # shape: #subject by #freqs + #substr + 1(for subject_ID column) = 560 * 481
                                    row["left_sensors"][0:8], 
                                    row["right_sensors"][0:8], 
                                    substr_lat_df)
    if random_selection:
        working_df = working_df.sample(n=random_subject_num, random_state=97)
    
    # Create a color map to assign a different color to each participant (for comparison with other plots)
    num_subjects = len(working_df)
    colors = plt.cm.RdBu(np.linspace(0, 1, num_subjects))

    output_cloud_dir = op.join(fig_output_dir, f'{row["left_sensors"][0:8]}_{row["right_sensors"][0:8]}')
    if not op.exists(output_cloud_dir):
        os.makedirs(output_cloud_dir)
 
    # Calculate correlation in each substr
    for substr in substrs:
        print(f'Working on {substr}')
        
        output_cloud_substr_dir = op.join(output_cloud_dir, f'{substr}')
        if not op.exists(output_cloud_substr_dir):
            os.makedirs(output_cloud_substr_dir)
    
        # Calculate correlation with each freq 
        for freq in freqs[:120]:
            print(f'Plotting clouds for {freq} Hz')

            # Plot each point with a different color
            plt.figure(figsize=(10, 6))
            for i, (x, y) in enumerate(zip(working_df[f'{freq}'], working_df[substr])):
                plt.plot(x, y, marker='o', linestyle=' ', color=colors[i], label=f'{freq}')
                plt.text(x, y, str(working_df['subject_ID'][i]), fontsize=4, verticalalignment='bottom', horizontalalignment='right')  # add subject id next to its dot
        
                if correlation_method == 'linear_regression':
                    coeffs = np.polyfit(working_df[f'{freq}'], working_df[substr], 1)  # Linear regression
                    regression_line = np.poly1d(coeffs)
                    
                    x_values = np.linspace(min(working_df[f'{freq}']), max(working_df[f'{freq}']), 100)
                    plt.plot(x_values, regression_line(x_values), color='red', label='Regression Line')
                    
                    # Add linear regression equation as text
                    eq_str = f'Y = {coeffs[0]:.4f} * X + {coeffs[1]:.4f}'
                    plt.text(0.1, 0.9, eq_str, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

                elif correlation_method == 'spearman':  # Calculate Spearman correlation
                    correlation, _ = spearmanr(working_df[f'{freq}'], working_df[substr])

                    # Plot a line representing the Spearman correlation
                    x_values = np.linspace(min(working_df[f'{freq}']), max(working_df[f'{freq}']), 100)
                    y_values = correlation * x_values  # Spearman correlation line
                    plt.plot(x_values, y_values, color='red', label='Spearman Correlation Line')

                    # Add Spearman correlation coefficient as text
                    plt.text(0.1, 0.9, f'Spearman Correlation: {correlation:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # Add labels and title
            plt.xlabel(f'Lateralised Power in {row["left_sensors"][0:8]}, {row["right_sensors"][0:8]}')
            plt.ylabel(f'Lateralisation Index of {substr}')
            plt.title(f'Lateralisation Indices in {freq} Hz')
            plt.grid(True)

            #plt.show()
            fig_output_fname = op.join(output_cloud_substr_dir, f'{freq}.png')
            plt.savefig(fig_output_fname)
            plt.close()
