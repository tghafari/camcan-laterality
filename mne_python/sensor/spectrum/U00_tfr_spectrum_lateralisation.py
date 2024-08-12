"""
===============================================
S01. Spectrum lateralisation

This code will:
    1. Loads in one subject
    2. calculates tfr multitaper for 50 frequencies
    and 306 sensors
    3. loops over frequencies
    4. finds channel names on the right and their
    corresponding channels on the left
    5. calculates lateralisation index for that 
    subject and that sensor pair
    6. saves the lateralisation index of all sensor
    pairs and all frequencies for one subject
    in one dataframe
    7. read subjects from an array in a bash script
    for parallelisation

    the dataframes will be formed into their final
    structure (folders:freqs, csvs:sensor pairs,
    rows:subjects) in 
    S02_lat_index_dataframe_organisation


written by Tara Ghafari
==============================================
ToDos:

Issues/ contributions to community:
  
Questions:
"""
# import libraries
import os.path as op
import os
import pandas as pd
import numpy as np
import mne

bluebear = True  # are you running on bluebear or windows?

# Define functions
def calculate_spectrum_lateralisation(right_power, left_power):
    # calculate lateralisation index for each pair of sensors for all freqs
    spectrum_lat_sensor_pairs = (right_power.data - left_power.data) /\
                                (right_power.data + left_power.data)
    return spectrum_lat_sensor_pairs

# Define where to read and write the data
if bluebear:
    rds_dir = '/rds/projects/q/quinna-camcan'
    epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched_data')
    info_dir = op.join(rds_dir, 'dataman/data_information')
else:
    epoched_dir = r'X:/derivatives/meg/sensor/epoched_data'
    info_dir = r'X:/dataman/data_information'
    output_dir = r'X:/derivatives/meg/sensor/lateralized_index/frequency_bins'

good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/frequency_bins')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Define parameters of tfr_multitaper
freqs = np.append([0.1, 0.5], np.arange(1, 51, 0.5))  # tfr_multitaper does not work on 0, so had to replace it with 0.1
n_cycles = freqs / 2
time_bandwidth = 2.0

# Read subjects one by one and calculate lateralisation index for each pair of sensor
all_subject_IDs = []
sensor_pairs = []
all_sensor_pairs_all_freqs_one_sub_lat = []
all_freqs = []

for j, subject_ID in enumerate(good_subject_pd[375:385].index):
    epoched_fname = 'sub-CC' + str(subject_ID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)

    try:
        print(f'calculating lateralisation in subject # {subject_ID}, {j}')
        # Read epoched file and calculate tfr            
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)
        all_sensors_all_freqs_one_sub_tfr = mne.time_frequency.tfr_multitaper(epochs, 
                                                freqs=freqs, 
                                                n_cycles=n_cycles,
                                                time_bandwidth=time_bandwidth, 
                                                picks=['meg','grad'],
                                                use_fft=True, 
                                                return_itc=False,
                                                average=True, 
                                                decim=2,
                                                n_jobs=12,
                                                verbose=True)

        # Calculate laterality index for each sensor pair
        for _, row in sensors_layout_names_df.iterrows():
            one_sensor_pair_all_freqs_one_sub_lat = []

            # Find right sensor power and corresponding left sensor power
            print(f'reading {row["right_sensors"][1:8]} _ {row["left_sensors"][1:8]} from TFR')
            right_powers = all_sensors_all_freqs_one_sub_tfr.copy().pick(row['right_sensors'][1:8])
            left_powers = all_sensors_all_freqs_one_sub_tfr.copy().pick(row['left_sensors'][1:8])

            try:
                # Calculate laterality index for all pairs of sensors
                """ the shape of this variable  is (#sensor_pair=1, freqs=102, time=626) """
                print('calculating laterality index')
                one_sensor_pair_all_freqs_one_sub_lat = calculate_spectrum_lateralisation(right_powers, left_powers)
                # flatten and get rid of the first dimension
                flat_one_sensor_pair_all_freqs_one_sub_lat = [item for sublist in one_sensor_pair_all_freqs_one_sub_lat for item in sublist]

                # Create one array for sensor pair names and one for lateralisation values and one for frequencies
                current_sensor_pair = [f'{row["right_sensors"][1:8]} _ {row["left_sensors"][1:8]}'] * len(freqs)
                sensor_pairs.extend(current_sensor_pair)
                all_sensor_pairs_all_freqs_one_sub_lat.extend(flat_one_sensor_pair_all_freqs_one_sub_lat)
                all_freqs.extend(freqs)

            except:
                print(f'Error in laterality for {row["right_sensors"][1:8]} and {row["left_sensors"][1:8]} for #{subject_ID}')
                all_sensor_pairs_all_freqs_one_sub_lat.extend(np.nan)  # put nan for subjects for which we couldn't calculate spectral lat
                pass

        # Create subject_id column for the main df
        current_subject_ID = [subject_ID] * len(freqs) * int(len(all_sensors_all_freqs_one_sub_tfr.ch_names)/2)  # ensures correct number of rows
        all_subject_IDs.extend(current_subject_ID)
    
    except:
        print(f'an error occured while reading subject # {subjectID}, {j}')
        pass
    
# Save data for all subjects
print('saving data to dataframe')
data = {'subject_ID':all_subject_IDs, 'sensor_pairs':sensor_pairs, 
'freqs': all_freqs, 'lateralised_spec':all_sensor_pairs_all_freqs_one_sub_lat}
all_subs_all_sensor_pairs_all_freqs_df = pd.DataFrame(data=data)

# Save to disc 
print('saving data to disc')
output_fname = op.join(output_dir, 'ten_subs_all_sensor_pairs_all_freqs_lat_index.csv')
all_subs_all_sensor_pairs_all_freqs_df.to_csv(output_fname)

