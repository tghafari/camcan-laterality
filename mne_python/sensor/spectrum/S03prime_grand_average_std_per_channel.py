"""
===============================================
S03prime. Grand average per channel

The aim of this code is to find out if there's a 
systematic level of noise in the sensors across
all participants in the std of welch segments.

Thia code will plot the grand average psd for
each channel on the right side and plots the left
channels grand average on top of that for comparison
in a plot_topo.

This code will:
    1. read each participant's epochs
    2. calculates the spectra for each channel separately
    3. put the spectra of each channel of each subject in 
    one list (total a list of 306 columns (#sensors) by 
     590 rows (#subjects))
    4. loops over subjects and appends to the list
    5. calculates grand average for each column of that
    list
    6. plots grand average psd of right sensor and left 
    sensor on top of each other in half a plot_topo 


written by Tara Ghafari
==============================================
ToDos:
    show freqs instead of times in plot_topos

Issues/ contributions to community:
  
Questions:
"""

# import libraries
import os.path as op
import os
import pandas as pd
import numpy as np
import mne
import sys 
import matplotlib.pyplot as plt

platform = 'mac'  # are you running on bluebear or windows or mac?

# import libraries
import os.path as op
import os
import pandas as pd
import numpy as np
import mne
import sys 
import matplotlib.pyplot as plt

platform = 'mac'  # are you running on bluebear or windows or mac?
test_plot = False  # do you want sanity check plots?

def pick_sensor_pairs_epochs_array(epochs, right_sensor, left_sensor):
    """this definition will read right and left sensor data from
     epochs into arrays for later use. 
     Shape of arrays:(1, 1, 117501) #epochs, #sensors, #times"""
    
    print('picking right and left sensors')
    array_right_sensor = epochs.copy().pick(picks=right_sensor).get_data(copy=True)  # freqs is just for a reference
    array_left_sensor = epochs.copy().pick(picks=left_sensor).get_data(copy=True)

    return array_right_sensor, array_left_sensor

def calculate_std_segs(sensor_array, sfreq, fmin, fmax):
    """
    The data are divided into sections being 2 s long. 
      (n_fft = 500 samples) with a 1 s overlap 
      (250 samples). This results in a 0.5 Hz 
      resolution Prior to calculating the FFT of each 
      section a Hamming taper is multiplied.
      average of segments is None for later calculating
      the std over the segments.
      n_fft=500, fmin=1, fmax=120, average=None"""
    
   # define constant parameters
    n_fft = sfreq*2
    welch_params = dict(fmin=fmin, fmax=fmax, n_fft=int(n_fft), 
                        n_overlap=int(n_fft/2), 
                        window='hamming', sfreq=sfreq,
                        remove_dc=True, average=None)

    # calculate power spectrum
    """the returned array will have the same shape 
    as the input data plus two additional dimensions 
    orresponding to frequencies and the unaggregated segments, respectively.
    This returns psds and freqs in a tuple"""
    print('calculating psd array welch')
    sensor_array_welch_power = mne.time_frequency.psd_array_welch(sensor_array,  
                                        **welch_params,
                                        n_jobs=30,
                                        verbose=True)
    
    sensor_array_psds_segs = sensor_array_welch_power[0]  # psds-> Shape: (1, 1, 239, 469)  #epochs, #channels, #freqs, #segments
    sensor_array_freqs_segs = sensor_array_welch_power[1] # freqs-> Shape: (239,)

    # Calculate the standard deviation along the welch_windows axis (axis=3) and squeeze epoch dimension (which is 1)
    print('calculating std over segments')
    sensor_std_segs = np.std(sensor_array_psds_segs, axis=3)  # Shape: (1, 1, 239) -> #epochs, #channels, #freqs

    return sensor_std_segs, sensor_array_welch_power, sensor_array_psds_segs, sensor_array_freqs_segs

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
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')  #sensor_layout_name_grad_no_central.csv
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise')

# Read only data from subjects with good preprocessed data
good_subject_pd = pd.read_csv(good_sub_sheet)
good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# Read sensor layout sheet
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)
sub_IDs = []
psd_all_sens_all_subs = []

for i, subjectID in enumerate(good_subject_pd.index):
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    try:
        print(f'Reading subject # {i}')
                    
        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
        _, epochspectrum_arr, freqs = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)  #epochspectrum_arr shape: #epochs, #sensors, #freqs

        # Average PSDs across epochs to get one PSD per channel -> basically squeezing as epoch is only one
        psds_mean = np.mean(epochspectrum_arr, axis=0)
        
        # Append the PSDs to the list - shape: #subs, #sensors, #freqs
        psd_all_sens_all_subs.append(psds_mean)

    except:
        print(f'an error occured while reading subject # {subjectID} - moving on to next subject')
        pass

# Convert list to numpy array for easier manipulation
psd_all_sens_all_subs_arr = np.array(psd_all_sens_all_subs)  # shape: #subjects, #channels, #freqs

# Calculate grand average PSD for each channel
grand_average_psd = np.mean(psd_all_sens_all_subs_arr, axis=0)  # shape: #channels, #freqs

# Filter out frequencies below 60 Hz - to show low amplitudes in high freqs better
min_freq = 60
freq_mask = np.where(freqs > min_freq)[0]
filtered_freqs = freqs[freq_mask]

# Create Evoked objects for right and left
all_right_names = [f'{row["right_sensors"][0:8]}' for _, row in sensors_layout_names_df.iterrows()]
all_left_names = [f'{row["left_sensors"][0:8]}' for _, row in sensors_layout_names_df.iterrows()]

right_epochs = epochs.copy().pick(all_right_names)
left_epochs = epochs.copy().pick(all_left_names)

right_ch_indices = [epochs.info['ch_names'].index(ch) for ch in sensors_layout_names_df['right_sensors']]
left_ch_indices = [epochs.info['ch_names'].index(ch) for ch in sensors_layout_names_df['left_sensors']]

grand_average_psd_right = grand_average_psd[right_ch_indices][:, freq_mask]
grand_average_psd_left = grand_average_psd[left_ch_indices][:, freq_mask]

evoked_right = mne.EvokedArray(grand_average_psd_right, right_epochs.info, tmin=0, comment=f'right sensors')
evoked_left = mne.EvokedArray(grand_average_psd_left, right_epochs.info, tmin=0, comment=f'left sensors')

# Replace time of evoked objects with freqs to have x-axis in frequency range
"""still needs work, times has no setter in evoked array
evoked_right = evoked_right.crop(tmin=filtered_freqs[0], tmax=None)
evoked_left = evoked_left.crop(tmin=filtered_freqs[0], tmax=filtered_freqs[-1])

evoked_right.times = filtered_freqs
evoked_left.times = filtered_freqs
"""

# Create a figure for the topomap plot
grand_avg_fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Plot the right sensors' evoked responses
evoked_right.plot_topo(axes=ax, color='orange', show=False, legend=False)

# Plot the left sensors' evoked responses (on right sensors' locations)
evoked_left.plot_topo(axes=ax, color='blue', show=False, legend=False)

# Adding a custom legend to the plot
handles = [plt.Line2D([0], [0], color='orange', lw=2),
           plt.Line2D([0], [0], color='blue', lw=2)]
labels = ['Right Sensors', 'Left Sensors']
plt.legend(handles, labels, loc='upper right')

# Adding title and adjusting plot properties
plt.title('Grand Average PSD Right vs Left Sensors')
ax.grid(True)
#plt.show()

# Save
grand_avg_fig_output_fname = op.join(op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots', 'grand_average_plot_topo_60.tiff')
                                     )
grand_avg_fig.savefig(grand_avg_fig_output_fname, dpi=1500)