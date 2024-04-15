
"""
===============================================
CS06_left_right_spectra_per_sub

the aim of this code is to plot the spectra for 
a few participants that were selected from 
the cloud plots on right and left, separately.


This code will:
    1. inputs participnats id
    2. input a list of sensors for which
    we'd want to plot the spectrum
    3. plot right and left spectra, separately
    4. plot both side spectra 


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
import sys 
import matplotlib.pyplot as plt

platform = 'mac'


def calculate_spectral_power(epochs, n_fft, fmin, fmax):
    """
    The data are divided into sections being 2 s long. 
      (n_fft = 500 samples) with a 1 s overlap 
      (250 samples). This results in a 0.5 Hz 
      resolution Prior to calculating the FFT of each 
      section a Hamming taper is multiplied.
      n_fft=500, fmin=1, fmax=60"""
    
   # define constant parameters
    welch_params = dict(fmin=fmin, fmax=fmax, picks="meg", n_fft=n_fft, n_overlap=int(n_fft/2))

    # calculate power spectrum for right and left sensors separately
    """the returned array will have the same
      shape as the input data plus an additional frequency dimension"""
    epochspectrum = epochs.compute_psd(method='welch',  
                                        **welch_params,
                                        n_jobs=30,
                                        verbose=True)

    return epochspectrum

def pick_sensor_pairs_epochspectrum(epochspectrum, right_sensor, left_sensor, output_dir, subjectID):
    """this code will pick sensor pairs for calculating lateralisation 
        from epochspectrum (output of previous function).
        the shape of psd is (1, 1, 119) = #epochs, #sensors, #freqs
        freqs = np.arange(1, 60.5, 0.5)"""
    
    psd_right_sensor, freqs = epochspectrum.copy().pick(picks=right_sensor).get_data(return_freqs=True)  # freqs is just for a reference
    psd_left_sensor = epochspectrum.copy().pick(picks=left_sensor).get_data()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    psd_right_plot = epochspectrum.copy().pick(picks=right_sensor).plot(axes=axes[1])
    axes[1].set_title(f'Right sensor {right_sensor}')

    psd_left_plot = epochspectrum.copy().pick(picks=left_sensor).plot(axes=axes[0])
    axes[0].set_title(f'Left sensor {left_sensor}')

    fig.suptitle(f'Single sensor spectra for {subjectID}')
    plt.tight_layout()
    fig.savefig(op.join(output_dir, f'sub_{subjectID}_{right_sensor}_{left_sensor}.png'))
    plt.close()

    # Plot PSDs with no log transform
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[1].plot(freqs, psd_right_sensor.squeeze())
    axes[1].set_title(f'Right sensor {right_sensor}')
    axes[1].set(title=f"{subjectID} in {right_sensor}",
            xlabel="Frequency (Hz)",
            ylabel="Power",
            )

    axes[0].plot(freqs, psd_left_sensor.squeeze())
    axes[0].set_title(f'Left sensor {left_sensor}')
    axes[0].set(title=f"{subjectID} in {left_sensor}",
            xlabel="Frequency (Hz)",
            ylabel="Power",
            )

    fig.suptitle(f'Single sensor spectra for {subjectID}')
    plt.tight_layout()
    fig.savefig(op.join(output_dir, f'sub_{subjectID}_{right_sensor}_{left_sensor}_nolog.png'))
    plt.close()    

    return psd_right_sensor, psd_left_sensor, freqs


def calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor):
    """ calculates lateralisation index for each pair of sensors."""

    psd_right_sensor = psd_right_sensor.squeeze()  # squeezable as there's only one sensor and one epoch
    psd_left_sensor = psd_left_sensor.squeeze()

    # Perform element-wise subtraction and division
    subtraction = psd_right_sensor - psd_left_sensor
    sum_element = psd_right_sensor + psd_left_sensor
    spectrum_lat_sensor_pairs = subtraction / sum_element

    return spectrum_lat_sensor_pairs

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots')

# Preallocate lists
subjectIDs_to_plot = [321506, 620935, 721704]  # manually add subjects to plot
sensor_pairs_to_plot = [['MEG0422', 'MEG1112'],['MEG0233','MEG1343']]  # first is left sensor, second is right sensor in each pair


for subjectID in subjectIDs_to_plot:
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    print(f'Reading subject # {subjectID}')
                
    epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
    epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=60)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250 (not 1000Hz)

        # Read sensor pairs and calculate lateralisation for each
    for i in range(0, len(sensor_pairs_to_plot)):
            print(f'plotting {sensor_pairs_to_plot[i]}')

            working_pair = sensor_pairs_to_plot[i]

            psd_right_sensor, psd_left_sensor, freqs = pick_sensor_pairs_epochspectrum(epochspectrum, 
                                                                                    working_pair[1], 
                                                                                    working_pair[0],
                                                                                    output_dir,
                                                                                    subjectID) 
        
            spectrum_lat_sensor_pairs = calculate_spectrum_lateralisation(psd_right_sensor, 
                                                                        psd_left_sensor)
            
            # Plot spectrum lateralisation of these sensors vs. frequency
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(freqs, spectrum_lat_sensor_pairs)
            ax.set(title=f"lateralisation spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index",
            )

            fig.savefig(op.join(output_dir, f'sub_{subjectID}_{working_pair[1]}_{working_pair[0]}_lateralisation_spectra.png'))
            plt.close()