
"""
===============================================
S02_left_right_spectra_per_sub

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
plot_psds = True  # do you want to plot the PSDs or only the lat spectra?

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

def pick_sensor_pairs_epochspectrum(epochspectrum, right_sensor, left_sensor):
    """this code will pick sensor pairs for calculating lateralisation 
        from epochspectrum (output of previous function).
        the shape of psd is (1, 1, 119) = #epochs, #sensors, #freqs
        freqs = np.arange(1, 60.5, 0.5)"""
    
    psd_right_sensor, freqs = epochspectrum.copy().pick(picks=right_sensor).get_data(return_freqs=True)  # freqs is just for a reference
    psd_left_sensor = epochspectrum.copy().pick(picks=left_sensor).get_data()

    # Squeeze for easier calculations later
    psd_right_sensor_squeeze = psd_right_sensor.squeeze()  # squeezable as there's only one sensor and one epoch
    psd_left_sensor_squeeze = psd_left_sensor.squeeze()

    return psd_right_sensor_squeeze, psd_left_sensor_squeeze, freqs

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

    # Calculate the standard deviation along the welch_windows axis (axis=3) 
    print('calculating std over segments')
    sensor_std_segs = np.std(sensor_array_psds_segs, axis=3)  # Shape: (1, 1, 239) -> #epochs, #channels, #freqs

    std_array_sensor = sensor_std_segs.squeeze()  # squeezable as there's only one epoch and one sensor -> (239,) #freqs

    return std_array_sensor, sensor_array_freqs_segs

def calculate_spectrum_lateralisation(psd_right_sensor, psd_left_sensor, std_right_sensor, std_left_sensor):
    """ calculates lateralisation index for each pair of sensors."""

    # Perform element-wise subtraction and division
    subtraction = psd_right_sensor - psd_left_sensor
    sum_element = psd_right_sensor + psd_left_sensor
    spectrum_lat_sensor_pairs = subtraction / sum_element

    log_lateralisation = np.log(psd_right_sensor/psd_left_sensor)
    
    # Perform element-wise subtraction and division
    subtraction_std_pairs = std_right_sensor - std_left_sensor
    sum_std_element = std_right_sensor + std_left_sensor
    sumsub_std_lat_sensor_pairs = subtraction_std_pairs / sum_std_element


    return subtraction, spectrum_lat_sensor_pairs, log_lateralisation, sumsub_std_lat_sensor_pairs, subtraction_std_pairs

def remove_noise_bias(lateralised_power, freqs, h_fmin, h_fmax):
    """to eliminate noise bias in sensors, this definition 
    calculates the average of lateralised power in high 
    frequencies (between h_fmin=90, h_fmax=120), where we don't 
    expect much brain signal, and then subtracts
    that bias from the lateralisation index for each sensor pair
    lateralised_power: output of calculate_spectrum_lateralisation
    (works for all methods, i.e, subtraction, sumsub and log)
    freqs: output of pick_sensor_pairs_epochspectrum
    fmin and fmax: the high frequency section to use for nosie bias """
    
    print(f'Removing noise of {h_fmin} and {h_fmax} from lateralised power')

    # Calculate average power in the specified frequency range
    average_power_high_freqs = np.mean(lateralised_power[(freqs >= h_fmin) & (freqs <= h_fmax)])

    # Subtract the average power from all power values 
    bias_removed_log_lat = lateralised_power.copy()  
    bias_removed_log_lat -= average_power_high_freqs    

    return bias_removed_log_lat

# Define where to read and write the data
if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
output_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots/sanity-checks')

# Preallocate lists
subjectIDs_to_plot = [620935, 721704] #[321506] # manually add subjects to plot
sensor_pairs_to_plot = [['MEG0233','MEG1343'], ['MEG0422', 'MEG1112']]  # first is left sensor, second is right sensor in each pair


for subjectID in subjectIDs_to_plot:
    # Read subjects one by one and calculate lateralisation index for each pair of sensor and all freqs
    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
    epoched_fif = op.join(epoched_dir, epoched_fname)
    stacked_sensors = []

    print(f'Reading subject # {subjectID}')
                
    epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # one 7min50sec epochs
    epochspectrum = calculate_spectral_power(epochs, n_fft=500, fmin=1, fmax=120)   # changed n_fft to 2*info['sfreq'] which after preprocessing is 250 (not 1000Hz)

        # Read sensor pairs and calculate lateralisation for each
    for i in range(0, len(sensor_pairs_to_plot)):
            print(f'plotting {sensor_pairs_to_plot[i]}')

            working_pair = sensor_pairs_to_plot[i]

            psd_right_sensor, psd_left_sensor, freqs = pick_sensor_pairs_epochspectrum(epochspectrum, 
                                                                                      working_pair[1], 
                                                                                      working_pair[0],
                                                                                      ) 
            print(f'Calculating lateralisation in {working_pair[1]}, {working_pair[0]}')                   
            array_right_sensor, array_left_sensor = pick_sensor_pairs_epochs_array(epochs,
                                                                                    working_pair[1], 
                                                                                    working_pair[0])
            epochs_std_wins_right, _, = calculate_std_segs(array_right_sensor, 
                                                                sfreq=epochs.info['sfreq'], # 250.0Hz
                                                                fmin=1, 
                                                                fmax=120) 
            epochs_std_wins_left, _,= calculate_std_segs(array_left_sensor, 
                                                            sfreq=epochs.info['sfreq'], 
                                                            fmin=1, 
                                                            fmax=120)            
            if plot_psds:
                # Plot PSDs 
                fig, axes = plt.subplots(3, 2, figsize=(16, 12))
                axes[0,1].plot(freqs, psd_right_sensor)
                axes[0,1].set_title(f'Right sensor {working_pair[1]}')
                axes[0,1].set(title=f"{subjectID} in {working_pair[1]}",
                        xlabel="Frequency (Hz)",
                        ylabel="Power",
                        )
                axes[1,1].plot(freqs, np.log(psd_right_sensor), color='darkorange')
                axes[1,1].set_title(f'Log-Right sensor {working_pair[1]}')
                axes[1,1].set(title=f"{subjectID} in {working_pair[1]}",
                        xlabel="Frequency (Hz)",
                        ylabel="log Power",
                        )
                axes[2,1].plot(freqs, epochs_std_wins_right, color='darkred')
                axes[2,1].set_title(f'std-Right sensor {working_pair[1]}')
                axes[2,1].set(title=f"{subjectID} in {working_pair[1]}",
                        xlabel="Frequency (Hz)",
                        ylabel="std of Power",
                        )
                # Plot PSDs with log transform
                axes[0,0].plot(freqs, psd_left_sensor)
                axes[0,0].set_title(f'Left sensor {working_pair[0]}')
                axes[0,0].set(title=f"{subjectID} in {working_pair[0]}",
                        xlabel="Frequency (Hz)",
                        ylabel="Power",
                        )
                axes[1,0].plot(freqs, np.log(psd_left_sensor), color='darkorange')
                axes[1,0].set_title(f'Log-Left sensor {working_pair[0]}')
                axes[1,0].set(title=f"{subjectID} in {working_pair[0]}",
                        xlabel="Frequency (Hz)",
                        ylabel="log Power",
                        )
                axes[2,0].plot(freqs, epochs_std_wins_left, color='darkred')
                axes[2,0].set_title(f'std-Left sensor {working_pair[0]}')
                axes[2,0].set(title=f"{subjectID} in {working_pair[0]}",
                        xlabel="Frequency (Hz)",
                        ylabel="std of Power",
                        )
                fig.suptitle(f'Single sensor spectra for {subjectID}')
                plt.tight_layout()
                fig.savefig(op.join(output_dir, f'sub_{subjectID}_{working_pair[1]}_{working_pair[0]}.png'))
                plt.close()    

            (subtraction, spectrum_lat_sensor_pairs, 
             log_lateralisation, sumsub_std_lat_sensor_pairs,
             sub_std_lat_sensor_pairs) = calculate_spectrum_lateralisation(psd_right_sensor, 
                                                                            psd_left_sensor, 
                                                                            epochs_std_wins_right, 
                                                                            epochs_std_wins_left)
            sub_bias_removed_lat = remove_noise_bias(subtraction, freqs, h_fmin=90, h_fmax=120)
            log_bias_removed_lat = remove_noise_bias(log_lateralisation, freqs, h_fmin=90, h_fmax=120)
            
            # Plot spectrum sumsub of these sensors vs. frequency
            fig, axes = plt.subplots(7, 1, figsize=(20, 10))
            axes[0].plot(freqs, spectrum_lat_sensor_pairs, label='sumsub')
            axes[0].set(title=f"subsum lateralisation spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (subsum)",
            )

            # Plot spectrum subtraction lateralisation of these sensors vs. frequency
            axes[1].plot(freqs, subtraction, color='firebrick', label='subtraction')
            axes[1].set(title=f"subtraction lateralisation spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (subtraction)",
            )

            # Plot spectrum subtraction lateralisation noise removed of these sensors vs. frequency
            axes[2].plot(freqs, sub_bias_removed_lat, color='tomato', label='subtraction noise removed')
            axes[2].set(title=f"subtraction lateralisation no noise spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (sub-nonoise)",
            )
            
            # Plot spectrum log lateralisation of these sensors vs. frequency
            axes[3].plot(freqs, log_lateralisation, color='darkorange', label='log transformed')
            axes[3].set(title=f"log lateralisation spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (log)",           
            )

            # Plot spectrum log lateralisation noise removed of these sensors vs. frequency
            axes[4].plot(freqs, log_bias_removed_lat, color='burlywood', label='log transformed noise removed')
            axes[4].set(title=f"log lateralisation spectrum no noise for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (log-nonoise)",
            )
            
            # Plot spectrum std of power of these sensors vs. frequency
            axes[5].plot(freqs, sumsub_std_lat_sensor_pairs, color='darkred', label='sumsub-std of welch segments')
            axes[5].set(title=f"sumsub-std of spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (sumsub-std)",
            )

            # Plot spectrum std of power of these sensors vs. frequency
            axes[6].plot(freqs, sub_std_lat_sensor_pairs, color='red', label='sub-std of welch segments')
            axes[6].set(title=f"sub-std of spectrum for {subjectID} in {working_pair[0]}_{working_pair[1]}",
            xlabel="Frequency (Hz)",
            ylabel="Lateralisation Index (sub-std)",
            )
            
            for ax in axes:
                ax.legend()
            fig.savefig(op.join(output_dir, f'sub_{subjectID}_{working_pair[1]}_{working_pair[0]}_seven_lateralisation_spectra.png'))
            plt.close()
