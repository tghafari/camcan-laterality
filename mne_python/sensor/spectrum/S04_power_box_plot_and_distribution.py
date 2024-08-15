"""
===============================================
S04_power_box_plot_and_distribution

This code will:
    1. Calculate Spectral Power: For each participant 
    and each sensor.
    2. Store the Information: The power values are 
    stored in a DataFrame for each sensor.
    3. Plot Box Plots with Scatter: The box plots 
    present interquartile range of power with scatter
    plot with different colours for each participant.
    4. To detect unhealthy sensors, this code
    will also plot the average power vs number of 
    participants for all sensors and all frequencies.

Outputs:
Saved DataFrames: Each sensor's PSD values across all 
                    subjects are saved as CSV files.
Box Plot Visualization: The box plot visualizes the 
                        distribution of PSD values across 
                        subjects for each sensor.
Power Distribution Plot: The histogram visualizes 
                        the distribution of power
                        values of sensors vs 
                        number of participants.

     
written by Tara Ghafari
==============================================
"""

# Import libraries
import os.path as op

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import mne

def calculate_spectral_power(epochs, n_fft, fmin, fmax):
    welch_params = dict(fmin=fmin, fmax=fmax, picks="meg", n_fft=n_fft, n_overlap=int(n_fft/2))
    epochspectrum = epochs.compute_psd(method='welch', **welch_params, n_jobs=30, verbose=True)
    return epochspectrum

def calculate_and_store_spectral_power(good_subject_pd, sensors_layout_names_df, epoched_dir, output_dir, n_fft=500, fmin=1, fmax=120):
    """This function calculate_and_store_spectral_power calculates 
        and stores the power spectral density (PSD) for each sensor 
        across all subjects.
        If the PSD for a sensor is already calculated and saved 
        it will be loaded instead of recalculated."""
    
    sensor_power_dataframes = {}
    freqs = None  # Placeholder for frequency array

    for _, row in sensors_layout_names_df.iterrows():
        for sensor in [row['right_sensors'][0:8], row['left_sensors'][0:8]]:
            sensor_csv_path = op.join(output_dir, f"{sensor}_power.csv")
            if op.exists(sensor_csv_path):
                # Load the data from the CSV file if it exists
                sensor_power_dataframes[sensor] = pd.read_csv(sensor_csv_path, index_col=0)
                if freqs is None:
                    freqs = sensor_power_dataframes[sensor].columns.astype(float).values
            else:
                # If any sensor data is missing, we must calculate the PSD for all subjects
                for i, subjectID in enumerate(good_subject_pd.index):
                    epoched_fname = 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif'
                    epoched_fif = op.join(epoched_dir, epoched_fname)

                    try:
                        print(f'Reading subject # {i}')
                        epochs = mne.read_epochs(epoched_fif, preload=True, verbose=True)  # Load epochs
                        epochspectrum = calculate_spectral_power(epochs, n_fft=n_fft, fmin=fmin, fmax=fmax)

                        # Pick the sensor's PSD
                        psd_sensor, freqs = epochspectrum.copy().pick(picks=sensor).get_data(return_freqs=True)
                        psd_sensor = psd_sensor.squeeze()  # Flatten to 1D

                        # Initialize DataFrame if not done already
                        if sensor not in sensor_power_dataframes:
                            sensor_power_dataframes[sensor] = pd.DataFrame(index=good_subject_pd.index, columns=freqs)

                        # Store PSD data in DataFrame
                        sensor_power_dataframes[sensor].loc[subjectID] = psd_sensor

                    except Exception as e:
                        print(f'Error reading subject # {subjectID}: {e}')
                        continue

                # Save the sensor power dataframes to CSV
                if sensor in sensor_power_dataframes:
                    print('saving...')
                    #sensor_power_dataframes[sensor].to_csv(sensor_csv_path)

    return sensor_power_dataframes, freqs


def plot_sensor_power(sensor_power_dataframes, sensors_layout_names_df, 
                      freqs, min_freq=60, max_freq = 120, output_dir=None):
    """This function plot_sensor_power 
        plots the box plots for each sensor, taking care of the 
        spatial arrangement of sensors. The scatter plot
        overlays individual subject data points on each box plot."""
    
    freq_mask = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    step = 5  # steps to cover freq range, to reduce the number of points
    filtered_freqs = freqs[freq_mask][::step]

    sensors_per_figure = 8
    sensors_per_side = sensors_per_figure // 2
    total_sensors = len(sensors_layout_names_df)

    # Generate the color palette based on the number of participants
    num_subjects = sensor_power_dataframes['MEG0111'].shape[0]
    colours = plt.cm.RdBu(np.linspace(0, 1, num_subjects))

    for start_idx in range(0, total_sensors, sensors_per_side):
        # Calculate the number of rows needed for the current figure
        current_sensors = min(sensors_per_side, total_sensors - start_idx)
        fig, axes = plt.subplots(2, current_sensors, figsize=(20, 10))  # 2 rows, N columns
        axes = axes.flatten()

        # Separate indexing for right and left sensors
        right_index = current_sensors  # Start right sensors in the middle of axes
        left_index = 0

        # Plotting each sensor's data in the correct subplot
        for idx in range(current_sensors):
            row = sensors_layout_names_df.iloc[start_idx + idx]
            sensor_right = row['right_sensors'][0:8]
            sensor_left = row['left_sensors'][0:8]

            if sensor_left in sensor_power_dataframes:
                print('Plotting left sensors on the top')
                df_left = sensor_power_dataframes[sensor_left]

                sns.boxplot(data=df_left.iloc[:, freq_mask[::step]],
                             ax=axes[left_index], color='lightblue')  # this will only display if scatter
                                                                      # is off or does not have palette parameter.

                # Scatter plot of individual data points with hue set to identify participants by color
                sns.scatterplot(
                    x=np.tile(filtered_freqs, df_left.shape[0]),  # Repeat frequencies for all participants
                    y=df_left.iloc[:, freq_mask[::step]].values.flatten(),  # Flatten the power values for all frequencies and participants
                    hue=df_left.index.repeat(len(filtered_freqs)),  # Repeat participant IDs for all frequencies
                    ax=axes[left_index],
                    s=20,
                    palette=colours,  # Use the custom color palette
                    legend=False
                )

                axes[left_index].set_title(sensor_left, fontsize=10)
                axes[left_index].set_xlim(filtered_freqs[0], filtered_freqs[-1])
                axes[left_index].set_xlabel('Frequency (Hz)')
                axes[left_index].set_ylabel('Power')
                left_index += 1

            if sensor_right in sensor_power_dataframes:
                print('Plotting right sensors on the bottom')
                df_right = sensor_power_dataframes[sensor_right]

                sns.boxplot(data=df_right.iloc[:, freq_mask[::step]], 
                            ax=axes[right_index], color='orange')  # this will only display if scatter
                                                                   # is off or does not have palette parameter.

                # Scatter plot of individual data points with hue set to identify participants by color
                if current_sensors == total_sensors - 1:  # show legend only on last plot
                    sns.scatterplot(
                        x=np.tile(filtered_freqs, df_right.shape[0]),  # Repeat frequencies for all participants
                        y=df_right.iloc[:, freq_mask[::step]].values.flatten(),  # Flatten the power values for all frequencies and participants
                        hue=df_right.index.repeat(len(filtered_freqs)),  # Repeat participant IDs for all frequencies
                        ax=axes[right_index],
                        s=20,
                        palette=colours,  # Use the custom color palette
                        legend=True
                    )
                else:
                    sns.scatterplot(
                        x=np.tile(filtered_freqs, df_right.shape[0]),  # Repeat frequencies for all participants
                        y=df_right.iloc[:, freq_mask[::step]].values.flatten(),  # Flatten the power values for all frequencies and participants
                        hue=df_right.index.repeat(len(filtered_freqs)),  # Repeat participant IDs for all frequencies
                        ax=axes[right_index],
                        s=20,
                        palette=colours,  # Use the custom color palette
                        legend=False
                    )                    

                axes[right_index].set_title(sensor_right, fontsize=10)
                axes[right_index].set_xlim(filtered_freqs[0], filtered_freqs[-1])
                axes[right_index].set_xlabel('Frequency (Hz)')
                axes[right_index].set_ylabel('Power')
                right_index += 1

        plt.tight_layout()

        if output_dir:
            output_fname = op.join(output_dir, f'sensor_power_boxplots_{start_idx//sensors_per_side + 1}_60-120.png')
            plt.savefig(output_fname, dpi=300)
            plt.close()
        else:
            plt.show()


def plot_power_average(sensor_power_dataframes, freqs, min_freq=60, max_freq=120, test_plot_dir=None, plot_histogram=True, plot_filtered_histogram=True, plot_boxplot=True, plot_pooled_histogram=True):
    """Calculate and plot the average power between specified frequencies for each participant and each sensor.
    
    Args:
        sensor_power_dataframes (dict): Dictionary containing DataFrames of power data for each sensor.
        freqs (array): Array of frequency values.
        min_freq (int): Minimum frequency for power calculation. Default is 60 Hz.
        max_freq (int): Maximum frequency for power calculation. Default is 120 Hz.
        test_plot_dir (str): Directory for saving plots. Default is 'None'.
        plot_histogram (bool): Whether to plot the full histogram. Default is True.
        plot_filtered_histogram (bool): Whether to plot the filtered histogram (outside IQR). Default is True.
        plot_boxplot (bool): Whether to plot the boxplot. Default is True.
        plot_pooled_histogram (bool): Whether to plot the pooled histogram. Default is True.
    """

    freq_mask = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
    
    # Initialize arrays for storing power averages for pooled hist
    num_grad_sensors = sum(1 for sensor_name in sensor_power_dataframes if not sensor_name.endswith('1'))
    num_mag_sensors = sum(1 for sensor_name in sensor_power_dataframes if sensor_name.endswith('1'))
    num_subjects = sensor_power_dataframes['MEG0111'].shape[0]

    power_avg_mag_array = np.empty((num_subjects, num_mag_sensors))
    power_avg_grad_array = np.empty((num_subjects, num_grad_sensors))

    mag_sensor_idx = 0
    grad_sensor_idx = 0

    # Initialise dictionaries for hist
    power_avg_mag_dic = {}
    power_avg_grad_dic = {}


    for sensor_name in sensor_power_dataframes:
        df = sensor_power_dataframes[sensor_name]
        power_avg = df.iloc[:, freq_mask].mean(axis=1)  # Average power for each participant
        
        if sensor_name.endswith('1'):
            # Add power averaged in a dictionary to later separate sensors from subjects
            power_avg_mag_dic[sensor_name] = power_avg
            # Add the power averages to the corresponding column in the magnetometer array to pool sub and sensors together
            power_avg_mag_array[:, mag_sensor_idx] = power_avg
            mag_sensor_idx += 1  # Move to the next column
        else:
            power_avg_grad_dic[sensor_name] = power_avg
            power_avg_grad_array[:, grad_sensor_idx] = power_avg
            grad_sensor_idx += 1

    sub_sens_power_avg_mag_array = power_avg_mag_array.reshape(num_mag_sensors*num_subjects, -1)
    sub_sens_power_avg_grad_array = power_avg_grad_array.reshape(num_grad_sensors*num_subjects, -1)

    # Plot Magnetometers
    output_dir_mag = op.join(test_plot_dir, f'mag_{min_freq}-{max_freq}')
    plot_avg_power_distribution(power_avg_mag_dic, sub_sens_power_avg_mag_array, output_dir=output_dir_mag, title=f"Magnetometers_{min_freq}-{max_freq}", plot_histogram=plot_histogram, plot_filtered_histogram=plot_filtered_histogram, plot_boxplot=plot_boxplot, plot_pooled_histogram=plot_pooled_histogram)

    # Plot Gradiometers
    output_dir_grad = op.join(test_plot_dir, f'grad_{min_freq}-{max_freq}')
    plot_avg_power_distribution(power_avg_grad_dic, sub_sens_power_avg_grad_array, output_dir=output_dir_grad, title=f"Gradiometers_{min_freq}-{max_freq}", plot_histogram=plot_histogram, plot_filtered_histogram=plot_filtered_histogram, plot_boxplot=plot_boxplot, plot_pooled_histogram=plot_pooled_histogram)


def plot_avg_power_distribution(power_avg_dict, power_avg_array, output_dir, title, bin_count=10, q1=0, q2=0.9, plot_histogram=True, plot_filtered_histogram=True, plot_boxplot=True, plot_pooled_histogram=True):
    """Plot various distributions of average power values.

    Args:
        power_avg_dict (dict): Dictionary with sensor names as keys and power averages as values.
        power_avg_array (array): Array of all sensors and subjects power values pooled together.
        output_dir (str): Directory for saving plots.
        title (str): Title of the plot.
        bin_count (int): Number of bins to use for histograms. Default is 10.
        q1 (float between 0 and 1): first quantile for inter quantile range.
        q2 (float between 0 and 1): last quantile for inter quantile range.
        plot_histogram (bool): Whether to plot the full histogram. Default is True.
        plot_filtered_histogram (bool): Whether to plot the filtered histogram (outside IQR). Default is True.
        plot_boxplot (bool): Whether to plot the boxplot. Default is True.
        plot_pooled_histogram (bool): Whether to plot the pooled histogram. Default is True.
    """

    # Prepare data for plotting
    all_sensor_names = []
    all_avg_powers = []
    
    for sensor_name, power_avgs in power_avg_dict.items():
        all_sensor_names.extend([sensor_name] * len(power_avgs))
        all_avg_powers.extend(power_avgs)
    
    plot_data = pd.DataFrame({'Sensor': all_sensor_names, 'Power Avg': all_avg_powers})
    plot_data.dropna(subset=['Power Avg'], inplace=True)

    if plot_data.empty:
        print("No data available for plotting after removing NaN values.")
        return

    # Full Histogram Plot
    if plot_histogram:
        plot_histogram_with_bins(plot_data, output_dir, title, bin_count)

    # Filtered Histogram (Outside IQR)
    if plot_filtered_histogram:
        plot_filtered_histogram_with_bins(plot_data, output_dir, title, bin_count, q1, q2)

    # Boxplot of Sensor Power Averages
    if plot_boxplot:
        plot_boxplot_ranked_by_mean(plot_data, output_dir, title)

    # Pooled Histogram of Sensors and Subjects
    if plot_pooled_histogram:
        plot_pooled_histogram_of_sensors(power_avg_array, output_dir, title, bin_count)
    
def plot_histogram_with_bins(plot_data, output_dir, title, bin_count):
    """Plot a histogram of power averages."""
    min_power, max_power = plot_data['Power Avg'].min(), plot_data['Power Avg'].max()
    bins = np.linspace(min_power, max_power, bin_count + 1)
    
    plt.figure(figsize=(12, 8))
    sns.histplot(data=plot_data, x='Power Avg', hue='Sensor', bins=bins, multiple='dodge', shrink=0.8)
    plt.title(f'Power Averages for {title}')
    plt.xlabel('Power Average')
    plt.ylabel('Number of Participants')
    plt.xticks(bins)
    plt.legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}_{bin_count}bin.png', dpi=300)
    plt.close()


def plot_filtered_histogram_with_bins(plot_data, output_dir, title, bin_count, q1, q2):
    """Plot a histogram of power averages outside the IQR."""
    Q1, Q3 = plot_data['Power Avg'].quantile([q1, q2])
    filtered_plot_data = plot_data[(plot_data['Power Avg'] > Q3) | (plot_data['Power Avg'] < Q1)]

    min_power, max_power = filtered_plot_data['Power Avg'].min(), filtered_plot_data['Power Avg'].max()
    bins = np.linspace(min_power, max_power, bin_count + 1)
    
    plt.figure(figsize=(12, 8))
    sns.histplot(data=filtered_plot_data, x='Power Avg', hue='Sensor', bins=bins, multiple='dodge', shrink=0.8)
    plt.title(f'Power Averages Between {q1}th and {q2}th Percentile for {title}')
    plt.xlabel('Power Average')
    plt.ylabel('Number of Participants')
    plt.xticks(bins)
    plt.legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}_{bin_count}bin_filtered90th.png', dpi=300)
    plt.close()

def plot_boxplot_ranked_by_mean(plot_data, output_dir, title):
    """Plot a boxplot of power averages, ranked by sensor mean power."""
    sensor_mean_powers = plot_data.groupby('Sensor')['Power Avg'].mean().sort_values()
    sorted_sensors = sensor_mean_powers.index
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Sensor', y='Power Avg', data=plot_data, order=sorted_sensors, palette='coolwarm')
    plt.title(f'{title} - Box Plot of Power Averages Ranked by Mean Power')
    plt.xlabel('Sensors (Ranked)')
    plt.ylabel('Power Average')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}_box.png', dpi=300)
    plt.close()

def plot_pooled_histogram_of_sensors(power_avg_array, output_dir, title, bin_count):
    """Plot a histogram of pooled sensors and subjects.
        the aim of this plot is to find the healthy range of power across 
        all sensors and all subjects."""
    plt.figure(figsize=(12, 8))
    plt.hist(power_avg_array, bins=bin_count)
    plt.title(f'Histogram of Sensors and Subjects vs Power - {title}')
    plt.xlabel('Power')
    plt.ylabel('Number of Sensors + Subjects')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}_pooled.png', dpi=300)
    plt.close()


# Define paths (same as your original script)
platform = 'mac'  # 'bluebear' or 'mac'?

if platform == 'bluebear':
    rds_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
elif platform == 'mac':
    rds_dir = '/Volumes/quinna-camcan'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'

epoched_dir = op.join(rds_dir, 'derivatives/meg/sensor/epoched-7min50')
info_dir = op.join(rds_dir, 'dataman/data_information')
good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
sensors_layout_sheet = op.join(info_dir, 'sensors_layout_names.csv')
output_dir = op.join(rds_dir, 'derivatives/meg/sensor/power_per_sensor')
test_plot_dir = op.join(jenseno_dir, 'Projects/subcortical-structures/resting-state/results/CamCan/Results/test_plots/healthy_distribution')

# Load subject information and sensor layout
good_subject_pd = pd.read_csv(good_sub_sheet).set_index('Unnamed: 0')
sensors_layout_names_df = pd.read_csv(sensors_layout_sheet)

# Calculate spectral power, store and then plot
sensor_power_dataframes, freqs = calculate_and_store_spectral_power(good_subject_pd, sensors_layout_names_df, epoched_dir, output_dir)
# plot_sensor_power(sensor_power_dataframes, sensors_layout_names_df, freqs, min_freq=60, max_freq=120, output_dir=None)
plot_power_average(sensor_power_dataframes, freqs, min_freq=0, max_freq=120, test_plot_dir=test_plot_dir, plot_histogram=True, plot_filtered_histogram=True, plot_boxplot=True, plot_pooled_histogram=True)
