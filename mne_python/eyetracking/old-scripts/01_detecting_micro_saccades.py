# -*- coding: utf-8 -*-
"""
===============================================
01_detecting_micro_saccades

this code gets the asc file from the eyetracker 
and detects microsaccades.
edf files (out put of eyelink) should be
converted to .asc using 'visualEDF2ASC' app

written by Tara Ghafari
adapted from:
https://github.com/titoghose/PyTrack/blob/master/getting_started_SAMode.ipynb  
==============================================
ToDos:
"""

from PyTrack.Stimulus import Stimulus
from PyTrack.formatBridge import generateCompatibleFormat
import pandas as pd
import numpy as np
import os

# Define file names
sub_code = '103'
base_dir = r'Z:\Projects\Subcortical Structures\SubStr and behavioral bias\Data'
task_dir = r'Target Orientation Detection\Results'
sub_dir = os.path.join('sub-S' + sub_code, r'ses-01\beh')
eyetracking_fname = 'e01S' + sub_code 
eyetracking_fpath = os.path.join(base_dir, task_dir, sub_dir, eyetracking_fname + '.asc') 
eyetracking_compatibel_format = os.path.join(base_dir, task_dir, sub_dir, eyetracking_fname + '.csv') 

# function to convert data to generate csv file for data file recorded using 
# EyeLink on both eyes and the stimulus name specified in the message section
if not os.path.isfile(os.path.abspath(eyetracking_compatibel_format)):
    generateCompatibleFormat(exp_path=os.path.abspath(eyetracking_fpath),
                            device="eyelink",
                            stim_list_mode="NA",
                            start="START",  # case sensitive, check .asc file
                            stop="END",  # case sensitive, check .asc file
                            eye="B")

df = pd.read_csv(os.path.abspath(eyetracking_compatibel_format))

print('Max gaze (height): {:.2f} - Left eye, {:.2f} - Right eye'.format(
    df['GazeLefty'].max(), df['GazeRighty'].max()))
print('Max gaze (width): {:.2f} - Left eye, {:.2f} - Right eye'.format(
    df['GazeLeftx'].max(), df['GazeRightx'].max()))

# Dictionary containing details of recording. Please change the values 
# according to your experiment. If no AOI is desired, set aoi value to 
# [0, 0, Display_width, Display_height]
sensor_dict = {
    "EyeTracker":
    {
        "Sampling_Freq": 500,
        "Display_width": 1920,
        "Display_height": 1080,
        "aoi": [0, 0, 1920, 1080]
    }
}

# Creating Stimulus object. See the documentation for advanced parameters.
stim = Stimulus(path=os.path.abspath("PyTrack_Sample_Data/EyeLink"),
               data=df,
               sensor_names=sensor_dict)

# Some functionality usage. See documentation of Stimulus class for advanced use.
stim.findEyeMetaData()

# Getting dictionary of found metadata/features
features = stim.sensors["EyeTracker"].metadata  
f = stim.sensors["EyeTracker"].metadata["response_time"]
saccade_indices = stim.findSaccades()

# Extracting features
MS, ms_count, ms_duration, ms_vel, ms_amp = stim.findMicrosaccades(plot_ms=True)

# Visualization of plots
stim.gazePlot(save_fig=True)
stim.gazeHeatMap(save_fig=True)
stim.visualize()