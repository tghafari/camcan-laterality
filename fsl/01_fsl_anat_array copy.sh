#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 90
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 40G RAM and will ensure faster conversion to the .sif format
#SBATCH --array=609-619

module purge
module load bluebear
module load FSL/6.0.5.1-foss-2021a-fslpython

set -e

# Define the location of the file
export base_dir="/rds/projects/q/quinna-camcan"
preproc_dir="${base_dir}/cc700/mri/pipeline/release004/BIDS_20190411/anat"
info_dir="${base_dir}/dataman/data_information"
good_sub_sheet="${info_dir}/demographics_goodPreproc_subjects.csv"
output_dir="${base_dir}/derivatives/mri/subStr_segmented"

# Read good subject IDs
subjectID=$(cat $good_sub_sheet | tail -n +2 | cut -d',' -f1 | sed -n "${SLURM_ARRAY_TASK_ID}p")

base_name="sub-CC${subjectID}/anat"
T1w_name="sub-CC${subjectID}_T1w.nii.gz"
T1w_fpath="${preproc_dir}/${base_name}/${T1w_name}"
output_fpath="${output_dir}/sub-CC${subjectID}"

# Run on container
# apptainer exec FSL.sif fsl_anat -i $T1w_fpath -o $output_fpath --clobber

# Run with FSL module from bluebear
fsl_anat -i $T1w_fpath -o $output_fpath --clobber
