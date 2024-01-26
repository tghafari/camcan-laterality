#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 150
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 20G RAM and will ensure faster conversion to the .sif format

module purge
module load bluebear
# module load FSL/6.0.5.1-foss-2021a

set -e

# Define the location of the file
export base_dir="/rds/projects/q/quinna-camcan"
preproc_dir="${base_dir}/cc700/mri/pipeline/release004/BIDS_20190411/anat"
output_dir="${base_dir}/cc700/mri/pipeline/release004/BIDS_20190411/derivatives/subStr_segmented"

for subjectID in 110033 110037 110045 110056; do

	base_name="sub-CC${subjectID}/anat"
	T1W_name="sub-CC${subjectID}_T1w.nii.gz"
	T1W_fpath="${preproc_dir}/${base_name}/${T1W_name}"
	output_fpath="${output_dir}/sub-CC${subjectID}"
	apptainer exec FSL.sif fsl_anat -i $T1W_fpath -o $output_fpath --clobber
done
