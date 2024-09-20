#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 12:00:00
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --mem 32G
#SBATCH --array=10-20  # Adjust this based on the number of subjects

set -e

module purge
module load bluebear
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a
module load FreeSurfer/7.4.1-centos8_x86_64

# Define the SUBJECTS_DIR where recon-all will save output
export SUBJECTS_DIR=/rds/projects/q/quinna-camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat  # FreeSurfer subject output path
export FS_LICENSE=${HOME}/freesurfer_license.txt  # FreeSurfer license path

# Define the location of the file
export base_dir="/rds/projects/q/quinna-camcan"
info_dir="${base_dir}/dataman/data_information"
good_sub_sheet="${info_dir}/demographics_goodPreproc_subjects.csv"

# Read good subject IDs from the CSV, selecting the one based on the array job index
subjectID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$good_sub_sheet" | cut -d',' -f1)  # more efficient than the on in fsl_anat

# Define the subject directory where the input MRI file is located
subject_dir="${SUBJECTS_DIR}/sub-CC${subjectID}/anat"

# Define the subject ID and the file name (assumed to be in subject_dir)
subject_id="sub-CC${subjectID}_T1w"
input_file="${subject_dir}/${subject_id}.nii.gz"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "ERROR: cannot find ${input_file}"
    exit 1
fi

# Run recon-all with the correct file and subject ID
recon-all -i "$input_file" -s "$subject_id" -all -notal-check

# Run MNE-Python scalp surface and watershed BEM steps
#mne make_scalp_surfaces --overwrite --subject "$subject_id" --force
mne watershed_bem --overwrite --subject "$subject_id"
