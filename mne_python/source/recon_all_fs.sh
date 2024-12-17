#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --time 12:00:00
#SBATCH --qos bbdefault
#SBATCH --mem 32G

set -e

module purge; 
module load bluebear
module load bear-apps/2022a
module load GCC/11.3.0
module load VTK/9.2.2-foss-2022a
module load MNE-Python/1.3.1-foss-2022a
module load FreeSurfer/7.4.1-centos8_x86_64


# Define the SUBJECTS_DIR where recon-all will save output
export SUBJECTS_DIR=/rds/projects/q/quinna-camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat  # FreeSurfer subject output path
export FS_LICENSE=${HOME}/freesurfer_license.txt  # FreeSurfer license path

# Define the subject directory where the input MRI file is located
subject_id="sub-CC210023"
subject_dir="/rds/projects/q/quinna-camcan/cc700/mri/pipeline/release004/BIDS_20190411/anat/${subject_id}/anat"

# Define the subject ID and the file name (assumed to be in subject_dir)
input_file="${subject_dir}/${subject_id}_T1w.nii.gz"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "ERROR: cannot find ${input_file}"
    exit 1
fi

# Run recon-all with the correct file and subject ID
recon-all -i "$input_file" -s "$subject_id" -all -notal-check

# Run MNE-Python scalp surface and watershed BEM steps
mne watershed_bem --overwrite --subject "$subject_id"
mne make_scalp_surfaces --overwrite --subject "$subject_id" --force

