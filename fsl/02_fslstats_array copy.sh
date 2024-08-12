#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 90
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 5 # this will give you circa 40G RAM and will ensure faster conversion to the .sif format
#SBATCH --array=151-619

module purge
module load bluebear
module load FSL/6.0.5.1-foss-2021a-fslpython

set -e

# Define the location of the file
export base_dir="/rds/projects/q/quinna-camcan"
mri_deriv_dir="${base_dir}/derivatives/mri/subStr_segmented"
info_dir="${base_dir}/dataman/data_information"
good_sub_sheet="${info_dir}/demographics_goodPreproc_subjects.csv"

# Define variables for FSL command
labels=(10 11 12 13 16 17 18 26 49 50 51 52 53 54 58)
structures=("L-Thal" "L-Caud" "L-Puta" "L-Pall" "BrStem /4th Ventricle" \
    "L-Hipp" "L-Amyg" "L-Accu" "R-Thal" "R-Caud" "R-Puta" \
    "R-Pall" "R-Hipp" "R-Amyg" "R-Accu")

# Read good subject IDs
subjectID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$good_sub_sheet" | cut -d',' -f1)  # more efficient than the on in fsl_anat

subject_mri_dir="${mri_deriv_dir}/sub-CC${subjectID}.anat/first_results"

if [ -d "$subject_mri_dir" ]; then 
    mkdir -p "${mri_deriv_dir}/sub-CC${subjectID}.SubVol"
    
    echo "${subjectID}.anat/first_results was found"

    for low in ${labels[@]}; do
        low_minus_point_five=$(echo "$low - 0.5" | bc)
        low_plus_point_five=$(echo "$low + 0.5" | bc)

        VoxVol=$(fslstats "${subject_mri_dir}/T1_first_all_fast_firstseg.nii.gz" -l "$low_minus_point_five" -u "$low_plus_point_five" -V)
        echo "Volumetring: sub-CC${subjectID} structure: ${low}"
        
	output_fname="${mri_deriv_dir}/sub-CC${subjectID}.SubVol/volume${low}"
	echo "$VoxVol" > "$output_fname"            
    done

    echo "Volumetring ${subjectID} done"
else
    echo "no "$subject_mri_dir" found, continuing to the next sub"
fi
