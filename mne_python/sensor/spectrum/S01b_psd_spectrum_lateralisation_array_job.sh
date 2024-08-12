#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 1000
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 30 # number of cores
#SBATCH --array=100

module purge
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a

# Path to the Python script
python_script="S01b_psd_spectrum_nolog_lateralisation_per_sub.py"

# Execute the Python script with the array index as an argument
for index in $(seq $SLURM_ARRAY_TASK_MIN $SLURM_ARRAY_TASK_MAX); do
    python $python_script $index
done
