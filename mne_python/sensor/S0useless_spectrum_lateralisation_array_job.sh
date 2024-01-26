#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 300
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 12 # this will give you circa 48G RAM and will ensure faster conversion to the .sif format
#SBATCH --array=16-51

module purge
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a

# Path to the Python script
python_script="S01_spectrum_lateralisation.py"

# Execute the Python script with the array index as an argument
for index in $(seq $SLURM_ARRAY_TASK_MIN $SLURM_ARRAY_TASK_MAX); do
    python $python_script $index
done
