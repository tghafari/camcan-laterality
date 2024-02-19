#!/bin/bash
#SBATCH --account quinna-camcan
#SBATCH --qos bbdefault
#SBATCH --time 300
#SBATCH --nodes 1 # ensure the job runs on a single node
#SBATCH --ntasks 12 # this will give you circa 48G RAM and will ensure faster conversion to the .sif format

module purge
module load bluebear
module load bear-apps/2022a
module load MNE-Python/1.3.1-foss-2022a

python S01_spectrum_lateralisation.py