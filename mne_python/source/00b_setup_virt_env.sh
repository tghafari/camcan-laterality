#!/bin/bash

module purge; module load bluebear
module load bear-apps/2021a/live
## FSL and deps
module load bc/1.07.1-GCC-10.3.0
module load FSL/6.0.5.1-foss-2021a-fsl
# Python
module load Python/3.9.5-GCCcore-10.3.0
module load IPython/7.25.0-GCCcore-10.3.0
module load Qt5/5.15.2-GCCcore-10.3.0

# Interactive matplotlib ->
module load Tkinter/3.9.5-GCCcore-10.3.0


# Virtual Environment
export VENV_DIR="${HOME}/virtual-environments"
export VENV_PATH="${VENV_DIR}/osl-fsl-${BB_CPU}"

# Create master dir if necessary
mkdir -p ${VENV_DIR}
echo ${VENV_PATH}

# Check if virtual environment exists and create it if not
if [[ ! -d ${VENV_PATH} ]]; then
      python3 -m venv --system-site-packages ${VENV_PATH}
fi

# Activate virtual environment
source ${VENV_PATH}/bin/activate

# Setup FSL dir
FSLDIR=/rds/bear-apps/2021a/EL8-cas/software/FSL/6.0.5.1-foss-2021a-fslpython/fsl

# Custom installs within personal virtual environment - doesn't depend on BEAR
pip install pyvistaqt  # for displayig coregistration -- has to be before pip install numpy (for version)
pip install osl
pip install --upgrade numpy==1.22.4
pip install fslpy==3.11.3



# cd to the directory and run this code typing 
# "source ./00_setup_virt_env.sh" in bluebear terminal
# start ipython --pylab=tk
# then start importing libraries and code