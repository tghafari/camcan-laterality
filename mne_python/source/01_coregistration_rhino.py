"""
===============================================
S01. Coregistaring with RHINO

This script coregisters the MEG file with MRI
using RHINO (Registration of Headshapes Including
Nose in OSL)

steps:
    1. Compute surfaces
    2. Coregistration
    3. Compute forward model
    4. Batched RHINO (combined surface extraction, 
    coregistration and forward modelling over multiple subjects)


note;
    To run this tutorial you will need to have OSL and FSL installed

written by Tara Ghafari
adapted from /SourceRecon/tutorials directory at
https://osf.io/zxb6c
==============================================
"""

import os.path as op
import pandas as pd
import numpy as np
import mne
import osl

# Set up preprocessed MEG and MRI data path
# Define where to read and write the data
rds_dir = r'/rds/projects/q/quinna-camcan'
sMRI_dir = r'cc700/mri/pipeline/release004/BIDS_20190411/anat'
preproc_dir = r'camcan_bigglm/processed-data/CamCAN_firstlevel'

output_dir = op.join(rds_dir, r'derivatives/meg/source')

# info_dir = r'dataman\Data_information'
# good_sub_sheet = op.join(info_dir, 'demographics_goodPreproc_subjects.csv')
# Read only data from subjects with good preprocessed data
# good_subject_pd = pd.read_csv(good_sub_sheet)
# good_subject_pd = good_subject_pd.set_index('Unnamed: 0')  # set subject id codes as the index

# sample for developing the code
subjectID = '110037'

sMRI_fpath = op.join('sub-CC' + str(subjectID), 'anat', 'sub-CC' + str(subjectID) + '_T1w.nii.gz')
sMRI_file = op.join(rds_dir, sMRI_dir, sMRI_fpath)

output_fpath = 'sub-CC' + str(subjectID)

base_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef'
preproc_name = 'mf2pt2_sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_preproc_raw.fif'
preproc_fif = op.join(rds_dir, preproc_dir, base_name, preproc_name)

# 1. Compute surfaces
""" 
inputs we need to provide are for the first subject are:
smri_file - the full path to the structural MRI niftii file
output_dir - the full path to the directory that will contain the subject directories that RHINO will output
output_fname - the name of the subject directory that RHINO will output
include_nose - a boolean flag indicating whether or not to extract a head surface from the structural MRI that
 includes the nose. It your structural MRI includes the nose AND you have acquired polhemus headshape points 
 that include the nose, then it is recommend to set this flag to True

"""

print('computing surfaces for subject #', subjectID)

osl.source_recon.rhino.compute_surfaces(
    sMRI_file,
    output_dir,
    output_fpath,
    include_nose=True,
)

# View the results using fsleyes-- discuss with Andrew?
osl.source_recon.rhino.surfaces_display(output_dir, output_fpath)


#%%

filenames = osl.source_recon.rhino.surfaces.get_surfaces_filenames(output_dir, output_fpath)

print("fsleyes {} {} {} {} {} &".format(
        filenames["smri_file"],
        filenames["bet_inskull_mesh_file"],
        filenames["bet_outskin_mesh_file"],
        filenames["bet_outskull_mesh_file"],
        filenames["bet_outskin_plus_nose_mesh_file"],
    )
)


# 2. Coregistration
"""
We can now perform coregistration so that the 
MEG sensors and head / brain surfaces can be placed 
into a common coordinate system.

We do this by running rhino.coreg and passing in:

- fif_file the full path to the MNE raw fif file.
- output_dir - the full path to the directory that contains
     the subject directories RHINO outputs
- output_fname - the name of the subject directories RHINO outputs to
- use_headshape - a boolean flag indicating whether or not to use
     the headshape points to refine the coregistration.
- use_nose - a boolean flag indicating whether or not to use the
     nose headshape points to refine the coregistration. Setting this
     to True requires that include_nose was set True in the call to 
     rhino.compute_surfaces, and requires that the polhemus headshape 
     points include the nose.
"""

# cd to the containing folder or search how to import from outside of the folder.
import polhemus

polhemus.extract_polhemus_from_info(
    preproc_fif,
    'polhemus_headshape.txt',
    'polhemus_nasion.txt',
    'polhemus_rpa.txt',
    'polhemus_lpa.txt',
    include_eeg_as_headshape=False,
    include_hpi_as_headshape=True,
    )

# print('plotting polhemus points') -- this snippet needs to be discussed with Andrew, the scatter plot isn't reading the points.

#coreg_dir = r'/rds/projects/q/quinna-camcan/derivatives/meg/source/sub-CC110037/rhino/coreg'
#txt_fname = op.join(coreg_dir, 'polhemus_nasion.txt')

#with open(txt_fname) as txt:
#    line = txt.readlines()

#polhemus.plot_polhemus_points(line, colors=None, 
#                scales=None, markers=None, alphas=None)

print('coregistaring subject #', subjectID)

osl.source_recon.rhino.coreg(
    preproc_fif,
    output_dir,
    output_fpath,
    use_headshape=True,    
    use_nose=True,
)

# View the results of coregistration -- discus with Andrew
""" The coregistration result is shown in MEG (device) space (in mm).

Grey disks - MEG sensors
Blue arrows - MEG sensor orientations
Yellow diamonds - MRI-derived fiducial locations
Pink spheres - Polhemus-derived fiducial locations
Green surface - Whole head scalp extraction
Red spheres - Polhemus-derived headshape points

A good coregistration shows:
MRI fiducials (yellow diamonds) in appropriate positions on the scalp
Polhemus-derived fiducial locations (pink spheres) in appropriate positions on the scalp
Good correspondence between the headshape points (red spheres) and the scalp
The scalp appropriately inside the sensors, and with a sensible orientation.

If you have a bad co-registration:
Go back and check that the compute_surfaces has worked well using fsleyes (see above).
Check for misleading or erroneous headshape points (red spheres) and remove them.
Check that the settings for using the nose are compatible with the available MRI and headshape points
The subject in question may need to be omitted from the ensuing analysis.
"""

osl.source_recon.rhino.coreg_display(
        output_dir,
        output_fpath,
        display_outskin_with_nose=False,
    )


# 3. Compute Forward Model
"""
Here we are modelling the brain/head using 'Single Layer', 
which corresponds to just modelling the inner skull surface, 
which is the standard thing to do in MEG forward modelling.

Lead fields will be computed for a regularly space dipole 
grid, with a spacing given by the passed in argument gridstep.
The dipole grid is confined to be inside the brain mask as 
computed by rhino.compute_surfaces.

The mdl-fwd.fif inside rhino contains the leadfields that map from 
source to sensor space, and which are used to do source reconstruction.
"""

gridstep = 10
osl.source_recon.rhino.forward_model(
    output_dir,
    output_fpath,
    model="Single Layer",
    gridstep=gridstep,
)

# View the results - same error as pyvistaqt (coreg_display)
"""
that the small black points inside the brain show the locations 
of the dipoles that the leadfields have been computed for.
"""
osl.source_recon.rhino.bem_display(
    output_dir,
    output_fpath,
    display_outskin_with_nose=False,
    display_sensors=True,
    plot_type="surf"
)


















