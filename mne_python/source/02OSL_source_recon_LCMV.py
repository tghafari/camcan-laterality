"""
============================================
02OSL_source_recon_LCMV

this code takes the fwd model calcaulted in
01_coregistraion_rhino to localise the
oscillatory power modulation in resting state.
We are taking these steps:
    1. Compute surfaces, perform coregistration, 
    and compute forward model using batching 
    (only for subjects that don't have
    coregistered rhino files)
    2. Temporal Filtering
    3. Compute beamformer weights
    4. Apply beamformer weights
    5. Parcellation


written by Tara Ghafari
adapted from OSL pipeline
    tutorials in source recon
    from OSL course (https://osf.io/w4pjg)
==========================================
"""

import os.path as op
import os
import numpy as np
import osl
import mne
from mne.cov import compute_covariance, compute_raw_covariance
from mne.beamformer import make_lcmv, apply_lcmv_cov
import matplotlib.pyplot as plt

# Define the paths
rds_dir = r'/rds/projects/q/quinna-camcan'
sMRI_anat_dir = r'cc700/mri/pipeline/release004/BIDS_20190411/anat'
preproc_dir = r'camcan_bigglm/processed-data/CamCAN_firstlevel'
deriv_dir = op.join(rds_dir, r'derivatives/meg')
epo_dir = op.join(deriv_dir, 'sensor', 'epoched_data')


subjectID = '110037'

sub_name = ('sub-CC' + str(subjectID))
sMRI_fpath = op.join('sub-CC' + str(subjectID), 'anat')
sMRI_dir = op.join(rds_dir, sMRI_anat_dir, sMRI_fpath)
source_fpath = op.join(deriv_dir, 'source') 
fwd_model_fname = op.join(deriv_dir, source_fpath, 
        'sub-CC' + str(subjectID), 'rhino', 'model-fwd.fif')
epo_fname = op.join(epo_dir, 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif')

# Define signed flipped parcellated (output) file
#output_fpath = os.mkdirs(op.join(deriv_dir, 'source', 'sub-CC' + str(subjectID),
#                'parcellation'), exist_ok=True)

import pathlib
fpath = op.join(deriv_dir, 'source', 'sub-CC' + str(subjectID), 'parcellation')
pathlib.Path(fpath).mkdir(parents=True, exist_ok=True)

output_fname = (op.join(deriv_dir, 'source', 'sub-CC' + str(subjectID),
                 'parcellation', 'sign_flip_parc.npy'))

# 1. Compute surfaces, coregister, and forward model (in batch)
# was not needed for this subject as the coregistration had been
# done for this single subject before. develop this part when 
# wanting to run in batch


# 2. Temporal filtering
"""
We temporally filter the data to focus on 
the oscillatory content that we are interest in. 
"""

# Read epochs/raw data
data = mne.read_epochs(epo_fname, preload=True)
chantypes = ['grad']
data = data.pick(chantypes)

# Filter to the band(s) of interest
print("Temporal Filtering")
data = data.filter(
    l_freq=3,
    h_freq=20,
    method="iir",
    iir_params={"order": 5, "btype": "bandpass", "ftype": "butter"},
)
print("Completed")

# 3. Compute beamformer weights
""" 
We now compute the beamformer weights (aka filters). 
These are computed using the (sensors x sensors) data
covariance matrix estimated from the preprocessed and 
the temporally filtered MEG data (contained in *raw*), 
and the forward models (contained inside the directory *recon_dir*.) 

Note that this automatically ignores any bad time 
segments when calculating the beamformer filters.

Here we source reconstructing using just the gradiometers.

If the MEG data has been maxfiltered the maximum rank is ~64. 
We therefore slightly conservatively set the rank to be 55. 
This is used to regularise the estimate of the data covariance matrix.

More generally, a dipole is a 3D vector in space. Setting 
*pick_ori="max-power-pre-weight-norm"* means that we are 
computing a scalar beamformer, by projecting this 3D vector 
on the direction in which there is maximum power. 
"""

# Make LCMV beamformer filters
filters = osl.source_recon.beamforming.make_lcmv(
            source_fpath,
            sub_name,
            data,
            chantypes,
            pick_ori="max-power-pre-weight-norm",
            rank={"grad": 55},
)

# 4. Applying beamformer weights
"""
We now apply the beamformer filters to the data 
to project the data into source space.

Note that although the beamformer filters were 
calculated by ignoring any bad time segments, 
we apply the filters to all time points including 
the bad time segments. This will make it easier 
to do epoching later.
"""

print("Applying beamformer spatial filters")

# stc is source space time series (in head/polhemus space).
stc = osl.source_recon.beamforming.apply_lcmv(data, filters)

parcellation_fname = 'HarvOxf-sub-Schaefer100-combined-2mm_4d.nii.gz'
mask_file = "MNI152_T1_2mm_brain.nii.gz"

# Convert from head/polhemus space to standard brain grid in MNI space
stc_mni = []
parcels = []
for blah in stc[:5]:
    stc_mni.append(osl.source_recon.beamforming.transform_recon_timeseries(source_fpath, 
                                                                           sub_name, 
                                                                           recon_timeseries=blah.data, 
                                                                           reference_brain="mni"))

    recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = stc_mni[-1]

    parcel_ts, _, _ = osl.source_recon.parcellation.parcellate_timeseries(
    parcellation_fname, 
    recon_timeseries_mni, 
    recon_coords_mni, 
    "spatial_basis", 
    source_fpath,
    )

    parcels.append(parcel_ts[None, :, :])

parcel_ts_all = np.concatenate(parcels, axis=0)


nii = parcellation.convert2niftii(parcel_ts_all.mean(axis=0), parcellation.find_file(parcellation_file), parcellation.find_file(mask_file))
nii.to_filename('testing123.nii')

print("Completed")
print("Dimensions of reconstructed timeseries in \
    MNI space is (dipoles x all_tpts) = {}".format(recon_timeseries_mni.shape))

# Error:
"""
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-28-0eec0e3d7bb8> in <module>
      2     stc = osl.source_recon.beamforming.transform_recon_timeseries(source_fpath, 
      3                                                                 sub_name,
----> 4                                                                 recon_timeseries=stc.data,
      5                                                                 reference_brain="mni")

AttributeError: 'list' object has no attribute 'data'
--------------------------------------------------------------------------
"""


# 5. Parcellation
"""
At this point, the data have been source reconstructed 
to dipoles (in this case, a scalar value) at each point 
on a regular 3D grid, with spacings of 10mm. We can 
then analyse the data across all these dipoles.

An alternative, is to map the data onto a brain parcellation.
This reduces the number of samples in the space from 
number of dipoles down to number of parcels. Using a 
parcellation helps to boost the signal to noise ratio, boost 
correspondance between subjects, reduce the severity of 
multiple comparison correction when doing any statistics, 
and aids anatomical interpretability.

The parcellation we use here is a combination of cortical 
regions from the Harvard Oxford atlas, and selected sub-cortical 
regions from the Schaefer 100 parcellation. 

Let's take a look at the positions of the centres of each
parcel in the parcellation.
"""

parcellation_fname = 'dk_cortical.nii.gz'

# plot centre of mass for each parcel
p = osl.source_recon.parcellation.plot_parcellation(parcellation_fname)
plt.show()



P = osl.source_recon.parcellation.load_parcellation(parcellation_fname)
pcentres = osl.source_recon.parcellation.parcel_centers(parcellation_fname)
# To match parcels left-right - flip the sign of the first dim and find closest coord match for each row between orig pcentres and flipped pcentres


#

mask_file = "MNI152_T1_2mm_brain.nii.gz"
nii = osl.source_recon.parcellation.convert2niftii(parcels[0], parcellation_fname, mask_file)


# 6. Compute parcel time-courses
"""
We use this parcellation to compute the parcel time courses 
using the parcellation and the dipole time courses. 
Note that the output parcel timepoints includes all time points, 
including any bad time segments.

This is done using the "spatial_basis" method, where the 
parcel time-course first principal component from all voxels, 
weighted by the spatial map for the parcel 
(see https://pubmed.ncbi.nlm.nih.gov/25862259/).
"""
print("Parcellating data")

# Apply parcellation to (voxels x all_tpts) data contained in recon_timeseries_mni.
# The resulting parcel_timeseries will be (parcels x all_tpts) in MNI space
# where all_tpts includes bad time segments
parcel_ts, _, _ = osl.source_recon.parcellation.parcellate_timeseries(
    parcellation_fname, 
    recon_timeseries_mni, 
    recon_coords_mni, 
    "spatial_basis", 
    source_fpath,
)

print("Completed")
print("Dimensions of parcel timeseries in MNI space is (nparcels x all_tpts) = {}".format(parcel_ts.shape))



plt.figure()
plt.plot(parcel_ts.T + np.arange(114)[None, :]*5)