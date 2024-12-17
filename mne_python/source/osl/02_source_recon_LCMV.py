"""
02_source_recon_LCMV
this code takes the fwd model calcaulted in
01_coregistraion_rhino to localise the
oscillatory power modulation in resting state.


written by Tara Ghafari
adapted from Flux pipeline
"""

import os.path as op
import numpy as np
import mne
from mne.cov import compute_covariance, compute_raw_covariance
from mne.beamformer import make_lcmv, apply_lcmv_cov

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


sub_fwd_fpath = op.join('source', 'sub-CC' + str(subjectID), 'rhino')
fwd_model_fname = op.join(deriv_dir, sub_fwd_fpath, 'model-fwd.fif')
epo_fname = op.join(epo_dir, 'sub-CC' + str(subjectID) + '_ses-rest_task-rest_megtransdef_epo.fif')

# 1. Read the forward solution
fwd = mne.read_forward_solution(fwd_model_fname)

# Read the epochs and plot the power spectra in the sensors
epochs = mne.read_epochs(epo_fname)
epochs.compute_psd(fmin=1, fmax=125).plot()  # 125 is the highest we can show as it is 1/2 of the sampling frequency
epochs.plot_psd_topomap(bands={'Alpha (8-12 Hz)':(8, 12)},ch_type='grad', normalize=True)
epochs = epochs.pick_types(meg='grad')
epochs_alpha = epochs.filter(8, 12).copy()

# 2. Calculating the covariance matrices
rank = mne.compute_rank(epochs_alpha, tol=1e-6, tol_kind='relative')
common_cov = compute_covariance(epochs_alpha, method='empirical',
            rank=rank, n_jobs=4, verbose=True)

# Plot the rank for covariance matrix
common_cov.plot(epochs_alpha.info)

# 3. Estimating a noise covariance with empty room data
""" since CAMCan does not include empty room recording
we are using ad_hoc function from MNE to create a 
noise covariance.
"""
noise_cov = mne.make_ad_hoc_cov(epochs_alpha.info, verbose=True)

# 4. Derive the spatial filters and apply
filters = make_lcmv(epochs_alpha.info, fwd, common_cov, reg=0.05,
                    noise_cov=noise_cov, rank = rank, pick_ori='max-power',
                    reduce_rank = True, depth = 0, inversion = 'matrix',
                    weight_norm = 'unit-noise-gain'
                    ) 

print('Applying beamforming spatial filters')
source_time_courses = apply_lcmv_cov(common_cov, filters)  # in head/polhemus space

print('Plotting...')
source_time_courses.plot(src=fwd['src'], mode='stat_map', 
            subjects_dir=sMRI_dir, subject=sub_name
            )

# https://mne.tools/stable/auto_examples/inverse/compute_mne_inverse_volume.html


# OSL Source Recon Example



from osl.source_recon import rhino, beamforming, parcellation

sMRI2 = '/rds/projects/q/quinna-camcan/derivatives/meg/source'
# Make LCMV beamformer filters
# Note that this will exclude any bad time segments when calculating the beamformer filters
filters = beamforming.make_lcmv(
    sMRI2,
    sub_name,
    epochs_alpha,
    'grad',
    pick_ori="max-power-pre-weight-norm",
    rank={"grad": 55},
)


print("Applying beamformer spatial filters")

# stc is source space time series (in head/polhemus space).
stc = beamforming.apply_lcmv(epochs_alpha, filters)

# Convert from head/polhemus space to standard brain grid in MNI space
recon_timeseries_mni, reference_brain_fname, recon_coords_mni, _ = \
        beamforming.transform_recon_timeseries(sMRI2, 
                                                sub_name, 
                                                recon_timeseries=stc[0].data, 
                                                reference_brain="mni")

print("Completed")
print("Dimensions of reconstructed timeseries in MNI space is (dipoles x all_tpts) = {}".format(recon_timeseries_mni.shape))