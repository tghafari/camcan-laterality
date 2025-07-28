
import os
import os.path as op
import numpy as np
import mne
from mne.datasets import fetch_fsaverage


def setup_paths(platform='mac'):
    """Set up file paths for the given platform."""
    if platform == 'bluebear':
        rds_dir = '/rds/projects/q/quinna-camcan'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        rds_dir = '/Volumes/quinna-camcan'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    return {
        'rds_dir': rds_dir,
        'info_dir': op.join(rds_dir, 'dataman/data_information'),
        'fs_sub_dir': op.join(rds_dir, 'cc700/mri/pipeline/release004/BIDS_20190411/anat'),
        'meg_source_dir': op.join(sub2ctx_dir, 'derivatives/meg/source/freesurfer'),
        'meg_sensor_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/epoched-2sec'),
    }


def construct_paths(subject_id, paths):
    """Construct file paths for subject-specific files."""
    fs_sub = f'sub-CC{subject_id}_T1w'
    deriv_folder = op.join(paths['meg_source_dir'], fs_sub[:-4])

    return {
        'fs_sub': fs_sub,
        'deriv_folder': deriv_folder,
        'fwd_vol_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-vol.fif'),
        'fwd_surf_fname': op.join(deriv_folder, f'{fs_sub[:-4]}_fwd-surf.fif'),
        'epoched_epo_fname': op.join(paths['meg_sensor_dir'], f'{fs_sub[:-4]}_2sec_epod-epo.fif'),
    }

def compute_mne_source_estimate(subject_id, fwd, epo_fname, save_dir):
    """Compute MNE source estimate from epochs and forward model."""
    print(f"Processing subject {subject_id}...")
    epochs = mne.read_epochs(epo_fname, preload=True, verbose=True, proj=False)
    noise_cov = mne.compute_covariance(epochs, tmax=0.0, method='empirical')

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov, loose='auto', depth=0.8
    )

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inverse_operator, lambda2=1./9., method='MNE', pick_ori=None
    )

    stc_mean = sum(stcs) / len(stcs)
    # stc_mean.save(op.join(save_dir, f"{subject_id}_mne"), overwrite=True)
    # print(f"Saved MNE source estimate for subject {subject_id}.")

    return stc_mean


def morph_mne_stc_to_fsaverage(stc, subject_id, src, paths):
    """Morph source estimate to fsaverage (volumetric morphing)."""
    fetch_fsaverage(paths["fs_sub_dir"])
    src_fs_path = op.join(paths["fs_sub_dir"], "fsaverage", "bem", "fsaverage-vol-5-src.fif")
    src_fs = mne.read_source_spaces(src_fs_path)

    morph = mne.compute_source_morph(
        src=src,
        subject_from=src[0]['subject_his_id'],
        src_to=src_fs,
        subjects_dir=paths["fs_sub_dir"],
        niter_sdr=[40, 20, 10],
        niter_affine=[100, 100, 50],
        zooms='auto',
        smooth=5,
        verbose=True,
    )
    stc_fs = morph.apply(stc)
    return stc_fs, src_fs

def plot_source_band_power(subject_id, band, stc, paths, src_fs):
    """Plot and save band-specific source estimate."""
    assert band in ['Delta', 'Theta', 'Alpha', 'Beta'], f"Unsupported band: {band}"
    initial_pos = np.array([19, -50, 49]) * 0.001

    stc.plot(
        src=src_fs,
        mode="stat_map",
        subjects_dir=paths["fs_sub_dir"],
        # initial_pos=initial_pos,
        verbose=True,
    )


def main():
    platform = 'mac'
    subject_id = '310410'
    band = 'Alpha'

    # === SETUP ===
    paths = setup_paths(platform)
    file_paths = construct_paths(subject_id, paths)
    space = 'volume'

    fwd_fname = file_paths['fwd_vol_fname'] if space == 'volume' else file_paths['fwd_surf_fname']
    fwd = mne.read_forward_solution(fwd_fname)
    src = fwd['src']

    # === COMPUTE SOURCE ESTIMATE ===
    stc_mean = compute_mne_source_estimate(
        subject_id=subject_id,
        fwd=fwd,
        epo_fname=file_paths['epoched_epo_fname'],
        save_dir=file_paths['deriv_folder']
    )

    # === MORPH TO FSAVERAGE ===
    stc_morphed, src_fs = morph_mne_stc_to_fsaverage(
        stc=stc_mean,
        subject_id=subject_id,
        src=src,
        paths=paths
    )

    # === PLOT ===
    plot_source_band_power(
        subject_id=subject_id,
        band=band,
        stc=stc_morphed,
        paths=paths,
        src_fs=src_fs
    )

    return stc_mean

if __name__ == "__main__":
    stc = main()
