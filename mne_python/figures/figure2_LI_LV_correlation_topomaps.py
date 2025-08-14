"""
Purpose
-------
Plot correlation topoplots showing significant sensors after cluster permutation tests.

What it does
------------
1) Loads the significant sensor indices (and names) saved by St01 into
   `paths['cluster_perm_signif_sensors']` (a CSV created earlier).
2) Loads sensor-wise observed correlations for a given band and subcortical structure.
3) Converts Spearman r to t for visualization.
4) Builds a significance mask from the cluster-permutation results (filtered by `substr` & `band`).
5) Plots topomaps with MNE and saves figures (TIFF, PNG, SVG) at 800 dpi.

Inputs (requested at runtime)
-----------------------------
- substr: subcortical structure, one of: Thal, Caud, Puta, Pall, Hipp, Amyg, Accu
- band:   frequency band, one of: Delta, Theta, Alpha, Beta
- ch_type: sensor type, one of: 'mag' or 'grad'

Outputs
-------
- Figures saved to:
  op.join(paths['save_path'], f"{substr}_{band}_cluster_sig_topoplots")
"""

import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne


# ----------------------------- Utilities ----------------------------- #

def setup_paths(platform: str = 'mac') -> dict:
    """Return project paths depending on the host platform."""
    if platform == 'bluebear':
        quinna_dir = '/rds/projects/q/quinna-camcan/'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/bands'),
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_no-vol-outliers.csv'),
        'sub_list': op.join(quinna_dir, 'dataman/data_information/dblCheck_last_FINAL_sublist-vol-outliers-removed.csv'),
        'correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands_sensor_pairs_subtraction_nooutlier-psd'),
        'signif_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_signif_correlations_subtraction_nooutlier-psd'),
        'all_correlation_dir': op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_all_correlations_subtraction_nonoise_no-vol-outliers'),
        'sample_meg_file': op.join(quinna_dir, 'cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'),
        'sensor_layout': op.join(quinna_dir, 'dataman/data_information/combined_sensors_layout_names.csv'),
        'spectra_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/all_sensors_all_subs_all_freqs_subtraction_nonoise_no-vol-outliers_combnd-grads'),
        'save_path': '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN/results/Manuscript/Figures',
        # Master CSV with significant clusters (created in previous step)
        'cluster_perm_signif_sensors': op.join(
            sub2ctx_dir,
            'derivatives/correlations/bands/cluster_perm_significant_sensors.csv'
        ),
    }
    return paths

def r_to_t(r_vals: np.ndarray, n: int) -> np.ndarray:
    """Convert Spearman r-values to t-values for visualization."""
    r_vals = np.asarray(r_vals, dtype=float)
    # Guard against r ~ 1.0 numerical issues:
    eps = 1e-12
    denom = np.clip(1 - r_vals**2, eps, None)
    return r_vals * np.sqrt((n - 2) / denom)

def read_raw_info(paths: dict, ch_type: str):
    """
    Read MEG info for plotting. For grads, also return magnetometer info (sometimes useful).
    Returns:
        - If ch_type == 'mag': (raw_mag, info_mag)
        - If ch_type == 'grad': (raw_grad, info_grad, info_mag)
    """
    raw = mne.io.read_raw_fif(paths['sample_meg_file'], preload=False, verbose=False)
    layout = pd.read_csv(paths['sensor_layout'])
    right_sensors = layout['right_sensors'].dropna().tolist()

    if ch_type == 'mag':
        channels = [ch for ch in right_sensors if ch.endswith('1')]
        raw = raw.copy().pick('mag').pick(channels)
        return raw, raw.info

    elif ch_type == 'grad':
        channels_mag = [ch for ch in right_sensors if ch.endswith('1')]
        info_mag = raw.copy().pick('mag').pick(channels_mag).info

        channels_grad = [ch for ch in right_sensors if ch.endswith('2')]
        raw = raw.copy().pick('grad').pick(channels_grad)
        return raw, raw.info, info_mag

    else:
        raise ValueError("ch_type must be 'mag' or 'grad'.")

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def save_figure_all_formats(fig: plt.Figure, out_dir: str, basename: str, dpi: int = 800) -> None:
    """Save figure as TIFF, PNG, and SVG with given dpi."""
    ensure_dir(out_dir)
    # Sanitize basename a bit
    basename = basename.replace(' ', '_')
    fig.savefig(op.join(out_dir, f"{basename}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{basename}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{basename}.svg"),              format='svg', bbox_inches='tight')

# --------------------------- Main plotting --------------------------- #
def plot_cluster_significant_topoplots(paths: dict, substr: str, band: str, ch_type: str) -> None:
    """
    Plot topomap of t-transformed Spearman correlations, masking sensors that were
    significant after cluster permutation, for the requested (substr, band, ch_type).
    """

    # --- Load MEG info and correlation table ---
    if ch_type == 'grad':
        raw, info, _info_mag = read_raw_info(paths, ch_type)
    else:
        raw, info = read_raw_info(paths, ch_type)

    # Observed correlations (per sensor)
    corr_path = op.join(paths['all_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
    if not op.isfile(corr_path):
        raise FileNotFoundError(f"Correlation file not found: {corr_path}")
    corr_df = pd.read_csv(corr_path)

    # Select correct sensor suffix by type
    if ch_type == 'mag':
        sensor_mask = corr_df['sensor_pair'].str.endswith('1')
    elif ch_type == 'grad':
        sensor_mask = corr_df['sensor_pair'].str.endswith('2')  # combined grads
    else:
        raise ValueError(f"Unsupported ch_type: {ch_type}")

    filtered_df = corr_df[sensor_mask].copy()
    if filtered_df.empty:
        print(f"[INFO] No matching sensors for ch_type={ch_type} in {substr}-{band}.")
        return

    # --- Compute t-values from r ---
    li_csv = op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv')
    if not op.isfile(li_csv):
        raise FileNotFoundError(f"LI CSV not found: {li_csv}")
    li_df = pd.read_csv(li_csv)

    r_obs = filtered_df[f'{band.lower()}_rval'].to_numpy()
    n_subjects = len(li_df)
    t_obs = r_to_t(r_obs, n=n_subjects)

    # --- Create 2D positions for plot_topomap ---
    # For grads, build 2D (x,y) from channel locs of the picked info object.
    if ch_type == 'grad':
        grad_picks = mne.pick_types(info, meg='grad', eeg=False, stim=False, exclude=[])
        # Sanity check: lengths must match data
        if len(grad_picks) != len(t_obs):
            raise RuntimeError(
                f"Length mismatch: {len(grad_picks)} grad channels in info vs {len(t_obs)} data points in t_obs."
            )
        pos2d = np.vstack([info['chs'][p]['loc'][:2] for p in grad_picks])
        pos_info = pos2d
    else:
        # For mags, MNE accepts Info directly if it matches data length
        mag_picks = mne.pick_types(info, meg='mag', eeg=False, stim=False, exclude=[])
        if len(mag_picks) != len(t_obs):
            raise RuntimeError(
                f"Length mismatch: {len(mag_picks)} mag channels in info vs {len(t_obs)} data points in t_obs."
            )
        pos_info = info

    # --- Load significant clusters master CSV and filter for (substr, band) ---
    signif_csv = paths['cluster_perm_signif_sensors']
    if not op.isfile(signif_csv):
        raise FileNotFoundError(
            f"Significant clusters CSV not found: {signif_csv}\n"
            "Make sure you created it in the previous step."
        )
    df_sig = pd.read_csv(signif_csv)

    df_sub_band = df_sig[(df_sig['structure'] == substr) & (df_sig['band'] == band)].copy()
    if df_sub_band.empty:
        print(f"[INFO] No significant clusters found for {substr}-{band}. Plotting without a mask.")
        sig_indices = []
    else:
        # Sensors stored as semicolon-separated indices in 'sensors'
        sig_indices = []
        for sensor_str in df_sub_band['sensors']:
            if isinstance(sensor_str, str) and sensor_str.strip():
                sig_indices.extend(map(int, sensor_str.split(';')))
        sig_indices = sorted(set(sig_indices))

    # --- Build significance mask ---
    mask = np.zeros(len(t_obs), dtype=bool)
    if len(sig_indices) > 0:
        # Guard against out-of-bounds indices in the CSV
        valid_idx = [i for i in sig_indices if 0 <= i < len(mask)]
        if len(valid_idx) < len(sig_indices):
            print("[WARN] Some significant sensor indices were out of bounds and were ignored.")
        mask[valid_idx] = True

    # --- Plot ---
    mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                       linewidth=1, markersize=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    im, cn = mne.viz.plot_topomap(
        t_obs, pos_info, mask=mask, mask_params=mask_params,
        vlim=(float(np.nanmin(t_obs)), float(np.nanmax(t_obs))),
        contours=0, image_interp='nearest', cmap='RdBu_r', show=False, axes=ax
    )
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label('t (from Spearman r)', fontsize=11)

    title = f'{substr}-{band} • t from Spearman r ({ch_type}) • after cluster permutation'
    ax.set_title(title, fontsize=12)

    # Optional: You had ax.set_xlim(0,) to "remove the left half of topoplot".
    # That can look odd depending on the head geometry; comment out by default.
    # ax.set_xlim(0, )

    fig.tight_layout()

    # --- Save outputs ---
    out_dir = op.join(paths['save_path'], f'{substr}_{band}_cluster_sig_topoplots')
    base = f'{substr}_{band}_{ch_type}_cluster_sig_topomap'
    save_figure_all_formats(fig, out_dir, base, dpi=800)
    plt.show()
    plt.close(fig)

    print(f"[DONE] Saved figures to: {out_dir}")

# ------------------------------- Run ------------------------------- #

if __name__ == "__main__":
    platform = 'mac'  # or 'bluebear'
    paths = setup_paths(platform)

    substr = input("Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    band = input("Enter band (Delta, Theta, Alpha, Beta): ").strip()
    ch_type = input("Enter sensortype (mag or grad): ").strip().lower()

    # Basic validation
    valid_sub = {'Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu'}
    valid_band = {'Delta', 'Theta', 'Alpha', 'Beta'}
    valid_ch = {'mag', 'grad'}

    if substr not in valid_sub:
        raise ValueError(f"Invalid substr '{substr}'. Must be one of {sorted(valid_sub)}.")
    if band not in valid_band:
        raise ValueError(f"Invalid band '{band}'. Must be one of {sorted(valid_band)}.")
    if ch_type not in valid_ch:
        raise ValueError(f"Invalid ch_type '{ch_type}'. Must be 'mag' or 'grad'.")

    plot_cluster_significant_topoplots(paths, substr, band, ch_type)




platform = 'mac'  # or 'bluebear'
paths = setup_paths(platform)
substr = input('Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu):').strip()
band = input('Enter band (Delta, Theta, Alpha, Beta):').strip()
ch_type = input('Enter sensortype (mag or grad):').strip()

def plot_cluster_significant_topoplots(paths, substr, band, ch_type):
    raw, info, *other = read_raw_info(paths, ch_type)  
    corr_path = op.join(paths['all_correlation_dir'], f'{substr}_allpairs_{band}_spearmanr.csv')
    corr_df = pd.read_csv(corr_path)

    # Filter sensor pairs based on magnetometer or gradiometer channel suffix
    if ch_type == 'mag':
        sensor_mask = corr_df['sensor_pair'].str.endswith('1')
        central_sensors = ['MEG0811', 'MEG1011','MEG2121']  # should be ignored in cluster testing
        central_sensor_indices = [11, 12, 30]
    elif ch_type == 'grad':
        sensor_mask = corr_df['sensor_pair'].str.endswith('2')  # grads are combined and saved in sensors ending in '2', no sensors ending in '3'
        central_sensors = ['MEG0812', 'MEG1012', 'MEG2122']  # should be ignored in cluster testing
        central_sensor_indices = [48, 49, 50]
    else:
        raise ValueError(f"Unsupported ch_type: {ch_type}")


    filtered_df = corr_df[sensor_mask]
    if filtered_df.empty:
        print(f"No matching sensors for {ch_type} in {substr}-{band}")
        return

    # Load lateralized power data for this band to calculate shuffled correlations and use variables for plotting
    li_df = pd.read_csv(op.join(paths['LI_dir'], f'{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv'))
    r_obs = filtered_df[f'{band.lower()}_rval'].to_numpy()  
    n_subjects = len(li_df)  # in our last_FINAl list this is 532 (vol outliers and errored subjects in lateralised psd calculation are removed)
    t_obs = r_to_t(r_obs, n=n_subjects)  # indices here correspond to r_obs and not adjacency

        # Prepare the data for visualisation
    """we use the positions of channels for grads.
    using mag info messes up the index of channels and is therefore not correct."""
    if ch_type == 'grad':  
        # trying out different methods to plot grads in the correct position - looks ok just needs to align to right sensor locations
        grad_picks = mne.pick_types(info, meg='grad')
        grad_names = [info['ch_names'][p] for p in grad_picks]  # this is essentialy identical to 
                                                                #sensor_names = [pair.split('_')[1] for pair in filtered_df['sensor_pair']]
                                                                # we are just using 'info' to be more principled so:
                                                                # In [198]: grad_names == sensor_names
                                                                # Out[198]: True

        pos2d = np.vstack([info['chs'][p]['loc'][:2] for p in grad_picks])
        pos_info = pos2d
    else:
        pos_info = info

    # Define the significant clusters
    mask_params = dict(
        marker='o', markerfacecolor='w', markeredgecolor='k',
        linewidth=1, markersize=10
    ) 

    # Load the CSV with all significant clusters
    df = pd.read_csv(paths['cluster_perm_signif_sensors'])

    # Filter for the specific band and structure
    df_substr_band = df[(df['structure'] == substr) & (df['band'] == band)]

    # Extract all significant sensor indices for these clusters
    # Sensors are stored as semicolon-separated strings, so split and flatten
    sig_indices = []
    for sensor_str in df_substr_band['sensors']:
        sig_indices.extend(map(int, sensor_str.split(';')))

    # Remove duplicates and sort
    sig_indices = sorted(set(sig_indices))

    # 5. Visualize topomap with significant mask and cluster labels
    fig, ax = plt.subplots()

    # Define the significant mask
    mask = np.zeros(len(info['ch_names']), dtype=bool)
    mask[sig_indices] = True  # use the filtered indices

    # Plot topomap
    im, cn = mne.viz.plot_topomap(
        t_obs, pos_info, mask=mask, mask_params=mask_params,
        vlim=(min(t_obs), max(t_obs)), contours=0, image_interp='nearest', 
        cmap='RdBu_r', show=False, axes=ax
    )  
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', location='bottom')
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation Values', fontsize=14)
    ax.set_xlim(0, )  # remove the left half of topoplot
    ax.set_title(f'{substr}-{band} t transformed Spearman r ({ch_type})-after cluster permutation')
    plt.show()
