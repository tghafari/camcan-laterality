# -*- coding: utf-8 -*-
"""
==========================================
CS05_visualise correlations in significant
sensor groups over spectra (0.5 Hz bins)

Reads significant sensor pairs from CSV saved after
cluster permutation (per substr/band/ch_type), loads
their per-frequency Spearman r, averages across pairs,
and plots correlation vs frequency at 0.5 Hz resolution.

Author: Tara Ghafari (adapted/cleaned)
==========================================
"""
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------ Paths ------------------------------ #
def setup_paths(platform='mac'):
    """
    Return a dict of all paths needed by the pipeline based on the platform.
    """
    if platform == 'bluebear':
        quinna_dir  = '/rds/projects/q/quinna-camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
    elif platform == 'mac':
        quinna_dir  = '/Volumes/quinna-camcan-1'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx-1/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'deriv_dir': op.join(sub2ctx_dir, 'derivatives'),
        # Adjust to match your data location:
        'corr_dir': op.join(sub2ctx_dir, 'derivatives', 'correlations', 'sensor_pairs_subtraction_nonoise_no-vol-outliers'),
        'sig_csv_dir': op.join(sub2ctx_dir, 'derivatives', 'correlations', 'bands', 'bands_significant_sensors_cluster-perm'),
        'fig_output_root': '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures',
    }
    return paths


# ----------------------- Load right_sensor list --------------------- #
def load_right_sensors(paths, substr, band, ch_type):
    """
    Read the significant right-sensor names from CSV.
    Expects a 'sensor_names' column with semicolon-separated tokens like 'MEG1141'.
    """
    sig_csv = op.join(paths['sig_csv_dir'], f'{substr}_{band}_{ch_type}_signif_sensors_after_cluster_perm.csv')
    if not op.exists(sig_csv):
        raise FileNotFoundError(f"Significant sensors CSV not found:\n{sig_csv}")

    sig_df = pd.read_csv(sig_csv)
    if 'sensor_names' not in sig_df.columns or sig_df.empty:
        raise RuntimeError(f"'sensor_names' column missing or CSV empty in {sig_csv}")

    # Collect unique tokens
    right_tokens = []
    for names in sig_df['sensor_names']:
        if isinstance(names, str) and names.strip():
            right_tokens.extend([tok.strip() for tok in names.split(';') if tok.strip()])

    # Deduplicate while preserving order
    seen = set()
    right_tokens = [x for x in right_tokens if not (x in seen or seen.add(x))]
    if not right_tokens:
        raise RuntimeError("No right-sensor tokens found in 'sensor_names'.")

    return right_tokens


# -------------------- Find pair folders & load spectra -------------- #
def load_pair_corr_series_from_folder(pair_folder, substr):
    """
    Read per-frequency Spearman r for a pair folder:
      pair_folder/{substr}/{substr}_lat_spectra_substr_spearmanr.csv

    CSV format:
      - first column = frequency (Hz, e.g., 1.0, 1.5, 2.0, ...)
      - second column = r value

    Returns a pandas Series indexed by frequency (float Hz), name=pair folder basename.
    """
    csv_path = op.join(pair_folder, substr, f"{substr}_lat_spectra_substr_spearmanr.csv")
    if not op.exists(csv_path):
        return None
    df = pd.read_csv(csv_path, header=0)
    if df.shape[1] < 2:
        return None

    freq = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    r    = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    s = pd.Series(r.values, index=freq.values, name=op.basename(pair_folder))
    s = s.dropna()
    s.index = s.index.astype(float)
    s = s.sort_index()
    return s


def find_pair_folders(paths, right_sensor):
    """
    Find pair folders under corr_dir that correspond to the given right_sensor.
    Pair folders are named 'MEG####_MEG####'. We keep those that end with '_{right_sensor}'.
    """
    corr_dir = paths['corr_dir']
    if not op.isdir(corr_dir):
        return []
    out = []
    for entry in os.listdir(corr_dir):
        full = op.join(corr_dir, entry)
        if not op.isdir(full):
            continue
        if '_' in entry and entry.endswith(f"_{right_sensor}"):
            left, right = entry.split('_', 1)
            if left.startswith('MEG') and right.startswith('MEG'):
                out.append(full)
    return out


def get_series_list_for_right_sensors(paths, right_sensors, substr):
    """
    For each right sensor, find its pair folders and load their (freq->r) series.
    Returns a list of Series (one per pair found).
    """
    series_list = []
    for right_sensor in right_sensors:
        pair_folders = find_pair_folders(paths, right_sensor)
        if not pair_folders:
            print(f"[INFO] No pair folders found for right sensor {right_sensor}; skipping.")
            continue
        for pf in pair_folders:
            s = load_pair_corr_series_from_folder(pf, substr)
            if s is None or s.empty:
                print(f"[INFO] Missing/empty spectra in: {pf}; skipping.")
                continue
            series_list.append(s)
    if not series_list:
        raise RuntimeError("No per-pair correlation spectra could be loaded. Check folder naming and file locations.")
    return series_list


# --------------------------- Align and bin -------------------------- #
def average_corr_by_bin(aligned_df, bin_hz=2.0, use_centers=True):
    """
    Average correlation within bins of width `bin_hz` (rows), then across columns (pairs).

    Parameters
    ----------
    aligned_df : DataFrame
        index = frequency (Hz), columns = pairs, values = r.
    bin_hz : float
        bin width in Hz (e.g., 0.5, 1.0, 2.0, 5.0).
    use_centers : bool
        If True, index of returned Series/DF are bin centers; else bin starts.

    Returns
    -------
    avg_corr_per_bin : DataFrame
        Per-bin averages per pair (rows=bins, cols=pairs).
    avg_corr_series : Series
        Per-bin average across pairs (rows=bins).
    """
    if bin_hz <= 0:
        raise ValueError("bin_hz must be > 0")

    freqs = aligned_df.index.values.astype(float)
    bin_starts = np.floor(freqs / bin_hz) * bin_hz
    bin_starts = np.round(bin_starts, 6)

    avg_corr_per_bin = aligned_df.groupby(bin_starts).mean().sort_index()
    avg_corr_series = avg_corr_per_bin.mean(axis=1)

    if use_centers:
        centers = np.round(avg_corr_series.index.values + bin_hz / 2.0, 6)
        avg_corr_series.index = centers
        avg_corr_per_bin.index = centers
        avg_corr_series.index.name = f'freq_bin_center_{bin_hz:g}Hz'
        avg_corr_per_bin.index.name = f'freq_bin_center_{bin_hz:g}Hz'
    else:
        avg_corr_series.index.name = f'freq_bin_start_{bin_hz:g}Hz'
        avg_corr_per_bin.index.name = f'freq_bin_start_{bin_hz:g}Hz'

    return avg_corr_per_bin, avg_corr_series


def align_and_bin(series_list, bin_hz=2.0, use_centers=True):
    """
    Align all pair series by inner-joining on frequency, then bin.
    Returns (aligned_df, avg_corr_per_bin_df, avg_corr_series).
    """
    aligned = pd.concat(series_list, axis=1, join='inner').sort_index()
    avg_corr_per_bin, avg_corr_series = average_corr_by_bin(aligned, bin_hz=float(bin_hz), use_centers=use_centers)
    return aligned, avg_corr_per_bin, avg_corr_series


# --------------------- Plot (publication style) --------------------- #
def plot_correlation_per_hz(avg_corr_series, substr, band, ch_type, correlation='pos',
                            font_family='Arial', bin_hz=2.0, out_dir='.', basename='plot'):
    """
    Plot correlation vs frequency (per bin), and save TIFF/PNG/SVG at 800 dpi.
    """
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def save_figure_all_formats(fig, out_dir, basename, dpi=800):
        ensure_dir(out_dir)
        base = basename.replace(' ', '_')
        fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
        fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
        fig.savefig(op.join(out_dir, f"{base}.svg"),              format='svg', bbox_inches='tight')

    # Color: RdBu-ish
    color = '#d6604d' if correlation == 'pos' else '#4393c3'

    # Set global font family
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_corr_series.index, avg_corr_series.values, marker='o', color=color)

    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Spearman r', fontsize=14, fontweight='bold')
    ax.set_title(f'{substr}-{band}: Average correlation across significant pairs', fontsize=16, fontweight='bold')
    ax.grid(True)

    # Bold tick labels
    ax.tick_params(axis='both', labelsize=12)
    for tick in (ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()):
        tick.label1.set_fontweight('bold')

    ensure_dir(out_dir)
    save_figure_all_formats(
        fig,
        out_dir,
        f'{substr}_{band}_{ch_type}_avg_corr_over_freq_{bin_hz:g}Hz'
    )
    plt.show()
    plt.close(fig)


# ------------------------------- Main ------------------------------- #
def main(platform='mac', substr='Thal', band='Beta', ch_type='mag',
         correlation='pos', bin_hz=2.0, use_centers=True, font_family='Arial'):
    """
    Orchestrates the full pipeline:
      1) setup_paths
      2) load_right_sensors
      3) find pair folders & load spectra
      4) align & bin
      5) plot & save

    Returns
    -------
    aligned_df : DataFrame
        Frequencies x pairs (inner-joined) raw r-values.
    avg_corr_per_bin_df : DataFrame
        Per-bin averages per pair.
    avg_corr_series : Series
        Per-bin averages across pairs (what is plotted).
    out_dir : str
        Output directory used for saving figures.
    """
    paths = setup_paths(platform=platform)

    right_sensors = load_right_sensors(paths, substr, band, ch_type)
    series_list = get_series_list_for_right_sensors(paths, right_sensors, substr)

    aligned_df, avg_corr_per_bin_df, avg_corr_series = align_and_bin(
        series_list, bin_hz=bin_hz, use_centers=use_centers
    )

    out_dir = op.join(
        paths['fig_output_root'],
        f'{substr}_{band}_{ch_type}_avg_corr_over_freq_{bin_hz:g}Hz'
    )
    plot_correlation_per_hz(
        avg_corr_series, substr, band, ch_type,
        correlation=correlation, font_family=font_family,
        bin_hz=bin_hz, out_dir=out_dir
    )

    print(f"[DONE] Used {len(right_sensors)} right-sensor tokens; "
          f"loaded {aligned_df.shape[1]} pair spectra; "
          f"saved figures to: {out_dir}")

    return aligned_df, avg_corr_per_bin_df, avg_corr_series, out_dir

if __name__ == '__main__':
    # Prompt for interactive inputs
    platform    = input("Enter platform ('mac' or 'bluebear'): ").strip().lower()
    substr      = input("Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    band        = input("Enter band (Delta, Theta, Alpha, Beta): ").strip()
    ch_type     = input("Enter sensortype (mag or grad): ").strip().lower()
    correlation = input(f"Is the correlation between {substr} and {band} positive (pos) or negative (neg)? ").strip().lower()
    bin_hz      = float(input("Enter bin width in Hz (e.g., 0.5, 1.0, 2.0): ").strip())

    # Run main pipeline
    main(
        platform=platform,
        substr=substr,
        band=band,
        ch_type=ch_type,
        correlation=correlation,
        bin_hz=bin_hz,
        use_centers=True,
        font_family='Arial'
    )




Read last chatgpt answer










































# import os
# import os.path as op
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import mne  # only used here to sanity-check availability if needed (optional)

# # ------------- User inputs ------------- #
# platform    = 'mac'   # 'mac' or 'bluebear'
# substr      = input("Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
# band        = input("Enter band (Delta, Theta, Alpha, Beta): ").strip()
# ch_type     = input("Enter sensortype (mag or grad): ").strip().lower()  # not directly used in paths below, but kept for file naming
# correlation = input(f"Is the correlation between {substr} and {band} positive (pos) or negative (neg)?").strip().lower() 


# # ------------- Paths ------------- #
# if platform == 'bluebear':
#     quinna_dir = '/rds/projects/q/quinna-camcan'
#     jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
#     sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
# elif platform == 'mac':
#     quinna_dir = '/Volumes/quinna-camcan-1'
#     jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
#     sub2ctx_dir = '/Volumes/jenseno-sub2ctx-1/camcan'
# else:
#     raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

# # Where per-pair frequency correlations live (as in your previous script)
# deriv_dir = op.join(sub2ctx_dir, 'derivatives')
# corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise_no-vol-outliers')

# # Where the per-(substr, band, ch_type) significant sensors CSV was saved previously
# # (created in an earlier step that saved *_signif_sensors_after_cluster_perm.csv)
# # Significant sensors CSV produced earlier
# sig_csv_dir = op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_significant_sensors_cluster-perm')
# sig_csv     = op.join(sig_csv_dir, f'{substr}_{band}_{ch_type}_signif_sensors_after_cluster_perm.csv')

# # Output dir for figures
# fig_output_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures'

# # --------------------- Helpers --------------------------- #
# def ensure_dir(path: str) -> None:
#     os.makedirs(path, exist_ok=True)

# def load_pair_corr_series_from_folder(pair_folder: str, substr: str) -> pd.Series | None:
#     """
#     Load per-frequency Spearman r for a pair folder:
#       pair_folder/{substr}/{substr}_lat_spectra_substr_spearmanr.csv

#     Assumes CSV first column = frequency (Hz), second column = r.
#     Returns a Series indexed by frequency (float Hz), name = pair folder basename.
#     """
#     csv_path = op.join(pair_folder, substr, f"{substr}_lat_spectra_substr_spearmanr.csv")
#     if not op.exists(csv_path):
#         return None
#     df = pd.read_csv(csv_path, header=0)
#     if df.shape[1] < 2:
#         return None

#     freq = pd.to_numeric(df.iloc[:, 0], errors='coerce')
#     r    = pd.to_numeric(df.iloc[:, 1], errors='coerce')
#     s = pd.Series(r.values, index=freq.values, name=op.basename(pair_folder))
#     s = s.dropna()
#     s.index = s.index.astype(float)
#     s = s.sort_index()
#     return s

# def find_pair_folders_for_right_sensor(right_sensor: str) -> list[str]:
#     """
#     Find pair folders under corr_dir that correspond to the given right_sensor.
#     Pair folders are named 'MEG####_MEG####'. We keep those that end with '_{right_sensor}'.
#     """
#     if not op.isdir(corr_dir):
#         return []
#     out = []
#     for entry in os.listdir(corr_dir):
#         full = op.join(corr_dir, entry)
#         if not op.isdir(full):
#             continue
#         # Expect pattern MEG####_MEG#### and exact match on right side
#         if '_' in entry and entry.endswith(f"_{right_sensor}"):
#             left, right = entry.split('_', 1)
#             if left.startswith('MEG') and right.startswith('MEG'):
#                 out.append(full)
#     return out
# import numpy as np

# def average_corr_by_bin(aligned: pd.DataFrame, bin_hz: float = 2.0, use_centers: bool = True):
#     """
#     Average correlation across (a) rows within frequency bins of width `bin_hz`,
#     then (b) across columns (pairs).

#     Parameters
#     ----------
#     aligned : DataFrame
#         Rows = frequencies (Hz, as index), columns = pairs, values = Spearman r.
#         Assumes your rows include 0.5 Hz steps (e.g., 1.0, 1.5, 2.0, ...).
#     bin_hz : float
#         Width of the bin in Hz (e.g., 1.0, 2.0, 5.0).
#     use_centers : bool
#         If True, returns the Series indexed by bin centers; otherwise by bin starts.

#     Returns
#     -------
#     avg_corr_per_bin : DataFrame
#         Rows = bins, columns = pairs, each cell averaged within the bin.
#     avg_corr : Series
#         Averaged across pairs for each bin.
#     """
#     if bin_hz <= 0:
#         raise ValueError("bin_hz must be > 0")

#     # Ensure float index and sorted
#     freqs = aligned.index.values.astype(float)
#     # Compute bin starts (avoid float artifacts by rounding)
#     bin_starts = np.floor(freqs / bin_hz) * bin_hz
#     bin_starts = np.round(bin_starts, 6)

#     # Average within each bin for every pair
#     avg_corr_per_bin = aligned.groupby(bin_starts).mean().sort_index()

#     # Average across pairs for each bin
#     avg_corr_across_sensors = avg_corr_per_bin.mean(axis=1)

#     # Optionally relabel index to bin centers for nicer plotting
#     if use_centers:
#         centers = np.round(avg_corr_across_sensors.index.values + bin_hz / 2.0, 6)
#         avg_corr_across_sensors.index = centers
#         avg_corr_per_bin.index = centers
#         avg_corr_across_sensors.index.name = f'freq_bin_center_{bin_hz:g}Hz'
#         avg_corr_per_bin.index.name = f'freq_bin_center_{bin_hz:g}Hz'
#     else:
#         avg_corr_across_sensors.index.name = f'freq_bin_start_{bin_hz:g}Hz'
#         avg_corr_per_bin.index.name = f'freq_bin_start_{bin_hz:g}Hz'

#     return avg_corr_per_bin, avg_corr_across_sensors

# def save_figure_all_formats(fig: plt.Figure, out_dir: str, basename: str, dpi: int = 800) -> None:
#     ensure_dir(out_dir)
#     base = basename.replace(' ', '_')
#     fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
#     fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
#     fig.savefig(op.join(out_dir, f"{base}.svg"),              format='svg', bbox_inches='tight')

# # ------------------ Load right-sensor names ------------------ #
# if not op.exists(sig_csv):
#     raise FileNotFoundError(f"Significant sensors CSV not found:\n{sig_csv}")

# sig_df = pd.read_csv(sig_csv)

# if 'sensor_names' not in sig_df.columns or sig_df.empty:
#     raise RuntimeError(f"'sensor_names' column missing or CSV empty in {sig_csv}")

# # Collect unique right-sensor tokens (e.g., 'MEG1141'), semicolon-separated per row
# right_tokens = []
# for names in sig_df['sensor_names']:
#     if isinstance(names, str) and names.strip():
#         right_tokens.extend([tok.strip() for tok in names.split(';') if tok.strip()])

# # Deduplicate while preserving order
# seen = set()
# right_tokens = [x for x in right_tokens if not (x in seen or seen.add(x))]

# if not right_tokens:
#     raise RuntimeError("No right-sensor tokens found in 'sensor_names'.")

# # ------------- Find pair folders and load spectra ------------- #
# series_list = []
# n_tokens = 0
# for right_sensor in right_tokens:
#     n_tokens += 1
#     pair_folders = find_pair_folders_for_right_sensor(right_sensor)
#     if not pair_folders:
#         print(f"[INFO] No pair folders found for right sensor {right_sensor}; skipping.")
#         continue

#     for pf in pair_folders:
#         s = load_pair_corr_series_from_folder(pf, substr)
#         if s is None or s.empty:
#             print(f"[INFO] Missing/empty spectra in: {pf}; skipping.")
#             continue
#         series_list.append(s)

# if not series_list:
#     raise RuntimeError("No per-pair correlation spectra could be loaded. Check folder naming and file locations.")


# # Align by inner-joining on frequency (index)
# aligned = pd.concat(series_list, axis=1, join='inner').sort_index()

# # Choose your bin width (Hz)
# bin_hz = 2  # try 1.0, 2.0, 5.0, etc.

# # Average across all matched pairs at each frequency - this only works for per 0.5 hz plotting
# # avg_corr = aligned.mean(axis=1)

# avg_corr_per_bin, avg_corr_across_sensors = average_corr_by_bin(aligned, bin_hz=bin_hz, use_centers=True)

# # ------------------------ Plot ---------------------------- #
# # Plot
# plt.figure(figsize=(10, 6))
# color = '#d6604d' if correlation == 'pos' else '#4393c3'
# plt.plot(avg_corr_across_sensors.index, avg_corr_across_sensors.values, marker='o', color=color)
# plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold', fontname='Arial')
# plt.ylabel('Average Spearman r', fontsize=14, fontweight='bold', fontname='Arial')
# plt.title(f'{substr}-{band}: Average correlation across significant pairs', 
#     fontsize=16, fontweight='bold', fontname='Arial')
# plt.grid(True)

# # ------------------ Save (TIFF/PNG/SVG) ------------------- #
# out_dir  = op.join(fig_output_root, f'{substr}_{band}_{ch_type}_avg_corr_over_freq_{bin_hz}')
# basename = f'{substr}_{band}_{ch_type}_avg_corr_over_freq_{bin_hz}'
# save_figure_all_formats(plt.gcf(), out_dir, basename, dpi=800)
# plt.show()

# print(f"[DONE] Used {len(right_tokens)} right-sensors; "
#       f"loaded {aligned.shape[1]} pair spectra; "
#       f"saved figures to: {out_dir}")



