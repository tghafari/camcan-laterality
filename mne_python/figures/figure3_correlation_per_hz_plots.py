# -*- coding: utf-8 -*-
"""
==========================================
figure3_correlation_per_hz_plots

Reads significant sensor pairs from CSV saved after
cluster permutation (per substr/band/ch_type), loads
their per-frequency Spearman r, averages across pairs,
and plots correlation vs frequency.

You have to make sure the substr and band already
have a cluster permutated significant correlation
and the results are saved in a csv.

Also you should input the sign of correlation
from the results plots document 
(/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-Universityof\
  Birmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures120820\
  25.docx).

Author: Tara Ghafari 
tara.ghafari@gmail.com
15/08/2025
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
        quinna_dir  = '/Volumes/quinna-camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'deriv_dir': op.join(sub2ctx_dir, 'derivatives'),
        # Adjust to match your data location:
        'corr_dir': op.join(sub2ctx_dir, 'derivatives', 'correlations', 'sensor_pairs_subtraction_nonoise_no-vol-outliers'),
        'sig_csv_dir': op.join(sub2ctx_dir, 'derivatives', 'correlations', 'sensor_bands', 'bands_significant_sensors_cluster-perm'),
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


def align_and_bin(series_list, bin_hz=1.0, use_centers=False):
    """
    Align all pair series by inner-joining on frequency, then bin.
    Returns (aligned_df, avg_corr_per_bin_df, avg_corr_series).
    """
    aligned = pd.concat(series_list, axis=1, join='inner').sort_index()
    avg_corr_per_bin, avg_corr_series = average_corr_by_bin(aligned, bin_hz=float(bin_hz), use_centers=use_centers)
    return aligned, avg_corr_per_bin, avg_corr_series


# --------------------- Plot (publication style) --------------------- #
def plot_correlation_per_hz(avg_corr_series, substr, band, ch_type, correlation,
                            out_dir, xlim_left=0, xlim_right=60.5, font_family='Arial'):
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
        fig.savefig(op.join(out_dir, f"{base}.svg"), format='svg', bbox_inches='tight')

    # Color: RdBu-ish
    color = '#d6604d' if correlation == 'pos' else '#4393c3'

    # Standard EEG/MEG band ranges (Hz)
    band_ranges = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta':  (14, 40),
    }
    band_key = str(band).strip().lower()
    band_range = band_ranges.get(band_key, None)

    # Set global font family
    plt.rcParams['font.family'] = font_family

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(avg_corr_series.index, avg_corr_series.values, marker='o', color=color)
    ax.set_xlim(left=xlim_left, right=xlim_right)

    # Highlight the band range (same color, higher transparency)
    if band_range is not None:
        # Clip the highlight to current x-lims so it doesn't spill over
        x0 = max(xlim_left, band_range[0])
        x1 = min(xlim_right, band_range[1])
        if x1 > x0:
            ax.axvspan(x0, x1, facecolor=color, alpha=0.15, zorder=0, edgecolor='none')

    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Spearman r', fontsize=14, fontweight='bold')
    ax.set_title(f'{substr}-{band} ({ch_type}): Average correlation across significant pairs', fontsize=16, fontweight='bold')
    ax.grid(True)

    # Bold tick labels
    ax.tick_params(axis='both', labelsize=12)
    # for tick in (ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks()):
    #     tick.label1.set_fontweight('bold')

    ensure_dir(out_dir)
    save_figure_all_formats(
        fig,
        out_dir,
        f'{substr}_{band}_{ch_type}_avg_corr_over_freq'
    )
    plt.show()
    plt.close(fig)

# --------------------------- Main Batch runner --------------------------- #
def average_corr_batch(
    substr, 
    band, 
    ch_type,
    corr_sign,
    bin_hz,
    platform='mac', 
    use_centers=False,
    font_family='Arial'
):
    """
    Batch-run averaged correlation plots across substr/band/ch_type.

    Parameters
    ----------
    platform : {'mac','bluebear'}
    substr : Thal, Caud, Puta, Pall, Hipp, Amyg, Accu
    band : Delta, Theta, Alpha, Beta
    ch_type : mag,grad
    corr_sign : pos for positive correlations, neg for negative 
        (you need to find this from the plots)
    bin_hz : float
        Bin width in Hz for averaging (e.g., 0.5, 1.0, 2.0, 5.0)
    font_family : str
        e.g., 'Arial', 'DejaVu Sans', 'Helvetica'
    """
    paths = setup_paths(platform=platform)
    try:
        # 1) load right-sensor list
        right_sensors = load_right_sensors(paths, substr, band, ch_type)

        # 2) load all pair spectra for these sensors
        series_list = get_series_list_for_right_sensors(paths, right_sensors, substr)

        # 3) align & bin
        aligned_df, per_bin_df, avg_series = align_and_bin(
            series_list, bin_hz=bin_hz, use_centers=use_centers)

        # 4) plot & save
        out_dir = op.join(
            paths['fig_output_root'],
            f'{substr}_{band}_{ch_type}_avg_corr_over_freq_{bin_hz:g}Hz'
        )
        plot_correlation_per_hz(
            avg_series, substr, band, ch_type,
            correlation=corr_sign, out_dir=out_dir,
            xlim_left=0, xlim_right=60.5, font_family=font_family, 
        )

        print(f"[DONE] {substr}-{band}-{ch_type}: "
                f"{aligned_df.shape[1]} pairs • saved -> {out_dir}")

    except FileNotFoundError as e:
        print(f"[MISS] {substr}-{band}-{ch_type}: {e}")
    except RuntimeError as e:
        print(f"[SKIP] {substr}-{band}-{ch_type}: {e}")


# ------------------------------- Entry point ------------------------------- #
def plot_sensor_avg_corr_vs_freq(platform='mac'):
 
    # # Input which combos to run
    # substr      = input("Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
    # band        = input("Enter band (Delta, Theta, Alpha, Beta): ").strip()
    # ch_type     = input("Enter sensortype (mag or grad): ").strip().lower()
    # correlation_sign = input(f"Is the correlation between {substr} and {band} positive (pos) or negative (neg)? ").strip().lower()

    # average_corr_batch(
    #     substr=substr, 
    #     band=band, 
    #     ch_type=ch_type,
    #     corr_sign=correlation_sign,
    #     bin_hz=1.0,  # float      
    #     platform=platform,
    #     use_centers=False,
    #     font_family='Arial'
    # )

    # Or run on exact (substr, band, ch_type, corr_sign) tuples
    significant_relationships = [
        ('Caud', 'Delta', 'grad', 'neg'),
        ('Caud', 'Beta',  'grad', 'neg'),
        ('Hipp', 'Delta', 'grad', 'neg'),
        ('Pall', 'Alpha', 'grad', 'pos'),
        ('Puta', 'Beta',  'grad', 'pos'),
        ('Caud', 'Delta', 'mag',  'neg'),
        ('Caud', 'Theta', 'mag',  'neg'),
        ('Caud', 'Alpha', 'mag',  'neg'),
        ('Caud', 'Beta',  'mag',  'neg'),
        ('Hipp', 'Beta',  'mag',  'pos'),
    ]

    for substr, band, ch_type, corr_sign in significant_relationships:
        average_corr_batch(
            platform=platform,
            substr=substr,               # list with one string
            band=band,
            ch_type=ch_type,
            corr_sign=corr_sign,
            bin_hz=1.0,
            use_centers=False,
            font_family='Arial'
        )

if __name__ == "__main__":
    plot_sensor_avg_corr_vs_freq(platform='mac')

