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
import mne  # only used here to sanity-check availability if needed (optional)

# ------------- User inputs ------------- #
platform = 'mac'   # 'mac' or 'bluebear'
substr   = input("Enter substr (Thal, Caud, Puta, Pall, Hipp, Amyg, Accu): ").strip()
band     = input("Enter band (Delta, Theta, Alpha, Beta): ").strip()
ch_type  = input("Enter sensortype (mag or grad): ").strip().lower()  # not directly used in paths below, but kept for file naming

# ------------- Paths ------------- #
if platform == 'bluebear':
    quinna_dir = '/rds/projects/q/quinna-camcan'
    jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention'
    sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
elif platform == 'mac':
    quinna_dir = '/Volumes/quinna-camcan-1'
    jenseno_dir = '/Volumes/jenseno-avtemporal-attention'
    sub2ctx_dir = '/Volumes/jenseno-sub2ctx/camcan'
else:
    raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

# Where per-pair frequency correlations live (as in your previous script)
deriv_dir = op.join(sub2ctx_dir, 'derivatives')
corr_dir = op.join(deriv_dir, 'correlations/sensor_pairs_subtraction_nonoise_no-vol-outliers')

# Where the per-(substr, band, ch_type) significant sensors CSV was saved previously
# (created in an earlier step that saved *_signif_sensors_after_cluster_perm.csv)
# Significant sensors CSV produced earlier
sig_csv_dir = op.join(sub2ctx_dir, 'derivatives/correlations/bands/bands_significant_sensors_cluster-perm')
sig_csv     = op.join(sig_csv_dir, f'{substr}_{band}_{ch_type}_signif_sensors_after_cluster_perm.csv')

# Output dir for figures
fig_output_root = '/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures'

# --------------------- Helpers --------------------------- #
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_pair_corr_series_from_folder(pair_folder: str, substr: str) -> pd.Series | None:
    """
    Load per-frequency Spearman r for a pair folder:
      pair_folder/{substr}/{substr}_lat_spectra_substr_spearmanr.csv

    Assumes CSV first column = frequency (Hz), second column = r.
    Returns a Series indexed by frequency (float Hz), name = pair folder basename.
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

def find_pair_folders_for_right_sensor(right_sensor: str) -> list[str]:
    """
    Find pair folders under corr_dir that correspond to the given right_sensor.
    Pair folders are named 'MEG####_MEG####'. We keep those that end with '_{right_sensor}'.
    """
    if not op.isdir(corr_dir):
        return []
    out = []
    for entry in os.listdir(corr_dir):
        full = op.join(corr_dir, entry)
        if not op.isdir(full):
            continue
        # Expect pattern MEG####_MEG#### and exact match on right side
        if '_' in entry and entry.endswith(f"_{right_sensor}"):
            left, right = entry.split('_', 1)
            if left.startswith('MEG') and right.startswith('MEG'):
                out.append(full)
    return out

def save_figure_all_formats(fig: plt.Figure, out_dir: str, basename: str, dpi: int = 800) -> None:
    ensure_dir(out_dir)
    base = basename.replace(' ', '_')
    fig.savefig(op.join(out_dir, f"{base}.tiff"), dpi=dpi, format='tiff', bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.png"),  dpi=dpi, format='png',  bbox_inches='tight')
    fig.savefig(op.join(out_dir, f"{base}.svg"),              format='svg', bbox_inches='tight')

# ------------------ Load right-sensor names ------------------ #
if not op.exists(sig_csv):
    raise FileNotFoundError(f"Significant sensors CSV not found:\n{sig_csv}")

sig_df = pd.read_csv(sig_csv)

if 'sensor_names' not in sig_df.columns or sig_df.empty:
    raise RuntimeError(f"'sensor_names' column missing or CSV empty in {sig_csv}")

# Collect unique right-sensor tokens (e.g., 'MEG1141'), semicolon-separated per row
right_tokens = []
for names in sig_df['sensor_names']:
    if isinstance(names, str) and names.strip():
        right_tokens.extend([tok.strip() for tok in names.split(';') if tok.strip()])

# Deduplicate while preserving order
seen = set()
right_tokens = [x for x in right_tokens if not (x in seen or seen.add(x))]

if not right_tokens:
    raise RuntimeError("No right-sensor tokens found in 'sensor_names'.")

# ------------- Find pair folders and load spectra ------------- #
series_list = []
n_tokens = 0
for right_sensor in right_tokens:
    n_tokens += 1
    pair_folders = find_pair_folders_for_right_sensor(right_sensor)
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

# Align by inner-joining on frequency (index)
aligned = pd.concat(series_list, axis=1, join='inner').sort_index()

# Average across all matched pairs at each frequency
avg_corr = aligned.mean(axis=1)

# ------------------------ Plot ---------------------------- #
plt.figure(figsize=(10, 6))
plt.plot(avg_corr.index, avg_corr.values, marker='o')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Average Spearman r (across significant pairs)')
plt.title(f'{substr}: Average correlation across significant pairs (from {band}, {ch_type})')
plt.grid(True)

# ------------------ Save (TIFF/PNG/SVG) ------------------- #
out_dir  = op.join(fig_output_root, f'{substr}_{band}_{ch_type}_avg_corr_over_freq_0p5Hz')
basename = f'{substr}_{band}_{ch_type}_avg_corr_over_freq_0p5Hz'
save_figure_all_formats(plt.gcf(), out_dir, basename, dpi=800)
plt.show()
plt.close()

print(f"[DONE] Used {len(right_tokens)} right-sensor tokens; "
      f"loaded {aligned.shape[1]} pair spectra; "
      f"saved figures to: {out_dir}")