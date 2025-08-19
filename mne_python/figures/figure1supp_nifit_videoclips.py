"""
==============================================
figure1supp_nifit_videoclips

This script takes the 10 NIfTI files, 
makes one video per file (both .mp4 and .gif), 
shows every axial slice, and 
saves the outputs next to each NIfTI.

It avoids piling all frames into memory and writes 
frames streaming‑style. 
It also normalizes each slice to 8‑bit and rotates for a 
viewer‑friendly orientation.

Tri-view NIfTI → MP4 + GIF
 - Coronal (Anterior→Posterior), Axial (Inferior→Superior), Sagittal (Right→Left)
 - Uses MRIcroGL's ACTC colormap (.clut)
 - Overlays slice index + world coords
 - Robust intensity window and HiDPI-safe rendering


written by Tara Ghafari
last updated 27/06/2025
tara.ghafari@gmail.com
==============================================
"""
import os
import numpy as np
import nibabel as nib
import imageio
import matplotlib
import matplotlib.pyplot as plt
from nibabel.orientations import aff2axcodes

# --------- CONFIG ---------
FIG_SAVE_PATH = "/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures/Figure4"
REL_NIFTI_PATHS = [
    "Caud_Alpha_mag_source_correlations/Caud_Alpha_mag_src-substr-correlation.nii.gz",
    "Caud_Beta_grad_source_correlations/Caud_Beta_grad_src-substr-correlation.nii.gz",
    "Caud_Beta_mag_source_correlations/Caud_Beta_mag_src-substr-correlation.nii.gz",
    "Caud_Delta_grad_source_correlations/Caud_Delta_grad_src-substr-correlation.nii.gz",
    "Caud_Delta_mag_source_correlations/Caud_Delta_mag_src-substr-correlation.nii.gz",
    "Caud_Theta_mag_source_correlations/Caud_Theta_mag_src-substr-correlation.nii.gz",
    "Hipp_Beta_mag_source_correlations/Hipp_Beta_mag_src-substr-correlation.nii.gz",
    "Hipp_Delta_grad_source_correlations/Hipp_Delta_grad_src-substr-correlation.nii.gz",
    "Pall_Alpha_grad_source_correlations/Pall_Alpha_grad_src-substr-correlation.nii.gz",
    "Puta_Beta_grad_source_correlations/Puta_Beta_grad_src-substr-correlation.nii.gz",
]


# MRIcroGL ACTC LUT path
ACTC_CLUT_PATH = "/Applications/MRIcroGL.app/Contents/Resources/lut/actc.clut"

# Timing: each slice displayed for 3 seconds
SECONDS_PER_SLICE = 0.2
FPS = 1.0 / SECONDS_PER_SLICE

# Figure layout (3 panels; increase for larger frames)
FIGURE_INCHES = (11, 4)
FIGURE_DPI = 140

# Optional display rotation for each 2D slice (does not affect axes/coords). Usually leave 0.
ROTATE_K = 0  # 0,1,2,3 -> 0°, 90°, 180°, 270° counter-clockwise

# ---------------- HELPERS ----------------
import re

def load_mricrogl_clut(path, target_len=256):
    """
    Robustly load an MRIcroGL .clut/.lut file with various header formats.
    Accepts lines like:
      [FLT]
      min=0
      max=255
      ncol=256
      0=255 0 0
      1=254 1 0
      ...
    or even bare "R G B [A]" rows.
    """
    # Read text robustly
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "latin-1", "mac_roman"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode LUT file with common encodings")

    rows = []
    idx_rgba = {}  # if lines have an explicit index

    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            # section header like [FLT]
            continue
        if s.startswith("#") or s.startswith(";"):
            # comment lines
            continue

        # Pull out ANY numbers on the line (ints or floats)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(nums) < 3:
            # metadata like "min=0" yields only 1 number
            continue

        # Convert to floats
        vals = [float(x) for x in nums]

        # Heuristic: if 4+ numbers AND the first is a small integer index, treat it as index + RGB[A]
        if len(vals) >= 4 and vals[0].is_integer() and 0 <= vals[0] <= 4096:
            idx = int(vals[0])
            rgb = vals[1:4]
            alpha = vals[4] if len(vals) >= 5 else 255.0
            idx_rgba[idx] = np.array([*rgb, alpha], dtype=float)
        else:
            # Bare RGB[A]
            rgb = vals[:3]
            alpha = vals[3] if len(vals) >= 4 else 255.0
            rows.append(np.array([*rgb, alpha], dtype=float))

    if idx_rgba:
        # Build from explicit indices
        max_idx = max(idx_rgba.keys())
        min_idx = min(idx_rgba.keys())
        N = max_idx - min_idx + 1
        lut = np.zeros((N, 4), dtype=float)
        for i in range(N):
            lut[i] = idx_rgba.get(i + min_idx, lut[i-1] if i > 0 else [0, 0, 0, 255])
    elif rows:
        lut = np.vstack(rows)
    else:
        raise ValueError(f"No numeric RGB(A) rows found in {path}")

    # Scale 0–255 → 0–1 if needed
    if lut.max() > 1.0:
        lut[:, :3] = np.clip(lut[:, :3], 0, 255) / 255.0
        lut[:, 3] = np.clip(lut[:, 3], 0, 255) / 255.0

    # Ensure we have RGBA (alpha=1 if missing)
    if lut.shape[1] == 3:
        lut = np.hstack([lut, np.ones((lut.shape[0], 1), dtype=float)])

    # Resample to target_len smoothly if needed
    if lut.shape[0] != target_len:
        x = np.linspace(0, 1, lut.shape[0])
        xi = np.linspace(0, 1, target_len)
        lut_res = np.zeros((target_len, 4), dtype=float)
        for c in range(4):
            lut_res[:, c] = np.interp(xi, x, lut[:, c])
        lut = lut_res

    # Build the colormap (matplotlib uses RGB; alpha kept internally)
    return matplotlib.colors.ListedColormap(lut[:, :3], name="MRIcroGL_ACTC")


def compute_global_window(vol):
    """Robust vmin/vmax from non-zero finite voxels; avoids blank frames."""
    flat = vol[np.isfinite(vol)]
    flat = flat[flat != 0]
    if flat.size == 0:
        flat = vol[np.isfinite(vol)]
    if flat.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(flat, [2, 98])
    if not np.isfinite(vmin): vmin = np.nanmin(flat)
    if not np.isfinite(vmax): vmax = np.nanmax(flat)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def slice_center_world_xyz(affine, vol_shape, axis, index):
    """
    Return world (x,y,z) in mm for the center of the slice plane along 'axis' at 'index'.
    Uses voxel center convention (i+0.5, j+0.5, k+0.5).
    """
    nx, ny, nz = vol_shape[:3]
    i = (nx / 2.0) - 0.5
    j = (ny / 2.0) - 0.5
    k = (nz / 2.0) - 0.5
    if axis == 0:
        i = index + 0.5
    elif axis == 1:
        j = index + 0.5
    else:
        k = index + 0.5
    voxel = np.array([i, j, k, 1.0], dtype=float)
    world = affine @ voxel
    return float(world[0]), float(world[1]), float(world[2])


def order_indices_for_direction(n, axis_code, desired_start, desired_end):
    """
    Given axis code from aff2axcodes (R/L, A/P, S/I), pick ascending or descending
    order so indices travel desired_start -> desired_end.
    """
    if axis_code == desired_end:
        return list(range(n))             # ascending indices move desired_start->desired_end
    elif axis_code == desired_start:
        return list(range(n - 1, -1, -1)) # descending indices move desired_start->desired_end
    else:
        # Unexpected/oblique labeling: default ascending
        return list(range(n))


def extract_slice(vol, axis, index):
    if axis == 0:
        sl = vol[index, :, :]
    elif axis == 1:
        sl = vol[:, index, :]
    else:
        sl = vol[:, :, index]
    if ROTATE_K:
        sl = np.rot90(sl, k=ROTATE_K)
    return np.nan_to_num(sl, nan=0.0)


def render_triview_frame(vol, affine, vmin, vmax, cmap, t, orders):
    """
    Build a tri-view RGB frame (Coronal A→P, Axial I→S, Sagittal R→L) for time index t.
    """
    # Safe index per axis (clamp for unequal axis lengths)
    i_cor = orders[1][min(t, len(orders[1]) - 1)]  # axis=1
    i_axi = orders[2][min(t, len(orders[2]) - 1)]  # axis=2
    i_sag = orders[0][min(t, len(orders[0]) - 1)]  # axis=0

    cor = extract_slice(vol, 1, i_cor)
    axi = extract_slice(vol, 2, i_axi)
    sag = extract_slice(vol, 0, i_sag)

    fig = plt.figure(figsize=FIGURE_INCHES, dpi=FIGURE_DPI)
    gs = fig.add_gridspec(1, 3, left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    titles = [
        "Coronal (Anterior → Posterior)",
        "Axial (Inferior → Superior)",
        "Sagittal (Right → Left)"
    ]

    for ax, sl, tt in zip(axs, [cor, axi, sag], titles):
        ax.axis('off')
        ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(tt, fontsize=11, pad=2)

    # Slice-center labels
    x_cor, y_cor, z_cor = slice_center_world_xyz(affine, vol.shape, 1, i_cor)
    x_axi, y_axi, z_axi = slice_center_world_xyz(affine, vol.shape, 2, i_axi)
    x_sag, y_sag, z_sag = slice_center_world_xyz(affine, vol.shape, 0, i_sag)

    axs[0].text(0.02, 0.02, f"i={i_cor}  center (mm): x={x_cor:.1f}, y={y_cor:.1f}, z={z_cor:.1f}",
                transform=axs[0].transAxes, color='w', fontsize=9,
                bbox=dict(facecolor='k', alpha=0.3, pad=2))
    axs[1].text(0.02, 0.02, f"i={i_axi}  center (mm): x={x_axi:.1f}, y={y_axi:.1f}, z={z_axi:.1f}",
                transform=axs[1].transAxes, color='w', fontsize=9,
                bbox=dict(facecolor='k', alpha=0.3, pad=2))
    axs[2].text(0.02, 0.02, f"i={i_sag}  center (mm): x={x_sag:.1f}, y={y_sag:.1f}, z={z_sag:.1f}",
                transform=axs[2].transAxes, color='w', fontsize=9,
                bbox=dict(facecolor='k', alpha=0.3, pad=2))

    fig.suptitle("Orthogonal slices • A→P (Cor), I→S (Ax), R→L (Sag)", fontsize=13, y=0.97)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4) uint8, HiDPI-safe
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb
# ------------------------------------------------------


# ---------------------- MAIN MAKER ----------------------
def make_triview_video(nifti_path, cmap):
    out_dir = os.path.dirname(nifti_path)
    img = nib.load(nifti_path)
    vol = img.get_fdata()
    aff = img.affine

    # If 4D, use first volume (customize as needed)
    if vol.ndim == 4:
        vol = vol[..., 0]

    vmin, vmax = compute_global_window(vol)
    axcodes = aff2axcodes(aff)  # e.g., ('R','A','S')

    n0, n1, n2 = vol.shape[:3]
    # Axis orders to enforce desired directions
    orders = {
        0: order_indices_for_direction(n0, axcodes[0], 'R', 'L'),  # Sagittal R->L
        1: order_indices_for_direction(n1, axcodes[1], 'A', 'P'),  # Coronal  A->P
        2: order_indices_for_direction(n2, axcodes[2], 'I', 'S'),  # Axial    I->S
    }
    n_frames = max(len(orders[0]), len(orders[1]), len(orders[2]))

    base = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]
    mp4_out = os.path.join(out_dir, f"{base}_tri_AP_IS_RL_actc.mp4")
    gif_out = os.path.join(out_dir, f"{base}_tri_AP_IS_RL_actc.gif")

    mp4_writer = imageio.get_writer(
        mp4_out, fps=FPS, codec="libx264", quality=8,
        ffmpeg_params=["-r", f"{FPS}"]  # explicit output rate helps avoid ffmpeg warning
    )
    gif_writer = imageio.get_writer(gif_out, mode="I", duration=SECONDS_PER_SLICE)

    num_frames = 0
    try:
        for t in range(n_frames):
            frame = render_triview_frame(vol, aff, vmin, vmax, cmap, t, orders)
            mp4_writer.append_data(frame)
            gif_writer.append_data(frame)
            num_frames += 1

        # Ensure at least 2 frames (silences ffmpeg "not enough frames" warning on very small volumes)
        if num_frames == 1:
            mp4_writer.append_data(frame)
            gif_writer.append_data(frame)
            num_frames += 1
    finally:
        mp4_writer.close()
        gif_writer.close()

    print(f"[DONE] {num_frames} frames → {mp4_out}")
    print(f"[DONE] {num_frames} frames → {gif_out}")


# ---------------------- RUN ----------------------
if __name__ == "__main__":
    # Load the ACTC colormap
    actc_cmap = load_mricrogl_clut(ACTC_CLUT_PATH)

    # Process each file
    for rel in REL_NIFTI_PATHS:
        path = os.path.join(FIG_SAVE_PATH, rel)
        if os.path.isfile(path):
            print(f"[INFO] Processing {path}")
            make_triview_video(path, actc_cmap)
        else:
            print(f"[WARN] Missing file: {path}")