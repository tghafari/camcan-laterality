"""
==============================================
figure1supp_nifit_videoclips

This script takes the 10 NIfTI files, 
makes one video per file (both .mp4 and .gif), 
shows every axial slice for 3 seconds, and 
saves the outputs next to each NIfTI.

It avoids piling all frames into memory and writes 
frames streaming‑style. 
It also normalizes each slice to 8‑bit and rotates for a 
viewer‑friendly orientation.


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
fig_save_path = "/Users/t.ghafari@bham.ac.uk/Library/CloudStorage/OneDrive-UniversityofBirmingham/Desktop/BEAR_outage/CamCAN-results/Manuscript/Figures/Figure4"
rel_nifti_paths = [
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

# Slicing / display
slice_axis = 2           # 0 = sagittal, 1 = coronal, 2 = axial
seconds_per_slice = 0.5  # 0.5 seconds per slice
fps = 1.0 / seconds_per_slice
rotate_k = 1             # visual 90° CCW rotation of each slice; set 0 to disable

# Figure size / DPI
figure_inches = (7, 7)
figure_dpi = 120

# MRIcroGL ACTC colormap path
actc_clut_path = "/Applications/MRIcroGL.app/Contents/Resources/lut/actc.clut"

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



def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    aff = img.affine
    return data, aff

def compute_global_window(vol):
    """Robust global vmin/vmax from non-zero finite voxels (prevents blank frames)."""
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

def iter_slices(vol, axis):
    n = vol.shape[axis]
    for i in range(n):
        if axis == 0:
            yield vol[i, :, :], i
        elif axis == 1:
            yield vol[:, i, :], i
        else:
            yield vol[:, :, i], i

def slice_center_world_xyz(affine, vol_shape, axis, index):
    """
    World (x,y,z) in mm for the center of the slice plane at given axis/index.
    Uses voxel center (i+0.5, j+0.5, k+0.5).
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
    voxel = np.array([i, j, k, 1.0])
    world = affine @ voxel
    return float(world[0]), float(world[1]), float(world[2])

def axis_name_and_direction(affine, axis):
    """
    Returns ("Sagittal"/"Coronal"/"Axial", "Left→Right"/"Posterior→Anterior"/"Inferior→Superior", etc.)
    based on aff2axcodes, which tells us the world direction of increasing voxel index.
    """
    axcodes = aff2axcodes(affine)
    names = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    dir_code = axcodes[axis]  # e.g., 'R','L','A','P','S','I'
    # Map code to a From→To description (movement as index increases)
    mapping = {
        'R': "Left→Right",
        'L': "Right→Left",
        'A': "Posterior→Anterior",
        'P': "Anterior→Posterior",
        'S': "Inferior→Superior",
        'I': "Superior→Inferior",
    }
    direction = mapping.get(dir_code, f"{dir_code} (increasing index)")
    return names[axis], direction


def render_frame(img2d, vmin, vmax, cmap, title, subtitle,
                 figure_inches=(7,7), figure_dpi=120):
    """Render a 2D slice to an RGB uint8 array using matplotlib, robust to HiDPI canvases."""
    fig = plt.figure(figsize=figure_inches, dpi=figure_dpi)
    # leave space at top for title/subtitle
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.94])
    ax.axis("off")
    ax.imshow(img2d, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.text(0.5, 0.98, title, ha="center", va="top", fontsize=14, weight="bold")
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=11)

    fig.canvas.draw()
    # Get the actual pixel buffer at the canvas backing resolution
    buf = np.asarray(fig.canvas.buffer_rgba())  # shape (H, W, 4), dtype=uint8
    rgb = buf[:, :, :3].copy()                  # drop alpha
    plt.close(fig)
    return rgb

# ---------------- LOAD COLORMAP ----------------
actc_cmap = load_mricrogl_clut(actc_clut_path)

# ---------------- MAIN ----------------
for rel_path in rel_nifti_paths:
    nifti_path = os.path.join(fig_save_path, rel_path)
    out_dir = os.path.dirname(nifti_path)

    if not os.path.isfile(nifti_path):
        print(f"[WARN] Missing file: {nifti_path}")
        continue

    print(f"[INFO] Processing {nifti_path}")
    vol, affine = load_nifti(nifti_path)

    # Handle 4D by taking first volume (customize if needed)
    if vol.ndim == 4:
        vol = vol[..., 0]

    # Robust window across whole volume so frames aren't empty
    vmin, vmax = compute_global_window(vol)

    # Axis name and direction (applies to slice order 0..N-1)
    axis_name, direction = axis_name_and_direction(affine, slice_axis)
    fixed_title = f"{axis_name} slices ({direction})"

    # Set outputs
    base = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]
    mp4_out = os.path.join(out_dir, f"{base}_slices_actc.mp4")
    gif_out = os.path.join(out_dir, f"{base}_slices_actc.gif")

    # Writers
    mp4_writer = imageio.get_writer(mp4_out, fps=fps, codec="libx264", quality=8)
    gif_writer = imageio.get_writer(gif_out, mode="I", duration=seconds_per_slice)

    num_frames = 0
    try:
        for sl, idx in iter_slices(vol, slice_axis):
            # Rotate purely for display (does not affect world coords)
            if rotate_k:
                sl = np.rot90(sl, k=rotate_k)
            sl = np.nan_to_num(sl, nan=0.0)

            # World (x,y,z) at slice center
            x_mm, y_mm, z_mm = slice_center_world_xyz(affine, vol.shape, slice_axis, idx)
            subtitle = f"Slice {idx} • Center (mm): x={x_mm:.1f}, y={y_mm:.1f}, z={z_mm:.1f}"

            # Render with ACTC colormap and overlays
            frame_rgb = render_frame(sl, vmin, vmax, actc_cmap, fixed_title, subtitle)

            mp4_writer.append_data(frame_rgb)
            gif_writer.append_data(frame_rgb)
            num_frames += 1
    finally:
        mp4_writer.close()
        gif_writer.close()

    print(f"[DONE] {num_frames} frames → {mp4_out}")
    print(f"[DONE] {num_frames} frames → {gif_out}")