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
import re
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
# 0,1,2,3 -> 0°, 90°, 180°, 270° counter-clockwise

# Set 0 (no rotation); we handle orientation via RAS + controlled flips below.
ROTATE_K = 0
# ----------------------------------------------------


# ---------------------- UTILITIES ----------------------
def load_mricrogl_clut(path, target_len=256):
    """Robust MRIcroGL .clut loader (handles headers, indexed rows, RGBA)."""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "latin-1", "mac_roman"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not decode LUT file")

    idx_rgba, rows = {}, []
    for ln in text.splitlines():
        s = ln.strip()
        if not s or (s.startswith("[") and s.endswith("]")) or s.startswith("#") or s.startswith(";"):
            continue
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(nums) < 3:
            continue
        vals = [float(x) for x in nums]
        if len(vals) >= 4 and float(vals[0]).is_integer() and 0 <= vals[0] <= 65535:
            idx = int(vals[0]); rgb = vals[1:4]; alpha = vals[4] if len(vals) >= 5 else 255.0
            idx_rgba[idx] = np.array([*rgb, alpha], dtype=float)
        else:
            rgb = vals[:3]; alpha = vals[3] if len(vals) >= 4 else 255.0
            rows.append(np.array([*rgb, alpha], dtype=float))

    if idx_rgba:
        max_idx, min_idx = max(idx_rgba), min(idx_rgba)
        N = max_idx - min_idx + 1
        lut = np.zeros((N, 4), dtype=float)
        for i in range(N):
            lut[i] = idx_rgba.get(i + min_idx, lut[i-1] if i > 0 else [0,0,0,255])
    elif rows:
        lut = np.vstack(rows)
    else:
        raise ValueError(f"No numeric RGB(A) rows found in {path}")

    if lut.max() > 1.0:
        lut[:, :3] = np.clip(lut[:, :3], 0, 255) / 255.0
        lut[:, 3]  = np.clip(lut[:, 3],  0, 255) / 255.0

    if lut.shape[1] == 3:
        lut = np.hstack([lut, np.ones((lut.shape[0], 1), dtype=float)])

    if lut.shape[0] != target_len:
        x  = np.linspace(0, 1, lut.shape[0])
        xi = np.linspace(0, 1, target_len)
        lut_res = np.zeros((target_len, 4), dtype=float)
        for c in range(4):
            lut_res[:, c] = np.interp(xi, x, lut[:, c])
        lut = lut_res

    return matplotlib.colors.ListedColormap(lut[:, :3], name="MRIcroGL_ACTC")


def compute_global_window(vol):
    """Robust vmin/vmax from non-zero finite voxels."""
    flat = vol[np.isfinite(vol)]; flat = flat[flat != 0]
    if flat.size == 0: flat = vol[np.isfinite(vol)]
    if flat.size == 0: return 0.0, 1.0
    vmin, vmax = np.percentile(flat, [2, 98])
    if not np.isfinite(vmin): vmin = np.nanmin(flat)
    if not np.isfinite(vmax): vmax = np.nanmax(flat)
    if vmax <= vmin: vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def to_RAS(img):
    """
    Return (data_RAS, affine_RAS) by reorienting with nib.as_closest_canonical.
    This ensures axes are X=Right, Y=Anterior, Z=Superior as indices increase.
    """
    can = nib.as_closest_canonical(img)
    return can.get_fdata(), can.affine


def slice_center_world_xyz(affine, vol_shape, i, j, k):
    """World (x,y,z) of voxel center at indices (i,j,k) in the (reoriented) volume."""
    voxel = np.array([i + 0.5, j + 0.5, k + 0.5, 1.0], dtype=float)
    x, y, z, _ = affine @ voxel
    return float(x), float(y), float(z)


# ---------- SLICE EXTRACTORS (display-ready, with guaranteed orientation) ----------
# After RAS reorientation:
#   Axial  plane (Z = k): data[k: constant] has in-plane X (R↔L) horizontally, Y (A↔P) vertically.
#   Coronal plane (Y = j): we want vertical = Z (S up), horizontal = X (R right).
#   Sagittal plane (X = i): we want vertical = Z (S up), horizontal = Y (A right).
# We’ll flip vertically for axial so TOP=Anterior (best possible; Superior is out-of-plane).
def get_axial_slice_RAS(data, k):
    sl = data[:, :, k]              # shape (X, Y)
    if ROTATE_K: sl = np.rot90(sl, k=ROTATE_K)
    sl = np.nan_to_num(sl, nan=0.0)
    # Make TOP = Anterior: Y increases Anterior with index; imshow shows row 0 at TOP.
    # So we flip vertically if row 0 corresponds to Posterior.
    # In RAS, Y increases A; row index increases downward → downward = A → TOP = P → flip:
    sl = np.flipud(sl.T)            # transpose so vertical is Y, then flip so top=A
    # Now: vertical up = Anterior, horizontal right = Right
    return sl

def get_coronal_slice_RAS(data, j):
    sl = data[:, j, :]              # shape (X, Z)
    if ROTATE_K: sl = np.rot90(sl, k=ROTATE_K)
    sl = np.nan_to_num(sl, nan=0.0)
    # We want vertical = Z with TOP = Superior, horizontal = X with RIGHT = Right
    sl = sl.T                       # (Z, X): rows = Z, cols = X
    # In RAS, Z increases Superior; row index increases downward → bottom = Superior → flip:
    sl = np.flipud(sl)
    # Columns already increase X→Right, so right side is Right
    return sl

def get_sagittal_slice_RAS(data, i):
    sl = data[i, :, :]              # shape (Y, Z)
    if ROTATE_K: sl = np.rot90(sl, k=ROTATE_K)
    sl = np.nan_to_num(sl, nan=0.0)
    # We want vertical = Z with TOP = Superior, horizontal = Y with RIGHT = Anterior
    sl = sl.T                       # (Z, Y): rows = Z (S up), cols = Y (A right)
    # In RAS, Z increases Superior; row index increases downward → bottom = Superior → flip:
    sl = np.flipud(sl)
    # Columns increase Y→Anterior; right side is Anterior (expected for sagittal)
    return sl


def render_triview_frame_RAS(data_RAS, affine_RAS, vmin, vmax, cmap, i, j, k):
    cor = get_coronal_slice_RAS(data_RAS, j)
    axi = get_axial_slice_RAS(data_RAS, k)
    sag = get_sagittal_slice_RAS(data_RAS, i)

    x_mm, y_mm, z_mm = slice_center_world_xyz(affine_RAS, data_RAS.shape, i, j, k)

    fig = plt.figure(figsize=FIGURE_INCHES, dpi=FIGURE_DPI)
    gs = fig.add_gridspec(1, 3, left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    titles = [
        "Coronal (Superior ↑, Right →)",
        "Axial (Anterior ↑, Right →)",
        "Sagittal (Superior ↑, Anterior →)"
    ]
    for ax, sl, tt in zip(axs, [cor, axi, sag], titles):
        ax.axis('off')
        ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(tt, fontsize=11, pad=2)

    # --- Add orientation labels ---
    # Left/right edges and top
    for ax in axs:
        ax.text(0.02, 0.5, "L", color='w', fontsize=10, ha='left', va='center',
                transform=ax.transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
        ax.text(0.98, 0.5, "R", color='w', fontsize=10, ha='right', va='center',
                transform=ax.transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
        ax.text(0.5, 0.02, "Bottom", color='w', fontsize=8, ha='center', va='bottom',
                transform=ax.transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
        ax.text(0.5, 0.98, "Top", color='w', fontsize=10, ha='center', va='top',
                transform=ax.transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))

    subtitle = f"Voxel indices (i={i}, j={j}, k={k}) • Center (mm): x={x_mm:.1f}, y={y_mm:.1f}, z={z_mm:.1f}"
    fig.suptitle(f"Orthogonal slices at SAME coords • {subtitle}", fontsize=12, y=0.97)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb
# ------------------------------------------------------


# ---------------------- MAIN MAKER ----------------------
def render_triview_frame_RAS(data_RAS, affine_RAS, vmin, vmax, cmap, i, j, k):
    """
    Build a tri-view frame from RAS data.
    Panels:
      Left   = Coronal  (A→P playback), display: Superior ↑, Right →
      Center = Axial    (S→I playback), display: Anterior ↑, Right →
      Right  = Sagittal (R→L playback), display: Superior ↑, Anterior →
    """
    cor = get_coronal_slice_RAS(data_RAS, j)   # (Z up, X right)
    axi = get_axial_slice_RAS(data_RAS, k)     # (A up, R right)
    sag = get_sagittal_slice_RAS(data_RAS, i)  # (Z up, A right)

    # Figure with 3 panels
    fig = plt.figure(figsize=FIGURE_INCHES, dpi=FIGURE_DPI)
    gs = fig.add_gridspec(1, 3, left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.03)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    titles = [
        "Coronal (A→P) • Superior ↑, Right →",
        "Axial (S→I) • Anterior ↑, Right →",
        "Sagittal (R→L) • Superior ↑, Anterior →",
    ]
    for ax, sl, tt in zip(axs, [cor, axi, sag], titles):
        ax.axis('off')
        ax.imshow(sl, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(tt, fontsize=11, pad=2)

    # Orientation labels per panel
    # Coronal: left=L, right=R, top=Superior, bottom=Inferior (we just label Top/Bottom text)
    axs[0].text(0.02, 0.5, "L", color='w', fontsize=10, ha='left',  va='center',
                transform=axs[0].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[0].text(0.98, 0.5, "R", color='w', fontsize=10, ha='right', va='center',
                transform=axs[0].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[0].text(0.5, 0.98, "Top",    color='w', fontsize=10, ha='center', va='top',
                transform=axs[0].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[0].text(0.5, 0.02, "Bottom", color='w', fontsize=10, ha='center', va='bottom',
                transform=axs[0].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))

    # Axial: left=L, right=R, top=Anterior, bottom=Posterior (we keep generic Top/Bottom text)
    axs[1].text(0.02, 0.5, "L", color='w', fontsize=10, ha='left',  va='center',
                transform=axs[1].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[1].text(0.98, 0.5, "R", color='w', fontsize=10, ha='right', va='center',
                transform=axs[1].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[1].text(0.5, 0.98, "Top",    color='w', fontsize=10, ha='center', va='top',
                transform=axs[1].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[1].text(0.5, 0.02, "Bottom", color='w', fontsize=10, ha='center', va='bottom',
                transform=axs[1].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))

    # Sagittal: horizontal is A↔P, so label A/P instead of R/L to avoid confusion
    axs[2].text(0.02, 0.5, "P", color='w', fontsize=10, ha='left',  va='center',
                transform=axs[2].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[2].text(0.98, 0.5, "A", color='w', fontsize=10, ha='right', va='center',
                transform=axs[2].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[2].text(0.5, 0.98, "Top",    color='w', fontsize=10, ha='center', va='top',
                transform=axs[2].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))
    axs[2].text(0.5, 0.02, "Bottom", color='w', fontsize=10, ha='center', va='bottom',
                transform=axs[2].transAxes, bbox=dict(facecolor='k', alpha=0.3, pad=1))

    # World coords (center of the three planes’ intersection)
    x_mm, y_mm, z_mm = slice_center_world_xyz(affine_RAS, data_RAS.shape, i, j, k)
    fig.suptitle(
        f"Tri‑view (A→P | S→I | R→L) • voxel (i={i}, j={j}, k={k}) • center mm: x={x_mm:.1f}, y={y_mm:.1f}, z={z_mm:.1f}",
        fontsize=12, y=0.97
    )

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    rgb = buf[:, :, :3].copy()
    plt.close(fig)
    return rgb


def make_triview_video_AP_SI_RL(nifti_path, cmap):
    """
    Reorients to RAS; animates:
      - Left  panel (Coronal):  A → P  (decreasing j)
      - Center panel (Axial):   S → I  (decreasing k)
      - Right panel (Sagittal): R → L  (decreasing i)
    """
    out_dir = os.path.dirname(nifti_path)
    img = nib.load(nifti_path)

    # Reorient to RAS for consistent axes
    data_RAS, affine_RAS = to_RAS(img)
    if data_RAS.ndim == 4:
        data_RAS = data_RAS[..., 0]

    vmin, vmax = compute_global_window(data_RAS)
    nx, ny, nz = data_RAS.shape[:3]

    # Build per‑panel index orders:
    # RAS means +x=Right, +y=Anterior, +z=Superior.
    # So:
    #   A→P  is high y → low y  -> j = ny-1 .. 0
    #   S→I  is high z → low z  -> k = nz-1 .. 0
    #   R→L  is high x → low x  -> i = nx-1 .. 0
    i_order = list(range(nx-1, -1, -1))  # R -> L
    j_order = list(range(ny-1, -1, -1))  # A -> P
    k_order = list(range(nz-1, -1, -1))  # S -> I

    n_frames = max(len(i_order), len(j_order), len(k_order))

    base = os.path.splitext(os.path.splitext(os.path.basename(nifti_path))[0])[0]
    mp4_out = os.path.join(out_dir, f"{base}_tri_AP_SI_RL_actc.mp4")
    gif_out = os.path.join(out_dir, f"{base}_tri_AP_SI_RL_actc.gif")

    mp4_writer = imageio.get_writer(
        mp4_out, fps=FPS, codec="libx264", quality=8,
        ffmpeg_params=["-r", f"{FPS}"]
    )
    gif_writer = imageio.get_writer(gif_out, mode="I", duration=SECONDS_PER_SLICE)

    num_frames = 0
    try:
        for t in range(n_frames):
            # Clamp each order to its last index if one axis runs out sooner
            i = i_order[min(t, len(i_order)-1)]
            j = j_order[min(t, len(j_order)-1)]
            k = k_order[min(t, len(k_order)-1)]

            frame = render_triview_frame_RAS(data_RAS, affine_RAS, vmin, vmax, cmap, i, j, k)
            mp4_writer.append_data(frame)
            gif_writer.append_data(frame)
            num_frames += 1

        # Ensure at least 2 frames (avoid ffmpeg rate warning on tiny volumes)
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
    actc_cmap = load_mricrogl_clut(ACTC_CLUT_PATH)

    for rel in REL_NIFTI_PATHS:
        path = os.path.join(FIG_SAVE_PATH, rel)
        if os.path.isfile(path):
            print(f"[INFO] Processing {path}")
            make_triview_video_AP_SI_RL(path, actc_cmap)
        else:
            print(f"[WARN] Missing file: {path}")