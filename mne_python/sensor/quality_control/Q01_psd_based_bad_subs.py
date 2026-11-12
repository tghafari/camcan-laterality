# -*- coding: utf-8 -*-
"""
================================================================
Q01: Diagnose the subjects lost during S01 PSD/lateralisation processing.

Reads the lost-subject CSV from the mounted quinna-camcan share, runs the
same Welch PSD and channel-pair checks as S01, and records the first failure
per subject with a full traceback. Does not overwrite S01 outputs.

Run:
    python mne_python/sensor/quality_control/Q01_psd_based_bad_subs.py --platform bluebear
    python mne_python/sensor/quality_control/Q01_psd_based_bad_subs.py --platform mac

The smb:// URL must be mounted as a local filesystem path first.
============================================================
"""
import argparse
import logging
import os
import os.path as op
import traceback

import mne
import numpy as np
import pandas as pd


def setup_paths(platform):
    if platform == "bluebear":
        quinna_dir = "/rds/projects/q/quinna-camcan"
        sub2ctx_dir = "/rds/projects/j/jenseno-sub2ctx/camcan"
    elif platform == "mac":
        quinna_dir = "/Volumes/quinna-camcan"
        sub2ctx_dir = "/Volumes/jenseno-sub2ctx/camcan"
    else:
        raise ValueError("platform must be 'mac' or 'bluebear'")
    info_dir = op.join(quinna_dir, "dataman", "data_information")
    return {
        "lost_csv": op.join(info_dir, "subjects_lost_after_S01_psd_lateralisation.csv"),
        "layout_csv": op.join(info_dir, "combined_sensors_layout_names.csv"),
        "epochs_dir": op.join(sub2ctx_dir, "derivatives", "meg", "sensor", "epoched-7min50"),
        "output_dir": op.join(sub2ctx_dir, "derivatives", "meg", "sensor", "quality_control", "Q01_psd_based_bad_subs"),
    }


def load_lost_ids(path):
    """Support subject_ID, SubjectID, subjectID, and S01's unnamed CSV column."""
    df = pd.read_csv(path, dtype=str)
    candidates = [c for c in df if c.strip().lower().replace("_", "") in
                  {"subjectid", "subject", "subid"}]
    if candidates:
        col = candidates[0]
    else:
        non_index = [c for c in df if not c.lower().startswith("unnamed")]
        if len(non_index) != 1:
            raise ValueError(f"Cannot identify subject-ID column in {path}: {list(df.columns)}")
        col = non_index[0]
    ids = df[col].dropna().astype(str).str.strip()
    ids = ids[ids.ne("") & ids.ne("nan")]
    ids = ids.str.replace(r"\\.0$", "", regex=True)
    if ids.duplicated().any():
        logging.warning("Duplicate subject IDs in lost-subject CSV; processing once.")
    return ids.drop_duplicates().tolist()


def check_subject(subject_id, paths, layout, n_fft=500):
    """Run S01 PSD and channel pairing, recording stage-specific failures."""
    epoch_path = op.join(
        paths["epochs_dir"],
        f"sub-CC{subject_id}_ses-rest_task-rest_megtransdef_epo.fif",
    )
    stage = "check_epochs_file"
    metrics = {"SubjectID": subject_id, "epochs_file": epoch_path}
    try:
        if not op.isdir(paths["epochs_dir"]):
            raise NotADirectoryError(
                f"Epochs directory is inaccessible or does not exist: "
                f"{paths['epochs_dir']}. Check the SMB mount."
            )

        if not op.isfile(epoch_path):
            raise FileNotFoundError(
                f"Subject FIF file not found at: {epoch_path}. "
                f"The epochs directory is accessible, but this exact filename "
                f"was not found."
            )

        stage = "read_epochs"
        epochs = mne.read_epochs(epoch_path, preload=True, verbose="ERROR")
        metrics["sfreq"] = float(epochs.info["sfreq"])
        metrics["n_epochs"] = len(epochs)
        metrics["n_channels"] = len(epochs.ch_names)
        metrics["n_times"] = len(epochs.times)
        if len(epochs) == 0:
            raise ValueError("No epochs available")
        if len(epochs.times) < n_fft:
            raise ValueError(f"Epoch length {len(epochs.times)} < n_fft {n_fft}")

        stage = "welch_psd"
        spectrum = epochs.compute_psd(
            method="welch", fmin=1, fmax=120, picks="meg",
            n_fft=n_fft, n_overlap=n_fft // 2, window="hamming",
            n_jobs=1, verbose="ERROR",
        )
        psd, freqs = spectrum.get_data(return_freqs=True)
        metrics["psd_shape"] = str(psd.shape)
        metrics["n_nonfinite_psd"] = int(np.size(psd) - np.isfinite(psd).sum())
        if not np.isfinite(psd).all():
            raise ValueError("PSD contains NaN/Inf")
        ch_names = spectrum.ch_names
        lookup = {name: pos for pos, name in enumerate(ch_names)}

        stage = "combine_grad_pairs"
        combined = {}
        for name in ch_names:
            if name.endswith("2") and name[:-1] + "3" in lookup:
                partner = name[:-1] + "3"
                combined[name] = (psd[:, lookup[name], :] +
                                  psd[:, lookup[partner], :]) / 2.0
        metrics["n_combined_grad_pairs"] = len(combined)

        stage = "sensor_pair_lateralisation"
        n_checked = 0
        for row_idx, row in layout.iterrows():
            right, left = str(row["right_sensors"])[:8], str(row["left_sensors"])[:8]
            stage = f"sensor_pair_row_{row_idx}_{right}_{left}"
            if right[-1:] != left[-1:]:
                raise ValueError(f"Right/left channel types differ: {right}, {left}")
            if right.endswith("3"):
                continue  # S01 combines these into the matching '2' pair
            if right.endswith("1"):
                missing = [ch for ch in (right, left) if ch not in lookup]
                if missing:
                    raise KeyError(f"Missing magnetometer(s): {missing}")
                right_psd = psd[:, lookup[right], :]
                left_psd = psd[:, lookup[left], :]
            elif right.endswith("2"):
                missing = [ch for ch in (right, left) if ch not in combined]
                if missing:
                    raise KeyError(f"Missing combined gradiometer pair(s): {missing}")
                right_psd, left_psd = combined[right], combined[left]
            else:
                raise ValueError(f"Unsupported sensor type: {right}, {left}")

            lateralisation = np.squeeze(right_psd - left_psd)
            noise_mask = (freqs >= 90) & (freqs <= 120)
            if not noise_mask.any():
                raise ValueError("No 90-120 Hz bins for noise correction")
            lateralisation = lateralisation - lateralisation[..., noise_mask].mean(axis=-1, keepdims=True)
            if not np.isfinite(lateralisation).all():
                raise ValueError(f"Nonfinite lateralisation for {right}, {left}")
            n_checked += 1

        metrics["n_sensor_pairs_checked"] = n_checked
        metrics.update(status="PASS", failed_stage="", error_type="", error_message="")
        return metrics, ""

    except Exception as exc:
        metrics.update(status="FAIL", failed_stage=stage,
                       error_type=type(exc).__name__, error_message=str(exc))
        return metrics, traceback.format_exc()


def main(platform="bluebear", output_dir=None):
    paths = setup_paths(platform)
    if output_dir:
        paths["output_dir"] = output_dir
    os.makedirs(paths["output_dir"], exist_ok=True)

    logging.basicConfig(
        filename=op.join(paths["output_dir"], "Q01_errors.log"),
        filemode="w", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    for key in ("lost_csv", "layout_csv"):
        if not op.isfile(paths[key]):
            raise FileNotFoundError(
                f"{paths[key]} not found. If using macOS, mount "
                "smb://its-rds.bham.ac.uk/rdsprojects/q/quinna-camcan "
                "and check the local mount path."
            )

    subject_ids = load_lost_ids(paths["lost_csv"])
    layout = pd.read_csv(paths["layout_csv"])
    if not {"right_sensors", "left_sensors"}.issubset(layout.columns):
        raise ValueError("Layout must have right_sensors and left_sensors columns")
    print(f"Checking {len(subject_ids)} subjects from {paths['lost_csv']}")

    results = []
    for idx, subject_id in enumerate(subject_ids, 1):
        print(f"[{idx}/{len(subject_ids)}] sub-CC{subject_id}", flush=True)
        record, tb = check_subject(subject_id, paths, layout)
        results.append(record)
        if tb:
            logging.error("sub-CC%s failed at %s:\n%s",
                          subject_id, record["failed_stage"], tb)
        print(f"  {record['status']}: {record['failed_stage']} {record['error_message']}", flush=True)

    result_df = pd.DataFrame(results)
    result_path = op.join(paths["output_dir"], "Q01_subject_qc_results.csv")
    result_df.to_csv(result_path, index=False)
    print("\nStatus counts:\n", result_df["status"].value_counts().to_string())
    print("\nFailure stages:\n", result_df.loc[result_df.status == "FAIL", "failed_stage"].value_counts().to_string())
    print(f"Saved QC results to {result_path}")
    print("PASS means PSD and sensor-pair QC succeeded; it does not guarantee every S01 outlier/stacking step succeeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("mac", "bluebear"), default="bluebear")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    main(platform=args.platform, output_dir=args.output_dir)
