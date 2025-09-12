"""
============================================
St02_canonical_correlation_analysis

In this code we are asking:
    What pattern of MEG power laterality best explains 
    the lateralised volume of the subcortical structures?

CCA needs two sets of variables:
Set 1 (Y): outcome set-> lateralised volume of one substr.

Set 2 (X): predictors set-> MEG power lateralisation features (bands x 
sensors in both sensor_type = 4 x (51(combined palanars) + 51(mags)))

On the MEG side, CCA finds a weighted combination of your MEG features 
(a “canonical variate”).

On the substr side, since you only have one variable, 
the canonical variate is just substr LV itself (maybe scaled).

CCA then maximizes the correlation between those two canonical variates.
In effect: We are finding the linear pattern of MEG lateralisation across 
sensors and bands that maximally correlates with each substr lateralisation.

workflow:

Prepare X:
    Subjects × features (all LIs of MEG sensors × bands).
    Standardize each column (z-score across subjects).

Prepare Y:
    Vector of substr LV.
    Standardize as well.

Run CCA:
    With only 1 Y variable, CCA will essentially give us 
    one canonical variate on X and one on Y.

Inspect weights:
    Map the canonical weights back to scalp maps for each band.
    This gives us a topography per band that says: “This is the 
    pattern of sensors/bands most strongly associated with substr asymmetry.”

Cross-validation: 
    Do k-fold CV. Train weights on 80% of participants, predict canonical
    variate scores in the 20%, compute correlation with substr LV.
    This shows the pattern generalizes.

Example of interpretation in your results context:
Let’s say the CCA finds:
Strong negative weights on posterior delta and beta sensors 
(gradiometers and magnetometers).
Little or no weight on frontal sensors.

Then your interpretation is:
“The MEG pattern that best explains caudate lateralisation is 
characterized by reduced delta and beta asymmetry over posterior 
sensors. This matches our univariate findings (negative 
caudate–delta/beta correlations at posterior sites). 
CCA shows that this posterior slow/fast oscillatory pattern, 
taken together, accounts for caudate asymmetry better than any single sensor or band.”

The canonical correlation r tells you the overall strength of this 
relationship (e.g., r = –0.30, p < 0.01 via permutation test).


This script will:
1) inputs lateralised volume of all subs with significant findings
2) inputs lateralised power (PLI) per band per sensor
3) writes a function that runs CCA on one substr
and all PLIs
4) writes a function that maps the canonical weights back 
to scalp maps for each band.
5) writes a function that does k-fold CV. Trains weights on 
80% of participants, predict canonical variate scores in the 20%, 
compute correlation with substr LV.
6) runs steps 3-6 on all subcortical structures.

Explanation by ChatGPT5
code by Tara Ghafari
tara.ghafari@gmail.com
12/09/2025
===========================================
"""

import os
import os.path as op
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ------------------------- CONFIG -------------------------
BANDS = ("Delta", "Theta", "Alpha", "Beta")
# file pattern per band inside LI_dir:
#   "{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv"
N_PERM = 5000
N_SPLITS = 5
RANDOM_STATE = 42

# ------------------------- PATHS -------------------------
def setup_paths(platform='mac'):
    """Set up and return file paths based on the system platform."""
    if platform == 'bluebear':
        quinna_dir = '/rds/projects/q/quinna-camcan/'
        sub2ctx_dir = '/rds/projects/j/jenseno-sub2ctx/camcan'
        jenseno_dir = '/rds/projects/j/jenseno-avtemporal-attention/Projects/'
    elif platform == 'mac':
        quinna_dir = '/Volumes/quinna-camcan-1/'
        sub2ctx_dir = '/Volumes/jenseno-sub2ctx-1/camcan'
        jenseno_dir = '/Volumes/jenseno-avtemporal-attention/Projects/'
    else:
        raise ValueError("Unsupported platform. Use 'mac' or 'bluebear'.")

    paths = {
        'LI_dir': op.join(sub2ctx_dir, 'derivatives/meg/sensor/lateralized_index/bands'),
        'LV_csv': op.join(sub2ctx_dir, 'derivatives/mri/lateralized_index/lateralization_volumes_no-vol-outliers.csv'),
        'sub_list': op.join(quinna_dir, 'dataman/data_information/last_FINAL_sublist-vol-outliers-removed.csv'),
        'out_dir': op.join(sub2ctx_dir, 'derivatives/cca_results')
    }
    os.makedirs(paths['out_dir'], exist_ok=True)
    return paths

# -------------------- LOAD & NORMALIZE --------------------
def _normalize_subject_id(df: pd.DataFrame) -> pd.DataFrame:
    """Rename SubjectID column to 'SubjectID' (case-insensitive variants) and cast to str."""
    for c in df.columns:
        if c.lower() in ('subjectid', 'subject_id', 'subject', 'subid'):
            df = df.rename(columns={c: 'SubjectID'})
            break
    if 'SubjectID' not in df.columns:
        raise ValueError("No SubjectID column found.")
    df['SubjectID'] = df['SubjectID'].astype(str)
    return df

def load_lv_and_subjects(paths) -> Tuple[pd.DataFrame, List[str]]:
    lv = pd.read_csv(paths['LV_csv'])
    lv = _normalize_subject_id(lv)

    sub_list = pd.read_csv(paths['sub_list'])
    sub_list = _normalize_subject_id(sub_list)

    keep_ids = [sid for sid in sub_list['SubjectID'].tolist() if sid in set(lv['SubjectID'])]
    lv = lv[lv['SubjectID'].isin(keep_ids)].copy()
    lv = lv.set_index('SubjectID').loc[keep_ids].reset_index()
    return lv, keep_ids

def load_li_band(paths, band: str) -> pd.DataFrame:
    """Load a single band LI table; expects both grads+mags columns inside."""
    fname = f"{band}_lateralised_power_allsens_subtraction_nonoise_no-vol-outliers.csv"
    fpath = op.join(paths['LI_dir'], fname)
    if not op.isfile(fpath):
        raise FileNotFoundError(f"LI file missing: {fpath}")
    df = pd.read_csv(fpath)
    df = _normalize_subject_id(df)
    return df

def build_feature_matrix_all_bands(paths, sub_ids: List[str]) -> pd.DataFrame:
    """
    Horizontally concatenate band tables (on SubjectID) and prefix each sensor column with 'Band__'.
    Returns X (subjects × features) aligned to sub_ids order; z-scored per column.
    """
    merged = None
    for band in BANDS:
        df = load_li_band(paths, band)
        df = df[df['SubjectID'].isin(sub_ids)].copy()
        numeric_cols = [c for c in df.columns if c != 'SubjectID']
        # prefix features with band
        df_pref = df[['SubjectID'] + numeric_cols].rename(columns={c: f"{band}__{c}" for c in numeric_cols})
        merged = df_pref if merged is None else pd.merge(merged, df_pref, on='SubjectID', how='inner')

    # align order and z-score
    merged = merged.set_index('SubjectID').loc[sub_ids]
    X = merged.astype(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0).replace(0, np.nan)
    X = X.fillna(0.0)
    return X

# -------------------- TARGETS (Y) --------------------
def pick_targets(lv_df: pd.DataFrame, separate_structs: list, combined_structs: bool) -> Dict[str, pd.Series]:
    """
    Build individual and combined targets from LV.
    Combined: caud+put, caud+put+gp (z-avg then z again). Fuzzy col matching.
    separate_structs: list of the columns you want to be added as LVs (separately)
    combined_structs: do you want to combine caud+puta and caud+puta+pall and make two separate features?
    """
    lv_df = lv_df.copy()
    lv_df = _normalize_subject_id(lv_df)
    lv_df = lv_df.set_index('SubjectID')

    cand_cols = [c for c in lv_df.columns[separate_structs]]
    targets = {c: lv_df[c].astype(float) for c in cand_cols}

    def zser(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return (s - s.mean()) / s.std(ddof=0)

    if combined_structs:
        comb = (zser(lv_df['Caud']) + zser(lv_df['Puta'])) / 2.0
        targets['comb_caud_put'] = zser(comb)

    if combined_structs:
        comb2 = (zser(lv_df['Caud']) + zser(lv_df['Puta']) + zser(lv_df['Pall'])) / 3.0
        targets['comb_caud_put_gp'] = zser(comb2)

    return targets

# -------------------- CCA + PERM + CV --------------------
def run_cca_one(X: pd.DataFrame, y: pd.Series,
                n_perm: int = N_PERM, random_state: int = RANDOM_STATE):
    """
    Fit 1-component CCA on standardized X and y.
    Returns dict with r, p_perm (two-sided), x_weights, x_scores, y_scores.
    """
    from sklearn.cross_decomposition import CCA

    # align y to X
    y = y.loc[X.index].astype(float).values.reshape(-1, 1)
    Xn = X.values

    # scale
    X_scaler = StandardScaler().fit(Xn)
    y_scaler = StandardScaler().fit(y)
    Xz = X_scaler.transform(Xn)
    yz = y_scaler.transform(y)

    cca = CCA(n_components=1, max_iter=5000)
    cca.fit(Xz, yz)
    X_c, y_c = cca.transform(Xz, yz)
    r_obs = np.corrcoef(X_c[:, 0], y_c[:, 0])[0, 1]

    # permutation on y (fixed X, refit transform each time for y perm)
    rng = np.random.default_rng(random_state)
    perm_rs = np.zeros(n_perm, dtype=float)
    for p in range(n_perm):
        perm = rng.permutation(yz.shape[0])
        yzp = yz[perm]
        X_cp, y_cp = cca.transform(Xz, yzp)
        perm_rs[p] = np.corrcoef(X_cp[:, 0], y_cp[:, 0])[0, 1]

    p_perm = (np.sum(np.abs(perm_rs) >= np.abs(r_obs)) + 1.0) / (n_perm + 1.0)

    x_weights = pd.Series(cca.x_weights_.ravel(), index=X.columns, name="x_weight")
    x_scores  = pd.Series(X_c[:, 0], index=X.index, name="x_score")
    y_scores  = pd.Series(y_c[:, 0], index=X.index, name="y_score")

    return dict(r=float(r_obs), p_perm=float(p_perm),
                x_weights=x_weights, x_scores=x_scores, y_scores=y_scores)

def run_cca_cv(X: pd.DataFrame, y: pd.Series,
               n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> Dict:
    """k-fold CV: fit on train, transform test, corr(Xc_test, yc_test)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_full = y.loc[X.index].astype(float).values.reshape(-1, 1)
    Xn = X.values

    cors = []
    for tr, te in kf.split(Xn):
        Xtr, Xte = Xn[tr], Xn[te]
        ytr, yte = y_full[tr], y_full[te]

        Xs, Ys = StandardScaler().fit(Xtr), StandardScaler().fit(ytr)
        Xtrz, ytrz = Xs.transform(Xtr), Ys.transform(ytr)
        Xtez, ytez = Xs.transform(Xte), Ys.transform(yte)

        cca = CCA(n_components=1, max_iter=5000)
        cca.fit(Xtrz, ytrz)
        Xc_te, yc_te = cca.transform(Xtez, ytez)
        r = np.corrcoef(Xc_te[:, 0], yc_te[:, 0])[0, 1]
        cors.append(float(r))

    return dict(cv_corrs=cors, cv_mean=float(np.mean(cors)), cv_std=float(np.std(cors)))

# -------------------- WEIGHT MAPPING --------------------
def split_weights_by_band(x_weights: pd.Series) -> Dict[str, pd.Series]:
    """feature names are 'Band__Sensor'; return dict[Band] -> Series(sensor -> weight)."""
    out = {b: {} for b in BANDS}
    for feat, w in x_weights.items():
        if '__' not in feat:
            continue
        band, sensor = feat.split('__', 1)
        if band in out:
            out[band][sensor] = w
    return {b: pd.Series(v).sort_index() for b, v in out.items() if v}

# -------------------- MAIN PIPELINE --------------------
def main(platform='mac', n_perm=N_PERM, n_splits=N_SPLITS):
    paths = setup_paths(platform)
    print(f"[INFO] LV: {paths['LV_csv']}")
    print(f"[INFO] LI dir: {paths['LI_dir']}")
    print(f"[INFO] Subject list: {paths['sub_list']}")
    print(f"[INFO] Output dir: {paths['out_dir']}")

    # Load LV + subjects
    lv_df, sub_ids = load_lv_and_subjects(paths)

    # Build X from all four bands (mags+grads combined columns in each file)
    X = build_feature_matrix_all_bands(paths, sub_ids)

    # Build Y targets (individual LV columns + combined)
    targets = pick_targets(lv_df, separate_structs=['Caud','Puta','Pall','Hipp'], combined_structs=False)
    # align all series to X index order
    X.index = pd.Index(sub_ids, name='SubjectID')

    summary_rows = []
    for target_name, y_ser in targets.items():
        print(f"\n[RUN] CCA target: {target_name}")
        y_aligned = y_ser.loc[X.index]
        # z-score Y across subjects
        y_aligned = (y_aligned - y_aligned.mean()) / y_aligned.std(ddof=0)
        y_aligned = y_aligned.fillna(0.0)

        # Fit, perm test
        res = run_cca_one(X, y_aligned, n_perm=n_perm, random_state=RANDOM_STATE)
        # CV
        cv = run_cca_cv(X, y_aligned, n_splits=n_splits, random_state=RANDOM_STATE)

        # Save weights (per band) and subject scores
        weights_by_band = split_weights_by_band(res['x_weights'])
        target_dir = op.join(paths['out_dir'], f'weights_{target_name}')
        os.makedirs(target_dir, exist_ok=True)
        for band, ser in weights_by_band.items():
            ser.to_csv(op.join(target_dir, f'{target_name}__{band}__weights.csv'))

        pd.DataFrame({
            'SubjectID': X.index,
            'X_score': res['x_scores'].values,
            'Y_score': res['y_scores'].values
        }).to_csv(op.join(paths['out_dir'], f'{target_name}__scores.csv'), index=False)

        print(f"    r = {res['r']:.3f}, p_perm = {res['p_perm']:.4f}, CV mean r = {cv['cv_mean']:.3f} ± {cv['cv_std']:.3f}")
        print(f"    Saved weights -> {target_dir}")
        print(f"    Saved scores  -> {op.join(paths['out_dir'], f'{target_name}__scores.csv')}")

        summary_rows.append({
            'target': target_name,
            'n_subjects': X.shape[0],
            'n_features': X.shape[1],
            'r_canonical': res['r'],
            'p_perm_two_sided': res['p_perm'],
            'cv_mean_r': cv['cv_mean'],
            'cv_std_r': cv['cv_std'],
        })

    summary = pd.DataFrame(summary_rows).sort_values('p_perm_two_sided')
    summary_path = op.join(paths['out_dir'], 'cca_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Summary saved: {summary_path}")

if __name__ == "__main__":
    # Set platform='bluebear' if running there
    main(platform='mac', n_perm=N_PERM, n_splits=N_SPLITS)