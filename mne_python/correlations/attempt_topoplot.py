

import mne
import pandas as pd

df = pd.read_csv('/rds/projects/q/quinna-camcan/derivatives/correlations/alpha/lat_spectra_substr_pearsonr.csv')

fname = '/rds/projects/q/quinna-camcan/cc700/meg/pipeline/release005/BIDSsep/derivatives_rest/aa/AA_movecomp/aamod_meg_maxfilt_00002/sub-CC110033/mf2pt2_sub-CC110033_ses-rest_task-rest_meg.fif'
raw = mne.io.read_raw_fif(fname)
raw.pick_types(meg='grad')

chans = [chs.split(' - ')[0] for chs in df['Unnamed: 0']] + [chs.split(' - ')[1] for chs in df['Unnamed: 0']]

metrics = np.zeros((204,))
for idx, ch in enumerate(chans):
    ind_chan = mne.pick_channels(raw.ch_names, [ch])
    ind_val = np.where(np.array([name.find(ch) for name in df['Unnamed: 0']]) >-1 )[0]
    metrics[ind_chan] = df['Thal'].values[ind_val]

# Simple plot
mne.viz.plot_topomap(metrics, raw.info)


# Replace mask with boolean indicating thresholded pvals
mask = metrics == 0
# Complex plot
ax = plt.subplot(2,4,2)
mne.viz.plot_topomap(metrics, raw.info, image_interp='nearest', mask=mask[::2], contours=0, cmap='RdBu_r', vlim=(-0.05, 0.05), mask_params={'marker': '*'}, axes=ax)