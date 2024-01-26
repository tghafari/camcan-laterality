# -*- coding: utf-8 -*-
"""
====================================
C01_correlating_lateralised_spectra_substr:
    this script:
    1. reads Zhongpeng's (Z01) lateralised
    values of one pair of sensors (one csv file)
    from each frequency band (each folder) into
    one variable called pair_lat_delta/theta...
    2. reads lateralised values of each 
    subcortical structure into hipp/thal_lat
    3. removes the participants without spectra
    or lateralised volume based on the subID 
    column
    4. calculates correlation for remaining
    participants between pair sensor lateralisation
    and lateralisation of one subcortical structure
    5. puts that one correlation value in a table
    called "lat_spectra_substr_corr"
    6. loops over all substrs and saves each
    correlation value column-wise
    7. loops over all pairs of sensors and
    save each correlation row-wise
    
    
Written by Tara Ghafari
t.ghafari@bham.ac.uk
=====================================
"""

import pandas as pd
import numpy as np
import os.path as op
import scipy.stats as stats
import glmtools as glm

# Define the directory  
base_spectra_dir = r'/rds/projects/q/quinna-camcan/Zhongpengdai-scripts'
base_substr_dir = r'/rds/projects/q/quinna-camcan/derivatives'
substr_path = r'mri/lateralized_index'

# Load substr file
substr_lat_df = pd.read_csv(op.join(base_substr_dir, substr_path, 'lateralization_volumes.csv'))
substr_lat_df = substr_lat_df.rename(columns={'SubID':'SubjectID'})

pairs = [
            ['MEG0522', 'MEG0523', 'MEG0912', 'MEG0913'],
            ['MEG0512', 'MEG0513', 'MEG0922', 'MEG0923'],
            ['MEG0532', 'MEG0533', 'MEG0942', 'MEG0943'],
            ['MEG2132', 'MEG2133', 'MEG2142', 'MEG2143'],
            ['MEG1742', 'MEG1743', 'MEG2542', 'MEG2543'],
            ['MEG1932', 'MEG1933', 'MEG2332', 'MEG2333'],
            ['MEG1732', 'MEG1733', 'MEG2512', 'MEG2513'],
            ['MEG1712', 'MEG1713', 'MEG2532', 'MEG2533'],
            ['MEG1922', 'MEG1923', 'MEG2342', 'MEG2343'],
            ['MEG2042', 'MEG2043', 'MEG2032', 'MEG2033'],
            ['MEG2012', 'MEG2013', 'MEG2022', 'MEG2023'],
            ['MEG1832', 'MEG1833', 'MEG2242', 'MEG2243'],
            ['MEG0742', 'MEG0743', 'MEG0732', 'MEG0733'],
            ['MEG0712', 'MEG0713', 'MEG0722', 'MEG0723'],
            ['MEG0632', 'MEG0633', 'MEG1042', 'MEG1043'],
            ['MEG0642', 'MEG0643', 'MEG1032', 'MEG1033'],
            ['MEG0612', 'MEG0613', 'MEG1022', 'MEG1023'],
            ['MEG0312', 'MEG0313', 'MEG1212', 'MEG1213'],
            ['MEG0542', 'MEG0543', 'MEG0932', 'MEG0933'],
            ['MEG0122', 'MEG0123', 'MEG1412', 'MEG1413'],
            ['MEG0112', 'MEG0113', 'MEG1422', 'MEG1423'],
            ['MEG0342', 'MEG0343', 'MEG1222', 'MEG1223'],
            ['MEG0332', 'MEG0333', 'MEG1242', 'MEG1243'],
            ['MEG1942', 'MEG1943', 'MEG2322', 'MEG2323'],
            ['MEG1722', 'MEG1723', 'MEG2522', 'MEG2523'],
            ['MEG1532', 'MEG1533', 'MEG2632', 'MEG2633'],
            ['MEG1642', 'MEG1643', 'MEG2432', 'MEG2433'],
            ['MEG1912', 'MEG1913', 'MEG2312', 'MEG2313'],
            ['MEG1632', 'MEG1633', 'MEG2442', 'MEG2443'],
            ['MEG1842', 'MEG1843', 'MEG2232', 'MEG2233'],
            ['MEG1822', 'MEG1823', 'MEG2212', 'MEG2213'],
            ['MEG0432', 'MEG0433', 'MEG1142', 'MEG1143'],
            ['MEG0422', 'MEG0423', 'MEG1112', 'MEG1113'],
            ['MEG0412', 'MEG0413', 'MEG1122', 'MEG1123'],
            ['MEG0222', 'MEG0223', 'MEG1312', 'MEG1313'],
            ['MEG0212', 'MEG0213', 'MEG1322', 'MEG1323'],
            ['MEG0132', 'MEG0133', 'MEG1442', 'MEG1443'],
            ['MEG0142', 'MEG0143', 'MEG1432', 'MEG1433'],
            ['MEG0442', 'MEG0443', 'MEG1132', 'MEG1133'],
            ['MEG0232', 'MEG0233', 'MEG1342', 'MEG1343'],
            ['MEG1542', 'MEG1543', 'MEG2622', 'MEG2623'],
            ['MEG1522', 'MEG1523', 'MEG2642', 'MEG2643'],
            ['MEG1612', 'MEG1613', 'MEG2422', 'MEG2423'],
            ['MEG1812', 'MEG1813', 'MEG2222', 'MEG2223'],
            ['MEG1622', 'MEG1623', 'MEG2412', 'MEG2413'],
            ['MEG0242', 'MEG0243', 'MEG1332', 'MEG1333'],
            ['MEG1512', 'MEG1513', 'MEG2612', 'MEG2613'],
            ['MEG0322', 'MEG0323', 'MEG1232', 'MEG1233'],
            ['MEG0622', 'MEG0623', 'MEG1012', 'MEG1013'],
            ['MEG0822', 'MEG0823', 'MEG2122', 'MEG2123'],
            ['MEG2112', 'MEG2113', 'MEG0812', 'MEG0813'],
        ]

bands = ['delta', 'theta', 'alpha', 'beta']
substrs = ['Thal', 'Caud', 'Puta', 'Pall', 'Hipp', 'Amyg', 'Accu']

def working_df_maker(pair_lat_spectra_dir, pair, which_pair, substr_lat_df, ls_pairs):
    # load lateralisation index for each pair
    spectrum_pair_lat_df = pd.read_csv(op.join(pair_lat_spectra_dir, pair[which_pair] + '-' + pair[which_pair+2] + '.csv'))
    print(f'working on {pair[which_pair]} - {pair[which_pair+2]}')

    # merge and match the subject_ID column
    working_df = spectrum_pair_lat_df.merge(substr_lat_df, on=['SubjectID'])
    working_df = working_df.dropna()
    ls_pairs.append(f'{pair[which_pair]} - {pair[which_pair+2]}')

    return working_df, ls_pairs

def pearson_calculator(working_df, substr, ls_corrs, ls_pvals):
    print(f'calculating pearson correlation for {substr}')
    temp_corr, temp_pvalue = stats.pearsonr(working_df['Value'].to_numpy(), working_df[substr].to_numpy()) 
    ls_corrs.append(temp_corr)
    ls_pvals.append(temp_pvalue)
    return ls_corrs, ls_pvals


band = 'delta'

# Predefine lists
ls_corrs = []
ls_pvals = []
ls_pairs = []
ls_corrs_arr = []
ls_pvals_arr = []

pair_lat_spectra_dir = op.join(base_spectra_dir, band)
pearsonrs_csv_file = op.join(base_substr_dir, 'correlations', band, 'lat_spectra_substr_pearsonr.csv')
pvals_csv_file = op.join(base_substr_dir, 'correlations', band, 'lat_spectra_substr_pvals.csv')
print(f'calculating correlations for {band} band')

DC = glm.design.DesignConfig()
DC.add_regressor(name='Constant', rtype='Constant')
DC.add_regressor(name='SubStrLat', rtype='Parametric', datainfo='substrlat', preproc='z')
DC.add_simple_contrasts()

for which_pair in [0,1]:
    print(f'working on pair {which_pair+1}')

    for substr in substrs:

        # Read pairs of sensors one by one
        for idx, pair in enumerate(pairs):
            first_working_df, ls_pairs = working_df_maker(pair_lat_spectra_dir, pair, which_pair, substr_lat_df, ls_pairs)

            # Calculate correlation with each substr
            x = first_working_df[substr].to_numpy()

            print(first_working_df['Value'].to_numpy().shape)

            if idx == 0:
                y = first_working_df['Value'].to_numpy()
            else:
                y = np.c_[y, first_working_df['Value'].to_numpy()]


        data = glm.data.TrialGLMData(data=y, substrlat=x)
        design = DC.design_from_datainfo(data.info)
        model = glm.fit.OLSModel(design, data)

        P = glm.permutations.MaxStatPermutation(design, data, 1, 500, metric='tstats', pooled_dims=1)
        P.nulls # null distribution
        t = P.get_thresh(95)

        plt.figure()
        plt.subplot(211)
        plt.plot(model.betas.T) # can change betas to tstats
        plt.legend(model.contrast_names)
        plt.subplot(212)
        plt.plot(model.tstats[1, :])
        plt.plot((0, data.data.shape[1]), (t, t), 'k:')
        plt.plot((0, data.data.shape[1]), (-t, -t), 'k:')



