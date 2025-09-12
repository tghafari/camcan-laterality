"""
============================================
St02_canonical_correlation_analysis

In this code we are asking:
    What pattern of MEG power laterality best explains 
    the lateralised volume of the subcortical structures?

CCA needs two sets of variables:
Set 1 (Y): outcome set-> lateralised volume of one substr.

Set 2 (X): predictors set-> MEG power lateralisation features (bands x 
sensors in both sensor_type = 4 x (102 + 51))

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

