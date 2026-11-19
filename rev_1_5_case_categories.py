# 5_case_categories.py
# Success and failure categories of class-conditional (Mondrian) conformal prediction at the 90% target, judged against argmax.
# Run after 1_code.ipynb. Reads modeling_table.csv and core_outputs/ (egd_probas.npy, utsw_probas.npy, is_test_egd.npy, is_test_utsw.npy).
# Writes core_outputs/case_categories_long.csv    (one row per patient per split in which the patient was in the test half)
#        core_outputs/case_categories_patients.csv (one row per patient: most common category and set, and how often they occurred)
#        core_outputs/table2_categories.csv        (Table 2: share of test patients in each category, mean over the 200 splits)
#
# Three questions per test patient: was argmax right, is the true subtype in the 90% set, and how many subtypes does the set name.
# Argmax is "confident" when its top probability is at least 0.80 (section 1 shows argmax is right in at least 90% of patients from there).
# Every test patient falls in exactly one category:
#   Successes
#   S1 Confident and correct               argmax right, set = {truth}
#   S2 Borderline call, ambiguity flagged  argmax right but borderline (top probability < 0.80), set = truth + one other subtype
#   S3 Wrong call caught                   argmax wrong, set = truth + one other subtype
#   S4 Wrong call overridden               argmax wrong, set = {truth}
#   Failures
#   F1 Correct call, unnecessary hedging   argmax right and confident, set = truth + one other subtype
#   F2 Valid but uninformative             set = all three subtypes (argmax right or wrong)
#   F3 Excludes the correct call           argmax right, truth not in the set (empty sets included)
#   F4 Wrong call not corrected            argmax wrong, truth not in the set (empty sets included)
# A = astrocytoma, G = glioblastoma, O = oligodendroglioma.

import os
import numpy as np
import pandas as pd
from itertools import product
from crepes import ConformalClassifier

OUT = 'core_outputs'
table = pd.read_csv('modeling_table.csv', usecols=['case_id', 'site', '2021_glioma_class'])
classes = np.array(['astrocytoma', 'glioblastoma', 'oligodendroglioma'])
letters = np.array(['A', 'G', 'O'])
level = 0.90
confident = 0.80

d_egd = table[table['site'] == 'EGD'].reset_index(drop=True)
d_utsw = table[table['site'] == 'UTSW'].reset_index(drop=True)
cohorts = {'EGD': d_egd, 'UTSW': d_utsw}
probas = {'EGD': np.load(os.path.join(OUT, 'egd_probas.npy')), 'UTSW': np.load(os.path.join(OUT, 'utsw_probas.npy'))}
is_test = {'EGD': np.load(os.path.join(OUT, 'is_test_egd.npy')), 'UTSW': np.load(os.path.join(OUT, 'is_test_utsw.npy'))}

# ---- 1. argmax accuracy by top probability, both external cohorts, one value per patient (justifies the 0.80 cutoff) ----
p_all = np.vstack([probas['EGD'], probas['UTSW']])
y_all = np.concatenate([np.searchsorted(classes, d_egd['2021_glioma_class'].values), np.searchsorted(classes, d_utsw['2021_glioma_class'].values)])
top_all = p_all.max(axis=1)
right_all = np.argmax(p_all, axis=1) == y_all
for cut in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
    keep = top_all >= cut
    print('top probability >= {:.2f}: argmax right in {:.1f}% ({:.0f}% of patients)'.format(cut, right_all[keep].mean() * 100, keep.mean() * 100))

# ---- 2. category of every test patient in every split (same Mondrian fit as 1_code.ipynb) ----
split_rows = []
for cohort, seed in product(['EGD', 'UTSW'], range(200)):
    d = cohorts[cohort]
    y = d['2021_glioma_class'].values
    col = np.searchsorted(classes, y)
    p = probas[cohort]
    scores = 1 - p
    test = np.where(is_test[cohort][seed])[0]
    cal = np.where(~is_test[cohort][seed])[0]
    cc = ConformalClassifier()
    cc.fit(scores[cal, col[cal]], bins=y[cal])
    sets = np.column_stack([cc.predict_set(scores[test], bins=np.full(len(test), cls), confidence=level, smoothing=False)[:, k]
                            for k, cls in enumerate(classes)])
    yt = col[test]
    size = sets.sum(axis=1)
    covered = sets[np.arange(len(test)), yt] == 1
    argmax = np.argmax(p[test], axis=1)
    right = argmax == yt
    sure = p[test].max(axis=1) >= confident

    # one condition per category; together they cover every patient exactly once
    conditions = [right & covered & (size == 1),
                  right & covered & (size == 2) & (~sure),
                  (~right) & covered & (size == 2),
                  (~right) & covered & (size == 1),
                  right & covered & (size == 2) & sure,
                  covered & (size == 3),
                  right & (~covered),
                  (~right) & (~covered)]
    names = ['S1 Confident and correct', 'S2 Borderline call, ambiguity flagged', 'S3 Wrong call caught', 'S4 Wrong call overridden',
             'F1 Correct call, unnecessary hedging', 'F2 Valid but uninformative', 'F3 Excludes the correct call', 'F4 Wrong call not corrected']
    category = np.select(conditions, names, default='unassigned')

    # the set written as text, e.g. {A, O}
    set_text = ['{' + ', '.join(letters[sets[i] == 1]) + '}' for i in range(len(test))]

    split_rows.append(pd.DataFrame({'cohort': cohort, 'seed': seed, 'case_id': d['case_id'].values[test],
                                    'true': letters[yt], 'argmax': letters[argmax], 'argmax_correct': right,
                                    'top_probability': p[test].max(axis=1), 'set': set_text, 'category': category}))

long = pd.concat(split_rows, ignore_index=True)
long.to_csv(os.path.join(OUT, 'case_categories_long.csv'), index=False)
print('\nunassigned rows (must be 0):', np.sum(long['category'] == 'unassigned'))

# ---- 3. Table 2: share of test patients in each category, mean and 2.5th-97.5th percentile over the 200 splits ----
counts = long.groupby(['cohort', 'seed', 'category']).size().unstack(fill_value=0)
shares = counts.div(counts.sum(axis=1), axis=0) * 100
mean_share = shares.groupby('cohort').mean().T
low_share = shares.groupby('cohort').quantile(0.025).T
high_share = shares.groupby('cohort').quantile(0.975).T
composition = pd.crosstab([long['cohort'], long['category']], long['true'], normalize='index') * 100

table2 = pd.DataFrame({'EGD, % of test patients': mean_share['EGD'].round(1),
                       'EGD, range': low_share['EGD'].round(1).astype(str) + '-' + high_share['EGD'].round(1).astype(str),
                       'UTSW-Glioma, % of test patients': mean_share['UTSW'].round(1),
                       'UTSW-Glioma, range': low_share['UTSW'].round(1).astype(str) + '-' + high_share['UTSW'].round(1).astype(str)})
table2.to_csv(os.path.join(OUT, 'table2_categories.csv'))
print(table2)
print('\ntrue subtype within each category (%):')
print(composition.round(0))

# ---- 4. one row per patient: most common category and set, and their share of the patient's test-half splits ----
patients = long.groupby(['cohort', 'case_id']).agg(true=('true', 'first'), argmax=('argmax', 'first'),
                                                   argmax_correct=('argmax_correct', 'first'), n_test=('seed', 'size')).reset_index()
modal_cat = long.groupby(['cohort', 'case_id'])['category'].agg(lambda s: s.value_counts().index[0]).rename('modal_category')
cat_share = long.groupby(['cohort', 'case_id'])['category'].agg(lambda s: s.value_counts().iloc[0] / len(s)).rename('category_stability')
modal_set = long.groupby(['cohort', 'case_id'])['set'].agg(lambda s: s.value_counts().index[0]).rename('modal_set')
set_share = long.groupby(['cohort', 'case_id'])['set'].agg(lambda s: s.value_counts().iloc[0] / len(s)).rename('set_stability')
patients = patients.join(modal_cat, on=['cohort', 'case_id']).join(cat_share, on=['cohort', 'case_id'])
patients = patients.join(modal_set, on=['cohort', 'case_id']).join(set_share, on=['cohort', 'case_id'])

prob_egd = pd.DataFrame({'case_id': d_egd['case_id'], 'pA': probas['EGD'][:, 0], 'pG': probas['EGD'][:, 1], 'pO': probas['EGD'][:, 2]})
prob_utsw = pd.DataFrame({'case_id': d_utsw['case_id'], 'pA': probas['UTSW'][:, 0], 'pG': probas['UTSW'][:, 1], 'pO': probas['UTSW'][:, 2]})
patients = patients.merge(pd.concat([prob_egd, prob_utsw]), on='case_id')
patients.to_csv(os.path.join(OUT, 'case_categories_patients.csv'), index=False)

# ---- 5. candidate examples: same category in at least 90% of the patient's test-half splits ----
stable = patients[patients['category_stability'] >= 0.90]
print('\nstable patients per category:')
print(stable.groupby('modal_category').size())

# the patients used in Figures 5 and 6
examples = ['BT1106', 'EGD-0106', 'EGD-0022', 'EGD-0368', 'EGD-0026', 'BT0438', 'BT0291', 'BT0011']
print(patients[patients['case_id'].isin(examples)].round(2).to_string(index=False))
