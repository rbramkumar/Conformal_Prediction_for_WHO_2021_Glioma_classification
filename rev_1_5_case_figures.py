# 5_case_figures.py
# Figure 5 (successes S1-S4) and Figure 6 (failures F1-F4) of class-conditional conformal prediction against argmax, 90% target.
# Run after 5_case_categories.py. Reads core_outputs/case_categories_patients.csv and table2_categories.csv
# and, for each example patient, t1ce_norm.nii.gz, flair_norm.nii.gz and seg.nii.gz from the preprocessed folder.
# Writes core_outputs/Figure5.png/.tiff and Figure6.png/.tiff
# Each row: category and how often it occurs (left), contrast-enhanced T1 and FLAIR at the axial slice with the largest tumor area,
# whole-tumor outline in yellow (middle), probabilities, argmax and the 90% set (right).
# Images: reoriented to RAS, shown anterior up in radiological convention (patient right on image left), window 1st-99.5th
# percentile of brain voxels on that slice, cropped to the brain.
# Examples: same category in at least 90% of the splits in which the patient was in the test half (5_case_categories.py, section 5).

import os
import textwrap
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'core_outputs'
IMG = r'G:\\My Drive\\bsd\\11_radiomics_for_gliomas\\02_train\\trial_80\\preprocessed'
patients = pd.read_csv(os.path.join(OUT, 'case_categories_patients.csv')).set_index('case_id')
table2 = pd.read_csv(os.path.join(OUT, 'table2_categories.csv')).set_index('category')
full_name = {'A': 'Astrocytoma', 'G': 'Glioblastoma', 'O': 'Oligodendroglioma'}
plt.rcParams.update({'font.family': ['Arial', 'Liberation Sans', 'DejaVu Sans'], 'font.size': 7})

# one row per panel: figure, letter, category (as in table2_categories.csv), description, patient
panels = [
    ('Figure5', 'A', 'S1 Correct, single subtype', 'Argmax is right, and the set names only the true subtype.', 'BT1106'),
    ('Figure5', 'B', 'S2 Borderline call, ambiguity flagged', 'Argmax is right but borderline (top probability below 0.80). The set adds the close second subtype.', 'EGD-0106'),
    ('Figure5', 'C', 'S3 Wrong call caught', 'Argmax is wrong. The set keeps the true subtype alongside the wrong call.', 'EGD-0022'),
    ('Figure5', 'D', 'S4 Wrong call overridden', 'Argmax is wrong. The set leaves out the wrong call and names the true subtype alone.', 'EGD-0368'),
    ('Figure6', 'A', 'F1 Correct call, unnecessary hedging', 'Argmax is confident and right, but the set adds a second subtype.', 'EGD-0026'),
    ('Figure6', 'B', 'F2 Valid but uninformative', 'The set contains the true subtype but excludes nothing.', 'BT0438'),
    ('Figure6', 'C', 'F3 Excludes the correct call', 'Argmax is right, but the glioblastoma probability is below its cutoff, so the set leaves the true subtype out.', 'BT0291'),
    ('Figure6', 'D', 'F4 Wrong call not corrected', 'Argmax is wrong, and the set names the same wrong subtype.', 'BT0011')]

rows_per_fig = {'Figure5': 4, 'Figure6': 4}
figs = {}
grids = {}
row_index = {'Figure5': 0, 'Figure6': 0}
for name in ['Figure5', 'Figure6']:
    n = rows_per_fig[name]
    height = 1.95 * n + 0.35
    figs[name] = plt.figure(figsize=(7.4, height))
    grids[name] = figs[name].add_gridspec(n, 4, width_ratios=[1.35, 1.25, 1.25, 1.55], wspace=0.06, hspace=0.12,
                                          left=0.01, right=0.99, top=1 - 0.3 / height, bottom=0.01)

for fig_name, letter, category, description, case in panels:
    fig = figs[fig_name]
    gs = grids[fig_name]
    i = row_index[fig_name]
    row_index[fig_name] = i + 1
    r = patients.loc[case]
    folder = os.path.join(IMG, 'MR_' + case if case.startswith('EGD') else case)

    # load and reorient to RAS
    t1ce = np.asarray(nib.as_closest_canonical(nib.load(os.path.join(folder, 't1ce_norm.nii.gz'))).dataobj, dtype=float)
    flair = np.asarray(nib.as_closest_canonical(nib.load(os.path.join(folder, 'flair_norm.nii.gz'))).dataobj, dtype=float)
    seg = np.asarray(nib.as_closest_canonical(nib.load(os.path.join(folder, 'seg.nii.gz'))).dataobj) > 0

    # axial slice with the largest tumor area, anterior up, patient right on image left
    k = int(np.argmax(seg.sum(axis=(0, 1))))
    t1ce_slice = np.fliplr(np.rot90(t1ce[:, :, k]))
    flair_slice = np.fliplr(np.rot90(flair[:, :, k]))
    mask_slice = np.fliplr(np.rot90(seg[:, :, k])).astype(float)
    brain_slice = (t1ce_slice != 0) | (flair_slice != 0)

    # crop to the brain with a 6-pixel margin
    ys, xs = np.where(brain_slice)
    y0 = max(ys.min() - 6, 0)
    y1 = ys.max() + 6
    x0 = max(xs.min() - 6, 0)
    x1 = xs.max() + 6
    t1ce_crop = t1ce_slice[y0:y1, x0:x1]
    flair_crop = flair_slice[y0:y1, x0:x1]
    mask_crop = mask_slice[y0:y1, x0:x1]

    # left column: category, description, how often it occurs
    freq = 'EGD {:.1f}%, UTSW-Glioma {:.1f}% of test patients.'.format(table2.loc[category, 'EGD, % of test patients'],
                                                                     table2.loc[category, 'UTSW-Glioma, % of test patients'])
    ax = fig.add_subplot(gs[i, 0])
    ax.axis('off')
    ax.text(0.0, 0.66, letter + '  ' + textwrap.fill(category, 24, subsequent_indent='    '), fontsize=7.5, fontweight='bold', va='bottom')
    ax.text(0.0, 0.60, textwrap.fill(description, 30) + '\n\n' + textwrap.fill(freq, 30), fontsize=6.6, va='top', color='#333333')

    # contrast-enhanced T1
    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(t1ce_crop, cmap='gray', vmin=np.percentile(t1ce_slice[brain_slice], 1), vmax=np.percentile(t1ce_slice[brain_slice], 99.5),
              interpolation='lanczos')
    ax.contour(mask_crop, levels=[0.5], colors='#FFD400', linewidths=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(case, fontsize=7, fontweight='bold')
    if i == 0:
        ax.set_title('Contrast-enhanced T1', fontsize=8, fontweight='bold', pad=4)

    # FLAIR
    ax = fig.add_subplot(gs[i, 2])
    ax.imshow(flair_crop, cmap='gray', vmin=np.percentile(flair_slice[brain_slice], 1), vmax=np.percentile(flair_slice[brain_slice], 99.5),
              interpolation='lanczos')
    ax.contour(mask_crop, levels=[0.5], colors='#FFD400', linewidths=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        ax.set_title('FLAIR', fontsize=8, fontweight='bold', pad=4)

    # right column: probabilities, argmax, 90% set
    set_names = ', '.join(full_name[s.strip()] for s in r['modal_set'].strip('{}').split(',')) if r['modal_set'] != '{}' else ''
    text = ('Cohort: ' + ('EGD' if case.startswith('EGD') else 'UTSW-Glioma') +
            '\nMolecular truth: ' + full_name[r['true']] +
            '\n\nModel probabilities: A {:.2f}, G {:.2f}, O {:.2f}'.format(r['pA'], r['pG'], r['pO']) +
            '\n\nArgmax: ' + full_name[r['argmax']] + ('' if r['argmax_correct'] else ' (wrong)') +
            '\n90% set: ' + textwrap.fill('{' + set_names + '}', 30, subsequent_indent='   ') +
            '\nSame set in {:.0f}% of test-half splits'.format(100 * r['set_stability']))
    ax = fig.add_subplot(gs[i, 3])
    ax.axis('off')
    ax.text(0.04, 0.5, text, fontsize=6.8, va='center', linespacing=1.35)

for name in ['Figure5', 'Figure6']:
    figs[name].savefig(os.path.join(OUT, name + '.png'), dpi=300)
    figs[name].savefig(os.path.join(OUT, name + '.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(figs[name])
print('saved Figure5 and Figure6')
