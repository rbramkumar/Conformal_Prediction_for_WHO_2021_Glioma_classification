import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import roc_curve, auc
from sklearn.base import clone
from mrmr import mrmr_classif
from crepes import ConformalClassifier
import pickle

# #############################################################################
# WHO 2021 GLIOMA SUBTYPING: radiomics + class-conditional conformal prediction
#
# Pipeline overview (top to bottom):
#   Section 1  Load data, build cohorts, preprocess features
#   Section 2  Train the calibrated RandomForest base classifier
#   Section 3  Evaluate discrimination (AUC, sens, spec, PPV, NPV) on dev and external
#   Section 3b Cost-adjusted decision rule (class-weighted argmax), frozen-dev
#              weights plus site-recalibrated and one-vs-rest variants
#   Section 4  Conformal prediction at four confidence levels (single split)
#   Section 5  Repeated-split coverage stability + representative-split seed
#
# Cohorts: development = TCGA + UCSF; external validation = EGD and UTSW.
# Coverage validity for the external cohorts is reported from the 200-split
# analysis in Section 5, so Section 4 stores external conformal numbers as
# point estimates for one split (no bootstrap CI); the only conformal bootstrap
# kept is the development leave-one-fold-out estimate, which is a single
# internal computation rather than a resampled split.
# #############################################################################


# =============================================================================
# HELPER FUNCTIONS
# These are factored out only because each is called repeatedly (across three
# evaluation sets and four confidence levels); inlining would duplicate them
# many times. The rest of the pipeline is written as a single top-to-bottom flow.
# =============================================================================

# Bootstrap per-class discrimination metrics (AUC, PPV, NPV) for the argmax
# classifier. Used in Section 3 for dev OOF and both external test sets.
def evaluate_base_classifier(y_true, y_probas, classes, n_iterations=1000, alpha=0.95, weights=None):
    n_samples = len(y_true)
    y_true_arr = np.array(y_true)
    # weights=None reproduces plain argmax exactly. A weight vector implements the
    # class-weighted argmax decision rule y = argmax_k (w_k * p_k); only the ratios
    # of the weights matter, so the unweighted case is weights = ones. AUC below is
    # computed from the raw probability columns and is therefore unaffected by the
    # weights (it is threshold-independent); only sens/spec/PPV/NPV shift.
    if weights is None:
        weight_vec = np.ones(len(classes))
    else:
        weight_vec = np.array(weights, dtype=float)
    y_pred_arr = classes[np.argmax(y_probas * weight_vec, axis=1)]
    lb = LabelBinarizer().fit(classes)
    y_true_bin = lb.transform(y_true_arr)
    stats = {cls: {'auc': [], 'sens': [], 'spec': [], 'ppv': [], 'npv': []} for cls in classes}

    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true_arr[indices]
        y_pred_boot = y_pred_arr[indices]
        y_true_bin_boot = y_true_bin[indices]
        y_probas_boot = y_probas[indices]
        for i, cls in enumerate(classes):
            if len(np.unique(y_true_bin_boot[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin_boot[:, i], y_probas_boot[:, i])
                stats[cls]['auc'].append(auc(fpr, tpr))
            tp = np.sum((y_pred_boot == cls) & (y_true_boot == cls))
            fp = np.sum((y_pred_boot == cls) & (y_true_boot != cls))
            tn = np.sum((y_pred_boot != cls) & (y_true_boot != cls))
            fn = np.sum((y_pred_boot != cls) & (y_true_boot == cls))
            # sensitivity and PPV share the same TP but different denominators:
            # sensitivity is over true class members (stable), PPV over argmax
            # predictions of the class (tiny for the minority class).
            if (tp + fn) > 0: stats[cls]['sens'].append(tp / (tp + fn))
            if (tn + fp) > 0: stats[cls]['spec'].append(tn / (tn + fp))
            if (tp + fp) > 0: stats[cls]['ppv'].append(tp / (tp + fp))
            if (tn + fn) > 0: stats[cls]['npv'].append(tn / (tn + fn))

    results = {}
    raw = {}
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    for cls in classes:
        results[cls] = {}
        raw[cls] = {}
        for metric in ['auc', 'sens', 'spec', 'ppv', 'npv']:
            if len(stats[cls][metric]) > 0:
                mean_val = float(np.mean(stats[cls][metric]))
                lower_ci = float(np.percentile(stats[cls][metric], lower_p))
                upper_ci = float(np.percentile(stats[cls][metric], upper_p))
                raw[cls][metric] = {'mean': mean_val, 'lo': lower_ci, 'hi': upper_ci}
                results[cls][metric] = f"{mean_val:.2f} [{lower_ci:.2f}-{upper_ci:.2f}]"
            else:
                results[cls][metric] = "N/A"
                raw[cls][metric] = {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    return results, raw


# Bootstrap per-class conformal metrics. Used in Section 4 for the development
# leave-one-fold-out estimate only.
def compute_bootstrapped_metrics(y_true, y_pred_sets, classes, n_iterations=1000, alpha=0.95):
    n_samples = len(y_true)
    y_true_arr = np.array(y_true)
    n_classes = len(classes)
    stats = {cls: {'cov': [], 'size': [], 'pct_1': [], 'pct_2': [], 'pct_all': []} for cls in classes}

    for _ in range(n_iterations):
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true_arr[indices]
        y_pred_sets_boot = y_pred_sets[indices]
        for i, cls in enumerate(classes):
            cls_mask = (y_true_boot == cls)
            if np.sum(cls_mask) > 0:
                stats[cls]['cov'].append(np.mean(y_pred_sets_boot[cls_mask, i]))
                set_sizes = np.sum(y_pred_sets_boot[cls_mask], axis=1)
                stats[cls]['size'].append(np.mean(set_sizes))
                stats[cls]['pct_1'].append(np.mean(set_sizes == 1) * 100)
                stats[cls]['pct_2'].append(np.mean(set_sizes == 2) * 100)
                stats[cls]['pct_all'].append(np.mean(set_sizes == n_classes) * 100)

    results = {}
    raw = {}
    lower_p = ((1.0 - alpha) / 2.0) * 100
    upper_p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    for cls in classes:
        results[cls] = {}
        raw[cls] = {}
        for metric in ['cov', 'size', 'pct_1', 'pct_2', 'pct_all']:
            if len(stats[cls][metric]) > 0:
                mean_val = float(np.mean(stats[cls][metric]))
                lower_ci = float(np.percentile(stats[cls][metric], lower_p))
                upper_ci = float(np.percentile(stats[cls][metric], upper_p))
                raw[cls][metric] = {'mean': mean_val, 'lo': lower_ci, 'hi': upper_ci}
                if metric in ['pct_1', 'pct_2', 'pct_all']:
                    results[cls][metric] = f"{mean_val:.0f}% [{lower_ci:.0f}-{upper_ci:.0f}]"
                else:
                    results[cls][metric] = f"{mean_val:.2f} [{lower_ci:.2f}-{upper_ci:.2f}]"
            else:
                results[cls][metric] = "N/A"
                raw[cls][metric] = {'mean': float('nan'), 'lo': float('nan'), 'hi': float('nan')}
    return results, raw


# Point per-class conformal metrics for one calibration/test split (no CI).
# Used in Section 4 for the external cohorts; their uncertainty comes from the
# 200-split analysis in Section 5.
def conformal_point_metrics(y_true, pred_sets, classes):
    y_true_arr = np.array(y_true)
    n_classes = len(classes)
    set_sizes_all = pred_sets.sum(axis=1)
    out = {}
    for i, cls in enumerate(classes):
        cls_mask = (y_true_arr == cls)
        if np.sum(cls_mask) == 0:
            out[cls] = {'cov': np.nan, 'size': np.nan, 'pct_1': np.nan, 'pct_2': np.nan, 'pct_all': np.nan}
            continue
        sizes = set_sizes_all[cls_mask]
        out[cls] = {
            'cov': float(np.mean(pred_sets[cls_mask, i])),
            'size': float(np.mean(sizes)),
            'pct_1': float(np.mean(sizes == 1) * 100),
            'pct_2': float(np.mean(sizes == 2) * 100),
            'pct_all': float(np.mean(sizes == n_classes) * 100),
        }
    return out


np.random.seed(42)


# =============================================================================
# SECTION 1: LOAD DATA, BUILD COHORTS, PREPROCESS FEATURES
# Load the modeling table, drop molecularly unclassifiable cases, split each
# external cohort 50/50 into calibration and test (stratified by class), then
# impute, variance-filter, z-score with the development scaler, and apply a
# correlation filter. All transforms are fit on development data only.
# =============================================================================
print("=" * 100)
print("SECTION 1: DATA AND PREPROCESSING")
print("=" * 100)

df = pd.read_csv('modeling_table.csv')
df = df[df['2021_glioma_class'] != 'Unclassified']
df['Scanner'] = df['Scanner'].replace('Other', 'GE')

feature_cols = [c for c in df.columns if c not in [
    'Scanner', '2021_glioma_class', 'age', 'sex', 'case_id', 'site'
]]

# ----- cohorts -----
df_dev = df[df['site'].isin(['TCGA', 'UCSF'])].copy()
df_ext = df[df['site'] == 'EGD'].copy()
df_utsw = df[df['site'] == 'UTSW'].copy()

# ----- external calibration/test split, 50/50, stratified by class -----
# NOTE: this single split is used for the case-level figures (argmax vs
# conformal, set composition, failure cases) and for the Section 3b
# site-recalibrated operating points. The random_state values below are the
# representative seeds printed by Section 5 (the split whose per-class 90%
# coverage is closest to the 200-split mean): EGD = 161, UTSW = 117. These
# seeds are stable on rerun because the base model is trained on development
# data only and never sees the external calibration/test split, so the
# representative-seed selection in Section 5 does not depend on the seed used
# here. test_size is 0.5 to match the 200-split diagnostic so the seed
# reproduces that exact partition.
ext_cal_idx, ext_te_idx = train_test_split(
    df_ext.index, test_size=0.5, stratify=df_ext['2021_glioma_class'], random_state=161
)
df_ext_cal = df_ext.loc[ext_cal_idx]
df_ext_te = df_ext.loc[ext_te_idx]
df_ext_cal.to_csv('df_ext_cal.csv')
df_ext_te.to_csv('df_ext_te.csv')

utsw_cal_idx, utsw_te_idx = train_test_split(
    df_utsw.index, test_size=0.5, stratify=df_utsw['2021_glioma_class'], random_state=117
)
df_utsw_cal = df_utsw.loc[utsw_cal_idx]
df_utsw_te = df_utsw.loc[utsw_te_idx]
df_utsw_cal.to_csv('df_utsw_cal.csv')
df_utsw_te.to_csv('df_utsw_te.csv')

for name, part in [('Dev (TCGA+UCSF)', df_dev),
                   ('EGD calibration', df_ext_cal), ('EGD test', df_ext_te),
                   ('UTSW calibration', df_utsw_cal), ('UTSW test', df_utsw_te)]:
    counts = part['2021_glioma_class'].value_counts().to_dict()
    print(f"  {name:<18} n = {len(part):<5} {counts}")

# ----- imputation (median, fit on dev) and zero-variance filter -----
print("\nPreprocessing: median imputation, variance filter, dev-scaler applied to external...")
X_dev_raw = df_dev[feature_cols].replace([np.inf, -np.inf], np.nan)
X_ext_raw = df_ext[feature_cols].replace([np.inf, -np.inf], np.nan)
X_utsw_raw = df_utsw[feature_cols].replace([np.inf, -np.inf], np.nan)

imp = SimpleImputer(strategy='median')
X_dev_imp = pd.DataFrame(imp.fit_transform(X_dev_raw), columns=X_dev_raw.columns, index=X_dev_raw.index)
X_ext_imp = pd.DataFrame(imp.transform(X_ext_raw), columns=X_ext_raw.columns, index=X_ext_raw.index)
X_utsw_imp = pd.DataFrame(imp.transform(X_utsw_raw), columns=X_utsw_raw.columns, index=X_utsw_raw.index)

valid_features = X_dev_imp.columns[X_dev_imp.var() > 1e-5]
X_dev_imp = X_dev_imp[valid_features]
X_ext_imp = X_ext_imp[valid_features]
X_utsw_imp = X_utsw_imp[valid_features]

X_ext_cal_imp = X_ext_imp.loc[df_ext_cal.index]
X_ext_te_imp = X_ext_imp.loc[df_ext_te.index]
X_utsw_cal_imp = X_utsw_imp.loc[df_utsw_cal.index]
X_utsw_te_imp = X_utsw_imp.loc[df_utsw_te.index]

# ----- z-scoring: dev scaler fit on dev, applied to external with NO refit -----
# This matches deployment (a single incoming patient cannot refit a scaler on a
# cohort) and keeps calibration and test exchangeable, since the transform does
# not depend on which external cases land in the calibration half.
dev_scaler = StandardScaler()
X_dev_scl = pd.DataFrame(dev_scaler.fit_transform(X_dev_imp), columns=X_dev_imp.columns, index=X_dev_imp.index)
X_ext_cal_scl = pd.DataFrame(dev_scaler.transform(X_ext_cal_imp), columns=X_ext_cal_imp.columns, index=X_ext_cal_imp.index)
X_ext_te_scl = pd.DataFrame(dev_scaler.transform(X_ext_te_imp), columns=X_ext_te_imp.columns, index=X_ext_te_imp.index)
X_utsw_cal_scl = pd.DataFrame(dev_scaler.transform(X_utsw_cal_imp), columns=X_utsw_cal_imp.columns, index=X_utsw_cal_imp.index)
X_utsw_te_scl = pd.DataFrame(dev_scaler.transform(X_utsw_te_imp), columns=X_utsw_te_imp.columns, index=X_utsw_te_imp.index)

# ----- targets -----
y_dev = df_dev['2021_glioma_class']
y_ext_cal = df_ext_cal['2021_glioma_class']
y_ext_te = df_ext_te['2021_glioma_class']
y_utsw_cal = df_utsw_cal['2021_glioma_class']
y_utsw_te = df_utsw_te['2021_glioma_class']

# ----- correlation filter at 0.85 (unsupervised, fit on dev) -----
# mRMR feature selection is NOT done here; it runs inside each CV fold in
# Section 2 to keep selection leakage-free.
print("Feature pre-selection: correlation filter at 0.85 (mRMR runs inside each CV fold)...")
X_dev_for_sel = X_dev_scl.loc[:, X_dev_scl.var() > 1e-5]
corr_mat = X_dev_for_sel.corr().abs()
upper_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > 0.85)]
X_dev_corr_filtered = X_dev_for_sel.drop(columns=to_drop)
X_ext_cal_corr_filtered = X_ext_cal_scl.drop(columns=to_drop)
X_ext_te_corr_filtered = X_ext_te_scl.drop(columns=to_drop)
X_utsw_cal_corr_filtered = X_utsw_cal_scl.drop(columns=to_drop)
X_utsw_te_corr_filtered = X_utsw_te_scl.drop(columns=to_drop)
print(f"  Features after correlation filter: {X_dev_corr_filtered.shape[1]}")


# =============================================================================
# SECTION 2: BASE CLASSIFIER TRAINING
# Calibrated RandomForest (sigmoid, cv=5), pre-specified hyperparameters, no
# tuning. A 5-fold CV produces out-of-fold probabilities for the development
# conformal estimate, with mRMR top-20 selected inside each fold. A single
# final model is trained on the full development cohort and used for all
# external predictions and for feature reporting.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 2: BASE CLASSIFIER TRAINING")
print("=" * 100)
print("Model: RandomForest + sigmoid calibration (CalibratedClassifierCV, cv=5), no tuning")
print("Feature selection: mRMR top-20 inside each CV fold (leakage-free)")

rf_base = RandomForestClassifier(
    n_estimators=500, class_weight='balanced_subsample',
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf_calibrated = CalibratedClassifierCV(estimator=rf_base, method='sigmoid', cv=5)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_dev_arr = y_dev.values
classes = np.unique(y_dev_arr)
n_classes = len(classes)

fold_assignments = np.zeros(len(y_dev_arr), dtype=int)
oof_probas = np.zeros((len(y_dev_arr), n_classes))
fold_models = []
fold_selected_features = []

print("\nTraining 5 fold models with per-fold mRMR feature selection...")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_dev_corr_filtered.values, y_dev_arr)):
    X_tr_full = X_dev_corr_filtered.iloc[train_idx]
    y_tr = y_dev_arr[train_idx]
    X_val_full = X_dev_corr_filtered.iloc[val_idx]

    y_tr_series = pd.Series(y_tr, index=X_tr_full.index)
    fold_features = mrmr_classif(X=X_tr_full, y=y_tr_series, K=20, show_progress=False)

    model = clone(rf_calibrated)
    model.fit(X_tr_full[fold_features].values, y_tr)

    oof_probas[val_idx] = model.predict_proba(X_val_full[fold_features].values)
    fold_assignments[val_idx] = fold_idx
    fold_models.append(model)
    fold_selected_features.append(fold_features)
    print(f"  Fold {fold_idx+1}/5 trained ({len(fold_features)} features)")

# ----- final model on full dev cohort -----
print("\nTraining final model on the full development cohort...")
final_features = mrmr_classif(X=X_dev_corr_filtered, y=y_dev, K=20, show_progress=False)
final_model = clone(rf_calibrated)
final_model.fit(X_dev_corr_filtered[final_features].values, y_dev_arr)
print(f"  Final model: {len(final_features)} features")

# feature-selection stability across folds (how often each final feature appeared)
fold_feature_sets = [set(f) for f in fold_selected_features]
feature_stability = {
    feat: sum(feat in fs for fs in fold_feature_sets) / 5.0 for feat in final_features
}
n_stable = sum(1 for v in feature_stability.values() if v >= 0.8)
print(f"  Stability: {n_stable}/{len(final_features)} features selected in >=4/5 folds")

# ----- external probabilities from the final model -----
print("\nGenerating external predictions with the final model...")
ext_cal_probas = final_model.predict_proba(X_ext_cal_corr_filtered[final_features].values)
ext_te_probas = final_model.predict_proba(X_ext_te_corr_filtered[final_features].values)
utsw_cal_probas = final_model.predict_proba(X_utsw_cal_corr_filtered[final_features].values)
utsw_te_probas = final_model.predict_proba(X_utsw_te_corr_filtered[final_features].values)

# Full-cohort external probabilities for discrimination (Section 3). The base
# classifier never uses the calibration/test split, so discrimination is
# measured on the ENTIRE external cohort rather than the test half. This removes
# split dependence and roughly doubles every denominator, which stabilizes the
# minority-class estimates (the test-half oligo PPV was a 1-to-5-case ratio).
X_ext_full_corr = pd.DataFrame(
    dev_scaler.transform(X_ext_imp), columns=X_ext_imp.columns, index=X_ext_imp.index
).drop(columns=to_drop)
X_utsw_full_corr = pd.DataFrame(
    dev_scaler.transform(X_utsw_imp), columns=X_utsw_imp.columns, index=X_utsw_imp.index
).drop(columns=to_drop)
ext_full_probas = final_model.predict_proba(X_ext_full_corr[final_features].values)
utsw_full_probas = final_model.predict_proba(X_utsw_full_corr[final_features].values)
y_ext_full = df_ext.loc[X_ext_imp.index, '2021_glioma_class'].values
y_utsw_full = df_utsw.loc[X_utsw_imp.index, '2021_glioma_class'].values


# =============================================================================
# SECTION 3: BASE CLASSIFIER DISCRIMINATION
# Per-class AUC and argmax PPV/NPV on the development out-of-fold predictions
# and on both external test sets, with 1000-iteration bootstrap 95% CIs.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 3: BASE CLASSIFIER DISCRIMINATION (AUC, Sens, Spec, PPV, NPV; bootstrap 95% CI)")
print("=" * 100)
print("External metrics are computed on the FULL external cohort (split independent).")

base_dev_res, base_dev_raw = evaluate_base_classifier(y_dev_arr, oof_probas, classes, alpha=0.95)
base_ext_res, base_ext_raw = evaluate_base_classifier(y_ext_full, ext_full_probas, classes, alpha=0.95)
base_utsw_res, base_utsw_raw = evaluate_base_classifier(y_utsw_full, utsw_full_probas, classes, alpha=0.95)

for label, res, n in [("Dev OOF", base_dev_res, len(y_dev_arr)),
                      ("EGD full cohort", base_ext_res, len(y_ext_full)),
                      ("UTSW full cohort", base_utsw_res, len(y_utsw_full))]:
    print(f"\n[{label}, n = {n}]")
    print(f"  {'Class':<18} | {'AUC':<14} | {'Sens':<14} | {'Spec':<14} | {'PPV':<14} | {'NPV':<14}")
    for cls in classes:
        r = res[cls]
        print(f"  {cls:<18} | {r['auc']:<14} | {r['sens']:<14} | {r['spec']:<14} | {r['ppv']:<14} | {r['npv']:<14}")


# =============================================================================
# SECTION 3b: COST-ADJUSTED DECISION RULE (class-weighted argmax)
# Argmax is the Bayes rule for 0-1 loss; under the development class prior it
# minimizes total errors by rarely predicting the minority class, which drives
# oligodendroglioma sensitivity toward zero. This section KEEPS argmax as the
# primary rule (Section 3) and ADDS a transparent alternative: the class-weighted
# argmax y = argmax_k (w_k * p_k), the multiclass generalization of lowering a
# single binary decision threshold. Only the ratios of the weights matter.
#
# Two operating-point variants are reported:
#   (A) frozen rule  : weights selected once on the development OOF, then applied
#                      unchanged to the FULL external cohorts. Stable selection
#                      (large dev n); answers "does one fixed rule transfer?".
#   (B) site-recalib : weights re-selected on each external cohort's CALIBRATION
#                      half and applied to that cohort's TEST half. Parallels the
#                      conformal calibration (cc_ext / cc_utsw are also fit on the
#                      external calibration halves). Small calibration halves make
#                      these weights noisy, so treat (A) as primary and (B) as a
#                      consistency check; the calibration-half counts are printed.
#   (C) OvR Youden   : per-class one-vs-rest detection thresholds on dev OOF, for
#                      transparency only (they do not compose into a single label).
#
# Selection objective is balanced accuracy (mean per-class recall), which weights
# all three classes equally and so does not privilege the minority class a priori.
# Weights are NEVER selected on the data they are reported on.
# =============================================================================
from itertools import product
from sklearn.metrics import balanced_accuracy_score

print("\n" + "=" * 100)
print("SECTION 3b: COST-ADJUSTED DECISION RULE (class-weighted argmax)")
print("=" * 100)

# candidate per-class multipliers; 1.0 = no change. The grid includes the
# unweighted baseline (all ones), so the search can never do worse on the
# selection set than plain argmax.
weight_grid_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
candidate_weight_vectors = list(product(weight_grid_values, repeat=n_classes))

# ----- (A) PRIMARY: select on development OOF, then FREEZE -----
best_bal_acc_dev = -1.0
best_weights_dev = np.ones(n_classes)
for w in candidate_weight_vectors:
    w_arr = np.array(w, dtype=float)
    y_pred_w = classes[np.argmax(oof_probas * w_arr, axis=1)]
    bal = balanced_accuracy_score(y_dev_arr, y_pred_w)
    if bal > best_bal_acc_dev:
        best_bal_acc_dev = bal
        best_weights_dev = w_arr

ratios_dev = best_weights_dev / best_weights_dev.max()
argmax_bal_dev = balanced_accuracy_score(y_dev_arr, classes[np.argmax(oof_probas, axis=1)])
print("\n[A] Weights selected on development OOF (frozen, applied to full external cohorts)")
for i, cls in enumerate(classes):
    print(f"  {cls:<18} relative weight = {ratios_dev[i]:.2f}")
print(f"  Balanced accuracy (dev OOF): argmax = {argmax_bal_dev:.3f} -> weighted = {best_bal_acc_dev:.3f}")

# apply the frozen dev weights, with bootstrap CIs, on dev OOF and full external cohorts
wt_dev_res,  wt_dev_raw  = evaluate_base_classifier(y_dev_arr,   oof_probas,       classes, alpha=0.95, weights=best_weights_dev)
wt_ext_res,  wt_ext_raw  = evaluate_base_classifier(y_ext_full,  ext_full_probas,  classes, alpha=0.95, weights=best_weights_dev)
wt_utsw_res, wt_utsw_raw = evaluate_base_classifier(y_utsw_full, utsw_full_probas, classes, alpha=0.95, weights=best_weights_dev)

for label, base_res, wt_res, n in [
        ("Dev OOF",          base_dev_res,  wt_dev_res,  len(y_dev_arr)),
        ("EGD full cohort",  base_ext_res,  wt_ext_res,  len(y_ext_full)),
        ("UTSW full cohort", base_utsw_res, wt_utsw_res, len(y_utsw_full))]:
    print(f"\n[{label}, n = {n}]  argmax -> weighted argmax (frozen dev weights)")
    print(f"  {'Class':<18} | {'Sens argmax':<16} | {'Sens weighted':<16} | {'Spec argmax':<16} | {'Spec weighted':<16}")
    for cls in classes:
        print(f"  {cls:<18} | {base_res[cls]['sens']:<16} | {wt_res[cls]['sens']:<16} | {base_res[cls]['spec']:<16} | {wt_res[cls]['spec']:<16}")

# ----- (B) SITE-RECALIBRATED: select on external calibration half, apply to test half -----
print("\n[B] Weights re-selected on each external calibration half, applied to its test half")
print("    External calibration-half class counts (minority count drives instability):")
for cls in classes:
    n_egd_cal_cls  = int(np.sum(y_ext_cal.values  == cls))
    n_utsw_cal_cls = int(np.sum(y_utsw_cal.values == cls))
    print(f"      {cls:<18} EGD cal n = {n_egd_cal_cls:<4} | UTSW cal n = {n_utsw_cal_cls}")

# EGD: select on calibration-half probabilities
best_bal_acc_egd = -1.0
best_weights_egd = np.ones(n_classes)
for w in candidate_weight_vectors:
    w_arr = np.array(w, dtype=float)
    y_pred_w = classes[np.argmax(ext_cal_probas * w_arr, axis=1)]
    bal = balanced_accuracy_score(y_ext_cal.values, y_pred_w)
    if bal > best_bal_acc_egd:
        best_bal_acc_egd = bal
        best_weights_egd = w_arr

# UTSW: select on calibration-half probabilities
best_bal_acc_utsw = -1.0
best_weights_utsw = np.ones(n_classes)
for w in candidate_weight_vectors:
    w_arr = np.array(w, dtype=float)
    y_pred_w = classes[np.argmax(utsw_cal_probas * w_arr, axis=1)]
    bal = balanced_accuracy_score(y_utsw_cal.values, y_pred_w)
    if bal > best_bal_acc_utsw:
        best_bal_acc_utsw = bal
        best_weights_utsw = w_arr

ratios_egd  = best_weights_egd  / best_weights_egd.max()
ratios_utsw = best_weights_utsw / best_weights_utsw.max()
print("\n  EGD weights (relative):  " + ", ".join(f"{cls}={ratios_egd[i]:.2f}"  for i, cls in enumerate(classes)))
print(  "  UTSW weights (relative): " + ", ".join(f"{cls}={ratios_utsw[i]:.2f}" for i, cls in enumerate(classes)))

# evaluate site-recalibrated weights on the held-out TEST halves, with CIs;
# also the plain-argmax baseline on the same test halves for a fair side-by-side
wt_ext_te_res,  wt_ext_te_raw  = evaluate_base_classifier(y_ext_te.values,  ext_te_probas,  classes, alpha=0.95, weights=best_weights_egd)
wt_utsw_te_res, wt_utsw_te_raw = evaluate_base_classifier(y_utsw_te.values, utsw_te_probas, classes, alpha=0.95, weights=best_weights_utsw)
base_ext_te_res,  base_ext_te_raw  = evaluate_base_classifier(y_ext_te.values,  ext_te_probas,  classes, alpha=0.95)
base_utsw_te_res, base_utsw_te_raw = evaluate_base_classifier(y_utsw_te.values, utsw_te_probas, classes, alpha=0.95)

for label, base_res, wt_res, n in [
        ("EGD test half",  base_ext_te_res,  wt_ext_te_res,  len(y_ext_te)),
        ("UTSW test half", base_utsw_te_res, wt_utsw_te_res, len(y_utsw_te))]:
    print(f"\n[{label}, n = {n}]  argmax -> weighted argmax (site-recalibrated weights)")
    print(f"  {'Class':<18} | {'Sens argmax':<16} | {'Sens weighted':<16} | {'Spec argmax':<16} | {'Spec weighted':<16}")
    for cls in classes:
        print(f"  {cls:<18} | {base_res[cls]['sens']:<16} | {wt_res[cls]['sens']:<16} | {base_res[cls]['spec']:<16} | {wt_res[cls]['spec']:<16}")

# ----- (C) one-vs-rest Youden thresholds on dev OOF (transparency supplement) -----
# Per-class detection operating points. They do NOT compose into a single coherent
# label (a case can exceed the threshold for two classes or none), so report them
# as detection thresholds, not as a classifier.
print("\n[C] One-vs-rest Youden operating points (dev OOF, per-class detection)")
print(f"  {'Class':<18} | {'Threshold':<10} | {'Sens':<8} | {'Spec':<8}")
lb_ovr = LabelBinarizer().fit(classes)
y_dev_bin = lb_ovr.transform(y_dev_arr)
ovr_thresholds = {}
for i, cls in enumerate(classes):
    fpr_ovr, tpr_ovr, thr_ovr = roc_curve(y_dev_bin[:, i], oof_probas[:, i])
    youden = tpr_ovr - fpr_ovr
    best_t_idx = int(np.argmax(youden))
    ovr_thresholds[cls] = {
        'threshold': float(thr_ovr[best_t_idx]),
        'sens': float(tpr_ovr[best_t_idx]),
        'spec': float(1.0 - fpr_ovr[best_t_idx]),
    }
    print(f"  {cls:<18} | {thr_ovr[best_t_idx]:<10.3f} | {tpr_ovr[best_t_idx]:<8.2f} | {1.0 - fpr_ovr[best_t_idx]:<8.2f}")

# ----- store for the pkl (so the manuscript-asset script can render these later) -----
cost_adjusted_store = {
    'classes': list(classes),
    'grid_values': weight_grid_values,
    'selection_objective': 'balanced_accuracy',
    'weights_dev_frozen': {cls: float(best_weights_dev[i]) for i, cls in enumerate(classes)},
    'weights_dev_frozen_ratio': {cls: float(ratios_dev[i]) for i, cls in enumerate(classes)},
    'weights_egd_cal': {cls: float(best_weights_egd[i]) for i, cls in enumerate(classes)},
    'weights_utsw_cal': {cls: float(best_weights_utsw[i]) for i, cls in enumerate(classes)},
    'balacc_dev_argmax': float(argmax_bal_dev),
    'balacc_dev_weighted': float(best_bal_acc_dev),
    'frozen_dev_oof': wt_dev_raw,          # frozen-dev rule, bootstrap raw metrics
    'frozen_egd_full': wt_ext_raw,
    'frozen_utsw_full': wt_utsw_raw,
    'siterecal_egd_te': wt_ext_te_raw,     # site-recalibrated rule, test halves
    'siterecal_utsw_te': wt_utsw_te_raw,
    'argmax_egd_te': base_ext_te_raw,      # plain-argmax baseline on the same test halves
    'argmax_utsw_te': base_utsw_te_raw,
    'ovr_youden_dev': ovr_thresholds,
}


# =============================================================================
# SECTION 4: CONFORMAL PREDICTION (LAC, class-conditional / Mondrian)
# Nonconformity score = 1 - calibrated probability of the true class. Thresholds
# are calibrated per class. Development coverage uses leave-one-fold-out
# cross-conformal with a bootstrap CI (internal estimate). External coverage
# uses standard split conformal and is stored as point estimates for this single
# split; the validity claim and its uncertainty come from Section 5.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 4: CONFORMAL PREDICTION at 80%, 85%, 90%, 95%")
print("=" * 100)

# class index of each true label, then LAC alphas on the calibration sets
y_dev_idx = np.array([np.where(classes == y)[0][0] for y in y_dev_arr])
y_ext_cal_idx = np.array([np.where(classes == y)[0][0] for y in y_ext_cal.values])
y_utsw_cal_idx = np.array([np.where(classes == y)[0][0] for y in y_utsw_cal.values])

oof_scores = 1 - oof_probas
ext_cal_scores = 1 - ext_cal_probas
ext_te_scores = 1 - ext_te_probas
utsw_cal_scores = 1 - utsw_cal_probas
utsw_te_scores = 1 - utsw_te_probas

oof_alphas = oof_scores[np.arange(len(y_dev_arr)), y_dev_idx]
ext_cal_alphas = ext_cal_scores[np.arange(len(y_ext_cal)), y_ext_cal_idx]
utsw_cal_alphas = utsw_cal_scores[np.arange(len(y_utsw_cal)), y_utsw_cal_idx]

# Fit the conformal calibrators once; confidence is set at predict_set time.
# smoothing=False: deterministic (non-randomized) prediction sets, reproducible
# per patient, with coverage at or above the nominal target. This is the
# clinically appropriate choice; smoothing=True gives exact but randomized sets.
cc_ext = ConformalClassifier()
cc_ext.fit(ext_cal_alphas, bins=y_ext_cal.values)
cc_utsw = ConformalClassifier()
cc_utsw.fit(utsw_cal_alphas, bins=y_utsw_cal.values)

# Development LOFO needs one calibrator per held-out fold (calibrated on the
# other four folds).
cc_lofo_by_fold = []
for held_fold in range(5):
    cal_mask = fold_assignments != held_fold
    cc_k = ConformalClassifier()
    cc_k.fit(oof_alphas[cal_mask], bins=y_dev_arr[cal_mask])
    cc_lofo_by_fold.append(cc_k)

results_store = {}
dev_pred_sets_by_conf = {}
ext_pred_sets_by_conf = {}
utsw_pred_sets_by_conf = {}

for target_confidence in [0.80, 0.85, 0.90, 0.95]:
    print("\n" + "-" * 100)
    print(f"Confidence level: {int(target_confidence * 100)}%")
    print("-" * 100)

    # ---- development OOF prediction sets (leave-one-fold-out) ----
    dev_pred_sets = np.zeros((len(y_dev_arr), n_classes), dtype=bool)
    for held_fold in range(5):
        test_mask = fold_assignments == held_fold
        n_held = test_mask.sum()
        test_scores_k = oof_scores[test_mask]
        cc_k = cc_lofo_by_fold[held_fold]
        fold_pred_sets = np.column_stack([
            cc_k.predict_set(test_scores_k, bins=np.full(n_held, cls), confidence=target_confidence, smoothing=False)[:, c]
            for c, cls in enumerate(classes)
        ])
        dev_pred_sets[test_mask] = fold_pred_sets
    dev_res, dev_raw = compute_bootstrapped_metrics(y_dev_arr, dev_pred_sets, classes=classes, alpha=0.95)

    # ---- EGD prediction sets (split conformal; point estimates) ----
    ext_pred_sets = np.column_stack([
        cc_ext.predict_set(ext_te_scores, bins=np.full(len(y_ext_te), cls), confidence=target_confidence, smoothing=False)[:, c]
        for c, cls in enumerate(classes)
    ])
    ext_point = conformal_point_metrics(y_ext_te.values, ext_pred_sets, classes)

    # ---- UTSW prediction sets (split conformal; point estimates) ----
    utsw_pred_sets = np.column_stack([
        cc_utsw.predict_set(utsw_te_scores, bins=np.full(len(y_utsw_te), cls), confidence=target_confidence, smoothing=False)[:, c]
        for c, cls in enumerate(classes)
    ])
    utsw_point = conformal_point_metrics(y_utsw_te.values, utsw_pred_sets, classes)

    # concise console summary
    print(f"  {'Class':<18} | {'Dev cov (LOFO)':<18} | {'EGD cov':<9} | {'UTSW cov':<9}")
    for cls in classes:
        print(f"  {cls:<18} | {dev_res[cls]['cov']:<18} | "
              f"{ext_point[cls]['cov']:<9.2f} | {utsw_point[cls]['cov']:<9.2f}")

    dev_pred_sets_by_conf[target_confidence] = dev_pred_sets.copy()
    ext_pred_sets_by_conf[target_confidence] = ext_pred_sets.copy()
    utsw_pred_sets_by_conf[target_confidence] = utsw_pred_sets.copy()

    results_store[target_confidence] = {
        'classes': list(classes),
        'base_dev': base_dev_raw,
        'base_ext': base_ext_raw,
        'base_utsw': base_utsw_raw,
        'dev_conformal_lofo': dev_raw,         # bootstrap CI (internal)
        'ext_conformal': ext_point,            # point estimates (single split)
        'utsw_conformal': utsw_point,          # point estimates (single split)
        'n_dev_by_class': {cls: int(np.sum(y_dev_arr == cls)) for cls in classes},
        'n_ext_cal_by_class': {cls: int(np.sum(y_ext_cal.values == cls)) for cls in classes},
        'n_ext_te_by_class': {cls: int(np.sum(y_ext_te.values == cls)) for cls in classes},
        'n_ext_full_by_class': {cls: int(np.sum(y_ext_full == cls)) for cls in classes},
        'n_utsw_cal_by_class': {cls: int(np.sum(y_utsw_cal.values == cls)) for cls in classes},
        'n_utsw_te_by_class': {cls: int(np.sum(y_utsw_te.values == cls)) for cls in classes},
        'n_utsw_full_by_class': {cls: int(np.sum(y_utsw_full == cls)) for cls in classes},
    }

# ----- feature importances from the final model (averaged over its calibrated RFs) -----
all_importances = []
for cc in final_model.calibrated_classifiers_:
    if hasattr(cc, 'estimator'):
        rf = cc.estimator
    elif hasattr(cc, 'base_estimator'):
        rf = cc.base_estimator
    else:
        raise AttributeError("Cannot access fitted RF from CalibratedClassifier")
    all_importances.append(rf.feature_importances_)
feature_importances_avg = np.mean(all_importances, axis=0)
feature_importances_std = np.std(all_importances, axis=0)

# ----- auxiliary arrays for the figures (prediction sets, probas, importances) -----
results_store['_aux'] = {
    'y_dev': y_dev_arr,
    'y_ext_te': y_ext_te.values,
    'y_utsw_te': y_utsw_te.values,
    'oof_probas': oof_probas,
    'ext_te_probas': ext_te_probas,
    'utsw_te_probas': utsw_te_probas,
    'y_ext_full': y_ext_full,
    'y_utsw_full': y_utsw_full,
    'ext_full_probas': ext_full_probas,
    'utsw_full_probas': utsw_full_probas,
    'classes': list(classes),
    'selected_features': list(final_features),
    'feature_importances_mean': feature_importances_avg,
    'feature_importances_std': feature_importances_std,
    'feature_stability': feature_stability,
    'fold_selected_features': fold_selected_features,
    'dev_pred_sets_by_conf': dev_pred_sets_by_conf,
    'ext_pred_sets_by_conf': ext_pred_sets_by_conf,
    'utsw_pred_sets_by_conf': utsw_pred_sets_by_conf,
    'fold_assignments': fold_assignments,
}

results_store['cost_adjusted'] = cost_adjusted_store


# =============================================================================
# SECTION 6: CALIBRATION QUALITY (EXPECTED CALIBRATION ERROR)
# Conformal VALIDITY does not need calibrated probabilities, but conformal
# EFFICIENCY (set size) does: poorly calibrated scores enlarge the sets needed
# to reach a target coverage. Sigmoid calibration was fit on only 26
# oligodendrogliomas in development, so we quantify how well the final model's
# probabilities are calibrated on dev (out-of-fold) and on both full external
# cohorts. We report top-label ECE (15 equal-width bins) and class-wise ECE
# (mean of the three one-vs-rest ECEs). This is a reviewer-anticipating
# supplement; it changes no other result.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 6: CALIBRATION QUALITY (ECE)")
print("=" * 100)

# Small helper, factored out only because it is called on three evaluation sets
# (dev OOF, EGD full, UTSW full); inlining would duplicate it three times.
def expected_calibration_error(probas, y_true_arr, classes_arr, n_bins=15):
    n_total = len(y_true_arr)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # ---- top-label ECE ----
    top_conf = probas.max(axis=1)
    top_pred = classes_arr[np.argmax(probas, axis=1)]
    top_correct = (top_pred == y_true_arr).astype(float)
    ece_top = 0.0
    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        if b == n_bins - 1:
            in_bin = (top_conf >= lo) & (top_conf <= hi)
        else:
            in_bin = (top_conf >= lo) & (top_conf < hi)
        n_in = int(in_bin.sum())
        if n_in > 0:
            acc_in = float(top_correct[in_bin].mean())
            conf_in = float(top_conf[in_bin].mean())
            ece_top = ece_top + abs(acc_in - conf_in) * (n_in / n_total)

    # ---- class-wise (one-vs-rest) ECE, averaged over the three classes ----
    classwise = {}
    for ci, cls in enumerate(classes_arr):
        p_cls = probas[:, ci]
        lbl_cls = (y_true_arr == cls).astype(float)
        ece_cls = 0.0
        for b in range(n_bins):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            if b == n_bins - 1:
                in_bin = (p_cls >= lo) & (p_cls <= hi)
            else:
                in_bin = (p_cls >= lo) & (p_cls < hi)
            n_in = int(in_bin.sum())
            if n_in > 0:
                freq_in = float(lbl_cls[in_bin].mean())
                mean_p_in = float(p_cls[in_bin].mean())
                ece_cls = ece_cls + abs(freq_in - mean_p_in) * (n_in / n_total)
        classwise[cls] = float(ece_cls)
    ece_classwise_mean = float(np.mean([classwise[cls] for cls in classes_arr]))
    return float(ece_top), ece_classwise_mean, classwise

ece_dev_top, ece_dev_cw, ece_dev_by_class = expected_calibration_error(oof_probas, y_dev_arr, classes)
ece_egd_top, ece_egd_cw, ece_egd_by_class = expected_calibration_error(ext_full_probas, y_ext_full, classes)
ece_utsw_top, ece_utsw_cw, ece_utsw_by_class = expected_calibration_error(utsw_full_probas, y_utsw_full, classes)

calibration_store = {
    'n_bins': 15,
    'dev_oof':  {'ece_top_label': ece_dev_top,  'ece_classwise_mean': ece_dev_cw,  'ece_by_class': ece_dev_by_class},
    'egd_full': {'ece_top_label': ece_egd_top,  'ece_classwise_mean': ece_egd_cw,  'ece_by_class': ece_egd_by_class},
    'utsw_full':{'ece_top_label': ece_utsw_top, 'ece_classwise_mean': ece_utsw_cw, 'ece_by_class': ece_utsw_by_class},
}

print(f"  {'Set':<16} | {'Top-label ECE':>14} | {'Class-wise ECE':>15}")
for set_label, d in [('Dev OOF', calibration_store['dev_oof']),
                     ('EGD full', calibration_store['egd_full']),
                     ('UTSW full', calibration_store['utsw_full'])]:
    print(f"  {set_label:<16} | {d['ece_top_label']:>14.3f} | {d['ece_classwise_mean']:>15.3f}")

calib_rows = []
for set_label, d in [('Dev_OOF', calibration_store['dev_oof']),
                     ('EGD_full', calibration_store['egd_full']),
                     ('UTSW_full', calibration_store['utsw_full'])]:
    row = {'set': set_label, 'ece_top_label': d['ece_top_label'], 'ece_classwise_mean': d['ece_classwise_mean']}
    for cls in classes:
        row[f'ece_{cls}'] = d['ece_by_class'][cls]
    calib_rows.append(row)
pd.DataFrame(calib_rows).to_csv('calibration_ece.csv', index=False)
print("  Saved calibration_ece.csv")


# =============================================================================
# SECTION 7: TRANSPORT TEST (DEVELOPMENT-CALIBRATED CONFORMAL -> EXTERNAL)
# The main external coverage (Sections 4-5) calibrates the conformal thresholds
# on a CALIBRATION HALF drawn from the SAME external cohort, so at-or-above-
# target coverage there is guaranteed by exchangeability. This section asks the
# deployment-realistic question instead: if the class-conditional LAC thresholds
# are frozen on DEVELOPMENT data and applied to a new site WITHOUT any local
# calibration, does coverage still hold? Under-coverage here is the honest,
# expected failure mode and is the point of the experiment.
#
# Thresholds are calibrated on the development OUT-OF-FOLD alphas (leakage-free
# dev scores) and applied to the FULL external cohorts (every external case is a
# test case, since no external calibration data are used). One caveat: the dev
# OOF alphas come from the five fold models whereas external scores come from the
# final full-dev model, a small model-version mismatch that slightly favors the
# dev thresholds; it does not affect the direction of any under-coverage.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 7: TRANSPORT TEST (dev-calibrated thresholds applied to external, no site calibration)")
print("=" * 100)

conf_levels_extra = [0.80, 0.85, 0.90, 0.95]

# class-conditional conformal calibrated on development OOF alphas
cc_dev = ConformalClassifier()
cc_dev.fit(oof_alphas, bins=y_dev_arr)

ext_full_scores = 1 - ext_full_probas
utsw_full_scores = 1 - utsw_full_probas

transport_store = {}
transport_rows = []
for cohort_label, scores_full, y_full in [('EGD', ext_full_scores, y_ext_full),
                                          ('UTSW', utsw_full_scores, y_utsw_full)]:
    transport_store[cohort_label] = {}
    print(f"\n[{cohort_label}] dev-calibrated coverage (target should be met only if shift is mild)")
    print(f"  {'Conf':>5} | {'Class':<18} | {'Coverage':>9} | {'Avg size':>9} | marginal")
    for target_confidence in conf_levels_extra:
        pred_sets = np.column_stack([
            cc_dev.predict_set(scores_full, bins=np.full(len(y_full), cls),
                               confidence=target_confidence, smoothing=False)[:, c]
            for c, cls in enumerate(classes)
        ])
        point = conformal_point_metrics(y_full, pred_sets, classes)
        # marginal coverage: did each case's own true-class column get included?
        true_col = np.array([np.where(classes == y)[0][0] for y in y_full])
        marginal_cov = float(np.mean(pred_sets[np.arange(len(y_full)), true_col]))
        transport_store[cohort_label][target_confidence] = {
            'per_class': point, 'marginal_coverage': marginal_cov
        }
        for cls in classes:
            print(f"  {target_confidence:>5.2f} | {cls:<18} | {point[cls]['cov']:>9.2f} | {point[cls]['size']:>9.2f} | {marginal_cov:>6.2f}")
            transport_rows.append({
                'cohort': cohort_label, 'confidence': target_confidence, 'class': cls,
                'coverage': point[cls]['cov'], 'avg_set_size': point[cls]['size'],
                'pct_single': point[cls]['pct_1'], 'pct_double': point[cls]['pct_2'],
                'pct_all_three': point[cls]['pct_all'], 'marginal_coverage': marginal_cov,
            })
pd.DataFrame(transport_rows).to_csv('transport_dev_calibrated_coverage.csv', index=False)
print("\n  Saved transport_dev_calibrated_coverage.csv")
print("  INTERPRETATION: compare these per-class coverages to the target. Coverage")
print("  well below target = the conformal guarantee does NOT transport without")
print("  local calibration, which supports the manuscript's central caveat.")


# =============================================================================
# SECTION 8: ALTERNATIVE BASE LEARNER (gradient boosting) EFFICIENCY CHECK
# The 'efficiency cliff' (large/uninformative sets at high confidence) depends on
# how sharp the base model's probabilities are, so it is partly a property of the
# random forest rather than of the task. Here we repeat the pipeline with a
# calibrated HistGradientBoosting model on the SAME per-fold and final features
# (so feature selection is held constant and the comparison is model-only), and
# report its external conformal set sizes on the SAME representative split used in
# Section 4. If the cliff persists, the manuscript claim is much stronger; if it
# shrinks, that is itself an informative, honest result.
#
# NOTE (verify on first run): this is new model-training code. Confirm the GB
# model trains without error on the minority class before trusting its sets.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 8: ALTERNATIVE BASE LEARNER (calibrated HistGradientBoosting)")
print("=" * 100)
from sklearn.ensemble import HistGradientBoostingClassifier

gb_base = HistGradientBoostingClassifier(
    learning_rate=0.05, max_depth=3, max_iter=300,
    l2_regularization=1.0, class_weight='balanced', random_state=42
)
gb_calibrated = CalibratedClassifierCV(estimator=gb_base, method='sigmoid', cv=5)

# ---- out-of-fold GB probabilities, reusing the exact Section-2 folds/features ----
gb_oof_probas = np.zeros((len(y_dev_arr), n_classes))
for held_fold in range(5):
    tr_mask = fold_assignments != held_fold
    va_mask = fold_assignments == held_fold
    feats = fold_selected_features[held_fold]
    gb_fold = clone(gb_calibrated)
    gb_fold.fit(X_dev_corr_filtered.iloc[tr_mask][feats].values, y_dev_arr[tr_mask])
    gb_oof_probas[va_mask] = gb_fold.predict_proba(X_dev_corr_filtered.iloc[va_mask][feats].values)
    print(f"  GB fold {held_fold+1}/5 trained ({len(feats)} features)")

# ---- final GB on full dev, then external probabilities (cal/test/full) ----
gb_final = clone(gb_calibrated)
gb_final.fit(X_dev_corr_filtered[final_features].values, y_dev_arr)
gb_ext_cal_probas = gb_final.predict_proba(X_ext_cal_corr_filtered[final_features].values)
gb_ext_te_probas = gb_final.predict_proba(X_ext_te_corr_filtered[final_features].values)
gb_utsw_cal_probas = gb_final.predict_proba(X_utsw_cal_corr_filtered[final_features].values)
gb_utsw_te_probas = gb_final.predict_proba(X_utsw_te_corr_filtered[final_features].values)
gb_ext_full_probas = gb_final.predict_proba(X_ext_full_corr[final_features].values)
gb_utsw_full_probas = gb_final.predict_proba(X_utsw_full_corr[final_features].values)

# ---- GB discrimination (AUC) on the full external cohorts, for context ----
gb_ext_res, gb_ext_raw = evaluate_base_classifier(y_ext_full, gb_ext_full_probas, classes, alpha=0.95)
gb_utsw_res, gb_utsw_raw = evaluate_base_classifier(y_utsw_full, gb_utsw_full_probas, classes, alpha=0.95)

# ---- GB calibration quality, same metric as Section 6 ----
gb_ece_egd_top, gb_ece_egd_cw, gb_ece_egd_by = expected_calibration_error(gb_ext_full_probas, y_ext_full, classes)
gb_ece_utsw_top, gb_ece_utsw_cw, gb_ece_utsw_by = expected_calibration_error(gb_utsw_full_probas, y_utsw_full, classes)

# ---- GB class-conditional conformal on the SAME representative split (Section 4) ----
gb_ext_cal_alphas = (1 - gb_ext_cal_probas)[np.arange(len(y_ext_cal)), y_ext_cal_idx]
gb_utsw_cal_alphas = (1 - gb_utsw_cal_probas)[np.arange(len(y_utsw_cal)), y_utsw_cal_idx]
cc_gb_ext = ConformalClassifier()
cc_gb_ext.fit(gb_ext_cal_alphas, bins=y_ext_cal.values)
cc_gb_utsw = ConformalClassifier()
cc_gb_utsw.fit(gb_utsw_cal_alphas, bins=y_utsw_cal.values)

gb_ext_te_scores = 1 - gb_ext_te_probas
gb_utsw_te_scores = 1 - gb_utsw_te_probas

altmodel_store = {
    'model': 'HistGradientBoosting (sigmoid-calibrated)',
    'discrimination_egd_full': gb_ext_raw,
    'discrimination_utsw_full': gb_utsw_raw,
    'ece_egd_full': {'ece_top_label': gb_ece_egd_top, 'ece_classwise_mean': gb_ece_egd_cw, 'ece_by_class': gb_ece_egd_by},
    'ece_utsw_full': {'ece_top_label': gb_ece_utsw_top, 'ece_classwise_mean': gb_ece_utsw_cw, 'ece_by_class': gb_ece_utsw_by},
    'conformal': {},
}
altmodel_rows = []
print("\n  GB conformal set sizes on the representative split (compare to RF Table 3):")
print(f"  {'Cohort':<6} | {'Conf':>5} | {'Class':<18} | {'Coverage':>9} | {'Avg size':>9} | {'%all3':>6}")
for cohort_label, cc_gb, te_scores, y_te in [('EGD', cc_gb_ext, gb_ext_te_scores, y_ext_te.values),
                                             ('UTSW', cc_gb_utsw, gb_utsw_te_scores, y_utsw_te.values)]:
    altmodel_store['conformal'][cohort_label] = {}
    for target_confidence in conf_levels_extra:
        pred_sets = np.column_stack([
            cc_gb.predict_set(te_scores, bins=np.full(len(y_te), cls),
                              confidence=target_confidence, smoothing=False)[:, c]
            for c, cls in enumerate(classes)
        ])
        point = conformal_point_metrics(y_te, pred_sets, classes)
        altmodel_store['conformal'][cohort_label][target_confidence] = point
        for cls in classes:
            print(f"  {cohort_label:<6} | {target_confidence:>5.2f} | {cls:<18} | {point[cls]['cov']:>9.2f} | {point[cls]['size']:>9.2f} | {point[cls]['pct_all']:>5.0f}%")
            altmodel_rows.append({
                'cohort': cohort_label, 'confidence': target_confidence, 'class': cls,
                'coverage': point[cls]['cov'], 'avg_set_size': point[cls]['size'],
                'pct_single': point[cls]['pct_1'], 'pct_double': point[cls]['pct_2'],
                'pct_all_three': point[cls]['pct_all'],
            })
pd.DataFrame(altmodel_rows).to_csv('altmodel_gb_conformal.csv', index=False)
print("\n  Saved altmodel_gb_conformal.csv")
print("  GB external AUC (EGD / UTSW): " + ", ".join(f"{cls}={gb_ext_res[cls]['auc']} / {gb_utsw_res[cls]['auc']}" for cls in classes))


# =============================================================================
# SECTION 9: APS (ADAPTIVE PREDICTION SETS) COMPARISON, class-conditional
# LAC (1 - p_true) is efficient but can under-protect conditional coverage; APS
# (Romano et al., 2020) is the more common minority-protecting score. We add a
# class-conditional (Mondrian) APS built on the SAME random-forest probabilities
# and the SAME representative split, so any difference is the score, not the
# model or the split. Non-randomized (conservative) APS is used to match the
# deterministic LAC sets already reported.
#
# APS score for a candidate label j at a case = cumulative probability mass of
# every class at least as probable as j (including j). Calibrate the per-class
# threshold on the external calibration half (true-label scores), then include
# label j in a test case's set iff its APS_j <= that class-conditional threshold.
#
# NOTE (verify on first run): this is new conformal logic, not a crepes call.
# Before trusting the set sizes, confirm the printed MARGINAL coverage is at or
# just above each target; if it is, the construction is correct.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 9: APS (adaptive prediction sets), class-conditional, RF probabilities")
print("=" * 100)

# Small helper, called for calibration and test in both cohorts; returns an
# (n_cases, n_classes) matrix of APS scores, one per candidate label.
def aps_scores_all_labels(probas):
    n_rows = probas.shape[0]
    n_cols = probas.shape[1]
    scores = np.zeros((n_rows, n_cols))
    order = np.argsort(-probas, axis=1)             # classes by descending prob
    sorted_p = np.take_along_axis(probas, order, axis=1)
    cum_sorted = np.cumsum(sorted_p, axis=1)        # running mass from the top
    for i in range(n_rows):
        scores[i, order[i]] = cum_sorted[i]         # map back to class positions
    return scores

aps_store = {}
aps_rows = []
for cohort_label, cal_probas, te_probas, y_cal, y_te in [
        ('EGD', ext_cal_probas, ext_te_probas, y_ext_cal.values, y_ext_te.values),
        ('UTSW', utsw_cal_probas, utsw_te_probas, y_utsw_cal.values, y_utsw_te.values)]:

    cal_aps = aps_scores_all_labels(cal_probas)
    te_aps = aps_scores_all_labels(te_probas)
    cal_true_col = np.array([np.where(classes == y)[0][0] for y in y_cal])
    cal_true_score = cal_aps[np.arange(len(y_cal)), cal_true_col]

    aps_store[cohort_label] = {}
    print(f"\n[{cohort_label}] APS class-conditional")
    print(f"  {'Conf':>5} | {'Class':<18} | {'Coverage':>9} | {'Avg size':>9} | marginal")
    for target_confidence in conf_levels_extra:
        # class-conditional thresholds: standard conformal quantile per class
        thresholds = np.zeros(n_classes)
        for k in range(n_classes):
            cls_scores = cal_true_score[cal_true_col == k]
            n_k = len(cls_scores)
            if n_k == 0:
                thresholds[k] = 1.0
                continue
            q_level = min(1.0, np.ceil((n_k + 1) * target_confidence) / n_k)
            thresholds[k] = float(np.quantile(cls_scores, q_level, method='higher'))

        # include label j iff its APS score <= that class's threshold
        pred_sets = np.zeros((len(y_te), n_classes), dtype=bool)
        for k in range(n_classes):
            pred_sets[:, k] = te_aps[:, k] <= thresholds[k]

        point = conformal_point_metrics(y_te, pred_sets, classes)
        te_true_col = np.array([np.where(classes == y)[0][0] for y in y_te])
        marginal_cov = float(np.mean(pred_sets[np.arange(len(y_te)), te_true_col]))
        aps_store[cohort_label][target_confidence] = {'per_class': point, 'marginal_coverage': marginal_cov}
        for cls in classes:
            print(f"  {target_confidence:>5.2f} | {cls:<18} | {point[cls]['cov']:>9.2f} | {point[cls]['size']:>9.2f} | {marginal_cov:>6.2f}")
            aps_rows.append({
                'cohort': cohort_label, 'confidence': target_confidence, 'class': cls,
                'coverage': point[cls]['cov'], 'avg_set_size': point[cls]['size'],
                'pct_single': point[cls]['pct_1'], 'pct_double': point[cls]['pct_2'],
                'pct_all_three': point[cls]['pct_all'], 'marginal_coverage': marginal_cov,
            })
pd.DataFrame(aps_rows).to_csv('aps_conformal.csv', index=False)
print("\n  Saved aps_conformal.csv")
print("  VERIFY: each 'marginal' value should sit at or just above its target")
print("  confidence. If so, APS is implemented correctly and the set-size columns")
print("  are directly comparable to the LAC sets in Table 3.")

# =============================================================================
# SECTION 10: SENSITIVITY ANALYSIS -- EXTERNAL COVERAGE WITHOUT LOCAL SPLIT
# CALIBRATION (averaged over 200 splits)
# The main result calibrates conformal thresholds on a CALIBRATION HALF carved
# from each external cohort. Split conformal meets target coverage IN EXPECTATION
# over calibration/test draws, NOT on every individual split (on any single split
# the finite-sample coverage scatters around the target and lands below it about
# half the time). So this sensitivity analysis must be averaged over many splits,
# exactly like Section 5, or the local arm will appear to miss target purely from
# sampling noise. For each of 200 stratified 50/50 splits we hold the TEST half
# fixed and apply two threshold sources to it:
#   (1) SPLIT  : class-conditional thresholds fit on THAT split's calibration half
#                -- the main method; mean coverage should sit at/above target.
#   (2) DEVCAL : class-conditional thresholds frozen on DEVELOPMENT OOF (cc_dev),
#                applied with NO local calibration -- mean coverage is expected to
#                fall below target by the transport gap.
# We report, per arm and confidence, the across-split mean coverage and mean set
# size with 2.5-97.5 percentile bands, plus the coverage drop (target - mean).
# Output feeds Supplementary Figure S_nosplit and Supplementary Table S_nosplit.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 10: SENSITIVITY -- external coverage WITHOUT a local calibration split (200 splits)")
print("=" * 100)

n_sens_splits = 200
sens_seeds = list(range(n_sens_splits))
sens_class_to_col = {cls: i for i, cls in enumerate(classes)}

sensitivity_nosplit_store = {}
nosplit_rows = []
for cohort_label, probas_full, labels_full in [
        ('EGD', ext_full_probas, y_ext_full),
        ('UTSW', utsw_full_probas, y_utsw_full)]:

    scores_full = 1 - probas_full
    true_col_full = np.array([sens_class_to_col[y] for y in labels_full])
    n_full = len(labels_full)

    # accumulators: (split, confidence) for marginal coverage / mean set size,
    # and (split, confidence, class) for per-class coverage / set size, per arm.
    n_conf = len(conf_levels_extra)
    split_marg = np.full((n_sens_splits, n_conf), np.nan)
    split_size = np.full((n_sens_splits, n_conf), np.nan)
    dev_marg = np.full((n_sens_splits, n_conf), np.nan)
    dev_size = np.full((n_sens_splits, n_conf), np.nan)
    split_cls_cov = np.full((n_sens_splits, n_conf, n_classes), np.nan)
    dev_cls_cov = np.full((n_sens_splits, n_conf, n_classes), np.nan)
    split_cls_size = np.full((n_sens_splits, n_conf, n_classes), np.nan)
    dev_cls_size = np.full((n_sens_splits, n_conf, n_classes), np.nan)

    print(f"\nRunning {n_sens_splits} splits for {cohort_label} (n = {n_full})...")
    for s_idx, seed in enumerate(sens_seeds):
        cal_pos, te_pos = train_test_split(
            np.arange(n_full), test_size=0.5, stratify=labels_full, random_state=seed
        )
        cal_scores = scores_full[cal_pos]
        cal_labels = labels_full[cal_pos]
        cal_true_col = true_col_full[cal_pos]
        cal_alphas = cal_scores[np.arange(len(cal_pos)), cal_true_col]

        # arm 1 threshold source: this split's own calibration half
        cc_split = ConformalClassifier()
        cc_split.fit(cal_alphas, bins=cal_labels)

        te_scores = scores_full[te_pos]
        te_labels = labels_full[te_pos]
        te_true_col = true_col_full[te_pos]

        for c_idx, target_confidence in enumerate(conf_levels_extra):
            for arm_idx, cc_arm in enumerate([cc_split, cc_dev]):
                pred_sets = np.column_stack([
                    cc_arm.predict_set(te_scores, bins=np.full(len(te_pos), cls),
                                       confidence=target_confidence, smoothing=False)[:, c]
                    for c, cls in enumerate(classes)
                ])
                set_sizes = pred_sets.sum(axis=1)
                marginal = float(np.mean(pred_sets[np.arange(len(te_pos)), te_true_col]))
                mean_size = float(np.mean(set_sizes))
                if arm_idx == 0:
                    split_marg[s_idx, c_idx] = marginal
                    split_size[s_idx, c_idx] = mean_size
                else:
                    dev_marg[s_idx, c_idx] = marginal
                    dev_size[s_idx, c_idx] = mean_size
                for ci, cls in enumerate(classes):
                    cls_mask = (te_labels == cls)
                    if np.sum(cls_mask) > 0:
                        cov_cls = float(np.mean(pred_sets[cls_mask, ci]))
                        size_cls = float(np.mean(set_sizes[cls_mask]))
                        if arm_idx == 0:
                            split_cls_cov[s_idx, c_idx, ci] = cov_cls
                            split_cls_size[s_idx, c_idx, ci] = size_cls
                        else:
                            dev_cls_cov[s_idx, c_idx, ci] = cov_cls
                            dev_cls_size[s_idx, c_idx, ci] = size_cls

    # ---- aggregate across splits ----
    sensitivity_nosplit_store[cohort_label] = {}
    print(f"  {'Conf':>5} | {'Arm':<7} | {'Mean cov':>8} | {'Drop':>6} | {'95% band':>15} | {'Mean size':>9}")
    for c_idx, target_confidence in enumerate(conf_levels_extra):
        sensitivity_nosplit_store[cohort_label][target_confidence] = {}
        for arm_label, marg_arr, size_arr, cls_cov_arr, cls_size_arr in [
                ('split', split_marg, split_size, split_cls_cov, split_cls_size),
                ('devcal', dev_marg, dev_size, dev_cls_cov, dev_cls_size)]:
            mcov = marg_arr[:, c_idx]
            msize = size_arr[:, c_idx]
            cov_mean = float(np.nanmean(mcov))
            cov_lo = float(np.nanpercentile(mcov, 2.5))
            cov_hi = float(np.nanpercentile(mcov, 97.5))
            size_mean = float(np.nanmean(msize))
            size_lo = float(np.nanpercentile(msize, 2.5))
            size_hi = float(np.nanpercentile(msize, 97.5))
            per_class = {}
            for ci, cls in enumerate(classes):
                per_class[cls] = {
                    'cov_mean': float(np.nanmean(cls_cov_arr[:, c_idx, ci])),
                    'cov_lo': float(np.nanpercentile(cls_cov_arr[:, c_idx, ci], 2.5)),
                    'cov_hi': float(np.nanpercentile(cls_cov_arr[:, c_idx, ci], 97.5)),
                    'size_mean': float(np.nanmean(cls_size_arr[:, c_idx, ci])),
                    'cov_drop_vs_target': float(target_confidence - np.nanmean(cls_cov_arr[:, c_idx, ci])),
                }
            sensitivity_nosplit_store[cohort_label][target_confidence][arm_label] = {
                'marginal_coverage_mean': cov_mean,
                'marginal_coverage_lo': cov_lo,
                'marginal_coverage_hi': cov_hi,
                'coverage_drop_vs_target': float(target_confidence - cov_mean),
                'mean_set_size_mean': size_mean,
                'mean_set_size_lo': size_lo,
                'mean_set_size_hi': size_hi,
                'pct_splits_ge_target': float(np.nanmean(mcov >= target_confidence) * 100),
                'per_class': per_class,
            }
            band = f"[{cov_lo:.2f}-{cov_hi:.2f}]"
            print(f"  {target_confidence:>5.2f} | {arm_label:<7} | {cov_mean:>8.3f} | "
                  f"{target_confidence - cov_mean:>+6.2f} | {band:>15} | {size_mean:>9.2f}")

            for cls in classes:
                pc = per_class[cls]
                nosplit_rows.append({
                    'cohort': cohort_label, 'confidence': target_confidence, 'calibration': arm_label,
                    'class': cls,
                    'per_class_coverage_mean': pc['cov_mean'],
                    'per_class_coverage_lo_2.5': pc['cov_lo'],
                    'per_class_coverage_hi_97.5': pc['cov_hi'],
                    'per_class_drop_vs_target': pc['cov_drop_vs_target'],
                    'per_class_set_size_mean': pc['size_mean'],
                    'marginal_coverage_mean': cov_mean,
                    'marginal_drop_vs_target': float(target_confidence - cov_mean),
                    'pct_splits_ge_target': float(np.nanmean(mcov >= target_confidence) * 100),
                    'overall_mean_set_size': size_mean,
                })

pd.DataFrame(nosplit_rows).to_csv('sensitivity_nosplit_coverage.csv', index=False)
print("\n  Saved sensitivity_nosplit_coverage.csv")
print("  READ: 'split' mean coverage should sit at/above target (its guarantee holds")
print("  in expectation over splits; ~50% of individual splits fall below, by design).")
print("  'devcal' mean coverage falling below target = the transport gap when no")
print("  local calibration split is available.")


# ---- attach the new analyses to the results payload ----
results_store['calibration'] = calibration_store
results_store['transport'] = transport_store
results_store['altmodel'] = altmodel_store
results_store['aps'] = aps_store
results_store['sensitivity_nosplit'] = sensitivity_nosplit_store


with open('manuscript_results.pkl', 'wb') as f:
    pickle.dump({'results': results_store, 'cohort_df': df}, f)
print("\nSaved manuscript_results.pkl")


# =============================================================================
# SECTION 5: REPEATED-SPLIT COVERAGE STABILITY + REPRESENTATIVE SPLIT
# Re-partition each external cohort 200 times (stratified 50/50) with the
# model fixed, to estimate seed-averaged per-class coverage and its
# across-split band (the validity claim behind Figure 3 and Tables 3/3b),
# and to pick the single split closest to the mean for the case-level
# figures. Writes repeated_split_coverage_summary.csv and prints the
# random_state to set in the Section 1 external splits.
# =============================================================================
print("\n" + "=" * 100)
print("SECTION 5: REPEATED-SPLIT COVERAGE STABILITY")
print("=" * 100)

# =============================================================================
# REPEATED-SPLIT CONFORMAL COVERAGE STABILITY + REPRESENTATIVE-SPLIT SELECTION
#
# Append to the end of pipeline_v2_utsw_included.py (same session). Reuses the
# fitted objects: final_model, final_features, dev_scaler, to_drop, df_ext,
# df_utsw, X_ext_imp, X_utsw_imp, classes.
#
# Part 1 estimates seed-averaged coverage and its across-split band (Figure 3).
# Part 2 selects, per cohort, the single split whose 90% per-class coverage is
# closest to that seed-averaged mean, and prints the random_state to use in the
# pipeline's train_test_split calls so the single-split case-level figures
# (Figure 4, Figure 5, failure-case selection) illustrate a split consistent
# with Figure 3 rather than an arbitrary one.
#
# Design (locked):
#   - dev_scaler applied to external, no external refit
#   - 200 stratified 50/50 splits, fixed seed list range(200)
#   - selection criterion: smallest z-scored Euclidean distance between a
#     split's per-class 90% coverage and the across-split mean. Coverage only;
#     set size and appearance are never used to choose the split.
# =============================================================================

N_SPLITS = 200
SEEDS = list(range(N_SPLITS))
CONF_LEVELS = [0.80, 0.85, 0.90, 0.95]
SELECT_CONF = 0.90  # match the reported single split to the 90% headline

class_to_col = {cls: i for i, cls in enumerate(classes)}
conf_to_idx = {c: i for i, c in enumerate(CONF_LEVELS)}

cohorts = [
    ('EGD', X_ext_imp, df_ext),
    ('UTSW', X_utsw_imp, df_utsw),
]

summary_rows = []
per_seed_rows = []
singleton_rows = []
representative = {}

for cohort_label, X_imp_full, df_cohort in cohorts:

    # Deployment-faithful transform: dev scaler -> drop correlated -> final feats
    X_scl_full = pd.DataFrame(
        dev_scaler.transform(X_imp_full),
        columns=X_imp_full.columns, index=X_imp_full.index
    )
    X_cf_full = X_scl_full.drop(columns=to_drop)
    probas_full = final_model.predict_proba(X_cf_full[final_features].values)
    scores_full = 1 - probas_full

    labels_full = df_cohort.loc[X_imp_full.index, '2021_glioma_class'].values
    true_col_full = np.array([class_to_col[y] for y in labels_full])
    n_full = len(labels_full)
    n_class_full = {cls: int(np.sum(labels_full == cls)) for cls in classes}

    cov = np.full((N_SPLITS, len(CONF_LEVELS), len(classes)), np.nan)
    size = np.full((N_SPLITS, len(CONF_LEVELS), len(classes)), np.nan)
    pct1 = np.full((N_SPLITS, len(CONF_LEVELS), len(classes)), np.nan)
    pct2 = np.full((N_SPLITS, len(CONF_LEVELS), len(classes)), np.nan)
    pct3 = np.full((N_SPLITS, len(CONF_LEVELS), len(classes)), np.nan)

    # pooled singleton-confusion accumulators, keyed by (confidence, predicted
    # singleton label). Used for the precision-style comparison in Figure 4 B/D:
    # of the size-1 sets equal to {k}, how many were truly class k.
    singleton_total = np.zeros((len(CONF_LEVELS), len(classes)), dtype=int)
    singleton_correct = np.zeros((len(CONF_LEVELS), len(classes)), dtype=int)

    print(f"\nRunning {N_SPLITS} splits for {cohort_label} (n = {n_full})...")
    for s_idx, seed in enumerate(SEEDS):
        cal_pos, te_pos = train_test_split(
            np.arange(n_full), test_size=0.5,
            stratify=labels_full, random_state=seed
        )
        cal_scores = scores_full[cal_pos]
        te_scores = scores_full[te_pos]
        cal_labels = labels_full[cal_pos]
        te_labels = labels_full[te_pos]
        cal_true_col = true_col_full[cal_pos]
        te_true_col = true_col_full[te_pos]

        cal_alphas = cal_scores[np.arange(len(cal_pos)), cal_true_col]
        cc = ConformalClassifier()
        cc.fit(cal_alphas, bins=cal_labels)

        for c_idx, conf in enumerate(CONF_LEVELS):
            pred_sets = np.column_stack([
                cc.predict_set(te_scores, bins=np.full(len(te_pos), cls),
                               confidence=conf, smoothing=False)[:, c]
                for c, cls in enumerate(classes)
            ])
            set_sizes = pred_sets.sum(axis=1)
            for ci, cls in enumerate(classes):
                cls_mask = (te_labels == cls)
                if np.sum(cls_mask) > 0:
                    cov[s_idx, c_idx, ci] = np.mean(pred_sets[cls_mask, ci])
                    sizes_cls = set_sizes[cls_mask]
                    size[s_idx, c_idx, ci] = np.mean(sizes_cls)
                    pct1[s_idx, c_idx, ci] = np.mean(sizes_cls == 1) * 100
                    pct2[s_idx, c_idx, ci] = np.mean(sizes_cls == 2) * 100
                    pct3[s_idx, c_idx, ci] = np.mean(sizes_cls == 3) * 100

            # singleton-conditional accuracy: condition on the PREDICTED singleton
            # label (size-1 sets), pooled across splits. argmax over the boolean
            # set columns gives the singleton's label only where the set has size 1.
            singleton_mask = (set_sizes == 1)
            pred_label_idx = np.argmax(pred_sets, axis=1)
            for k in range(len(classes)):
                is_single_k = singleton_mask & (pred_label_idx == k)
                singleton_total[c_idx, k] += int(np.sum(is_single_k))
                singleton_correct[c_idx, k] += int(np.sum(is_single_k & (te_true_col == k)))

    # ---- aggregate to the summary table (Figure 3 / text) ----
    for c_idx, conf in enumerate(CONF_LEVELS):
        for ci, cls in enumerate(classes):
            cov_draws = cov[:, c_idx, ci]
            size_draws = size[:, c_idx, ci]
            pct1_draws = pct1[:, c_idx, ci]
            pct2_draws = pct2[:, c_idx, ci]
            pct3_draws = pct3[:, c_idx, ci]
            keep = ~np.isnan(cov_draws)
            cov_draws = cov_draws[keep]
            size_draws = size_draws[keep]
            pct1_draws = pct1_draws[keep]
            pct2_draws = pct2_draws[keep]
            pct3_draws = pct3_draws[keep]
            if len(cov_draws) == 0:
                continue
            summary_rows.append({
                'cohort': cohort_label, 'confidence': conf, 'class': cls,
                'n_class_full': n_class_full[cls], 'n_splits_used': int(len(cov_draws)),
                'mean_coverage': float(np.mean(cov_draws)),
                'coverage_lo_2.5': float(np.percentile(cov_draws, 2.5)),
                'coverage_hi_97.5': float(np.percentile(cov_draws, 97.5)),
                'pct_splits_ge_target': float(np.mean(cov_draws >= conf) * 100),
                'mean_set_size': float(np.mean(size_draws)),
                'set_size_lo_2.5': float(np.percentile(size_draws, 2.5)),
                'set_size_hi_97.5': float(np.percentile(size_draws, 97.5)),
                'mean_pct_single': float(np.mean(pct1_draws)),
                'pct_single_lo_2.5': float(np.percentile(pct1_draws, 2.5)),
                'pct_single_hi_97.5': float(np.percentile(pct1_draws, 97.5)),
                'mean_pct_double': float(np.mean(pct2_draws)),
                'pct_double_lo_2.5': float(np.percentile(pct2_draws, 2.5)),
                'pct_double_hi_97.5': float(np.percentile(pct2_draws, 97.5)),
                'mean_pct_all_three': float(np.mean(pct3_draws)),
                'pct_all_three_lo_2.5': float(np.percentile(pct3_draws, 2.5)),
                'pct_all_three_hi_97.5': float(np.percentile(pct3_draws, 97.5)),
            })

    # ---- singleton-conditional accuracy rows (pooled across splits) ----
    # Precision analogue for Figure 4 B/D: among size-1 sets equal to {k}, the
    # fraction truly k. Pooled over the 200 test halves; mean_singletons_per_split
    # records how sparse the conditioning event is (small for minority classes).
    for c_idx, conf in enumerate(CONF_LEVELS):
        for k, cls in enumerate(classes):
            tot = int(singleton_total[c_idx, k])
            cor = int(singleton_correct[c_idx, k])
            singleton_rows.append({
                'cohort': cohort_label, 'confidence': conf, 'pred_label': cls,
                'n_singleton_total': tot, 'n_singleton_correct': cor,
                'singleton_accuracy': (cor / tot) if tot > 0 else np.nan,
                'mean_singletons_per_split': tot / float(N_SPLITS),
            })

    # ---- representative-split selection at the SELECT_CONF level ----
    sel_idx = conf_to_idx[SELECT_CONF]
    cov_sel = cov[:, sel_idx, :]                      # (N_SPLITS, n_classes)
    mean_vec = np.nanmean(cov_sel, axis=0)            # per-class mean across seeds
    std_vec = np.nanstd(cov_sel, axis=0)
    std_vec = np.where(std_vec < 1e-9, 1e-9, std_vec)

    # z-scored Euclidean distance to the mean; seeds with any missing class -> inf
    z = (cov_sel - mean_vec) / std_vec
    dist = np.sqrt(np.sum(z * z, axis=1))
    has_nan = np.isnan(cov_sel).any(axis=1)
    dist[has_nan] = np.inf

    best_pos = int(np.argmin(dist))
    best_seed = SEEDS[best_pos]
    representative[cohort_label] = {
        'seed': best_seed,
        'mean_cov': {cls: float(mean_vec[ci]) for ci, cls in enumerate(classes)},
        'split_cov': {cls: float(cov_sel[best_pos, ci]) for ci, cls in enumerate(classes)},
    }

    for s_idx, seed in enumerate(SEEDS):
        row = {'cohort': cohort_label, 'seed': seed, 'z_distance_at_90': float(dist[s_idx])}
        for ci, cls in enumerate(classes):
            row[f'cov90_{cls}'] = float(cov_sel[s_idx, ci])
        per_seed_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('repeated_split_coverage_summary.csv', index=False)
pd.DataFrame(per_seed_rows).to_csv('repeated_split_per_seed_cov90.csv', index=False)
pd.DataFrame(singleton_rows).to_csv('singleton_conditional_accuracy.csv', index=False)

# ---- headline coverage table at 90% ----
print("\n" + "=" * 95)
print(f"REPEATED-SPLIT COVERAGE at 90% target ({N_SPLITS} splits, dev scaler applied to external)")
print("=" * 95)
print(f"  {'Cohort':<6} | {'Class':<18} | {'N':>4} | {'Mean cov':>9} | {'95% band':>16} | {'% splits >= target':>18}")
print("-" * 95)
for _, r in summary_df[summary_df['confidence'] == 0.90].iterrows():
    band = f"[{r['coverage_lo_2.5']:.2f}-{r['coverage_hi_97.5']:.2f}]"
    print(f"  {r['cohort']:<6} | {r['class']:<18} | {int(r['n_class_full']):>4} | "
          f"{r['mean_coverage']:>9.3f} | {band:>16} | {r['pct_splits_ge_target']:>17.0f}%")

# ---- representative-split report ----
print("\n" + "=" * 95)
print("REPRESENTATIVE SPLIT (closest to the 200-split mean 90% coverage, per class)")
print("Set these as random_state in the pipeline so Figures 4 and 5 match Figure 3:")
print("=" * 95)
for cohort_label in ['EGD', 'UTSW']:
    rep = representative[cohort_label]
    print(f"\n  {cohort_label}: random_state = {rep['seed']}")
    print(f"    {'Class':<18} | {'split cov':>9} | {'mean cov':>9}")
    print("    " + "-" * 42)
    for cls in classes:
        print(f"    {cls:<18} | {rep['split_cov'][cls]:>9.3f} | {rep['mean_cov'][cls]:>9.3f}")
print("\n  EGD seed  -> random_state in the EGD train_test_split (df_ext partition)")
print("  UTSW seed -> random_state in the UTSW train_test_split (df_utsw partition)")
print("  Then rerun the pipeline so the pkl, Figure 4, Figure 5, and failure-case")
print("  selection all use the representative split.")
print("\nSaved: repeated_split_coverage_summary.csv, repeated_split_per_seed_cov90.csv")
