import os
import pandas as pd

# =============================================================================
# COMBINE PER-CASE FEATURE FILES INTO ONE TABLE
#
# Run this once after ALL extraction shards (05) have finished. It reads every
# CSV in FEATURES_DIR/per_case/, concatenates them (pandas aligns columns and
# fills any missing feature with NaN), and writes one combined table.
#
#   !python 06_combine_features.py
# =============================================================================

# --- CONFIG: edit these paths ---
FEATURES_DIR = '/content/drive/MyDrive/bsd/11_radiomics_for_gliomas/02_train/trial_80/features'
PER_CASE_DIR = os.path.join(FEATURES_DIR, 'per_case')
OUTPUT_CSV = os.path.join(FEATURES_DIR, 'all_radiomics_features.csv')

# --- Collect per-case files ---
per_case_files = sorted(f for f in os.listdir(PER_CASE_DIR) if f.endswith('.csv'))
print(f"Found {len(per_case_files)} per-case files.")

# --- Read each into a list of one-row frames ---
all_rows = []
for file_name in per_case_files:
    file_path = os.path.join(PER_CASE_DIR, file_name)
    df_one = pd.read_csv(file_path)
    all_rows.append(df_one)

# --- Concatenate (columns aligned automatically, missing -> NaN) ---
combined_df = pd.concat(all_rows, ignore_index=True, sort=False)

# Keep Case_ID and Cohort first for readability
lead_cols = [c for c in ['Case_ID', 'Cohort'] if c in combined_df.columns]
other_cols = [c for c in combined_df.columns if c not in lead_cols]
combined_df = combined_df[lead_cols + other_cols]

combined_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nCombined table: {len(combined_df)} rows, {len(combined_df.columns)} columns.")
print(f"Saved to: {OUTPUT_CSV}\n")

print("Cases per cohort:")
if 'Cohort' in combined_df.columns:
    for cohort_name, n in combined_df['Cohort'].value_counts().items():
        print(f"  {cohort_name}: {n}")

# Report any rows missing a whole sequence block (NaN across that prefix)
print("\nRows with at least one fully-missing sequence (NaN-filled):")
for prefix in ['FLAIR', 'T1CE', 'T1', 'T2']:
    prefix_cols = [c for c in combined_df.columns if c.startswith(f"{prefix}_")]
    if len(prefix_cols) == 0:
        print(f"  {prefix}: no columns present at all")
        continue
    all_nan_mask = combined_df[prefix_cols].isna().all(axis=1)
    print(f"  {prefix}: {int(all_nan_mask.sum())} cases")
