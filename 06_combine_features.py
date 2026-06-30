import os
import pandas as pd

FEATURES_DIR = '' # removed for manuscript.. it had the folder name 
PER_CASE_DIR = os.path.join(FEATURES_DIR, 'per_case')
OUTPUT_CSV = os.path.join(FEATURES_DIR, 'all_radiomics_features.csv')

per_case_files = sorted(f for f in os.listdir(PER_CASE_DIR) if f.endswith('.csv'))

all_rows = []
for file_name in per_case_files:
    file_path = os.path.join(PER_CASE_DIR, file_name)
    df_one = pd.read_csv(file_path)
    all_rows.append(df_one)

combined_df = pd.concat(all_rows, ignore_index=True, sort=False)

lead_cols = [c for c in ['Case_ID', 'Cohort'] if c in combined_df.columns]
other_cols = [c for c in combined_df.columns if c not in lead_cols]
combined_df = combined_df[lead_cols + other_cols]

combined_df.to_csv(OUTPUT_CSV, index=False)