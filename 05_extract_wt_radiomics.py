import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

# =============================================================================
# WHOLE-TUMOR RADIOMICS EXTRACTION  (Option A, shardable for parallel notebooks)
#
# Inputs are the within-brain z-scored images (<seq>_norm.nii.gz). Discretization
# is fixed bin COUNT (binCount = 64) with normalize = False, which keeps Ng fixed
# per ROI and makes features comparable across scanners/cohorts.
#
# OUTPUT MODEL: one CSV per case in FEATURES_DIR/per_case/. This is what makes
# five notebooks safe to run at once (no shared file to corrupt) and gives free
# resume (a case is done if its file exists). Run 06_combine_features.py at the
# end to concatenate them into one table.
#
# HOW TO RUN
#   Smoke test (one case, prints feature count, writes nothing):
#       !python 05_extract_wt_radiomics.py
#   Full run, shard S of N across N notebooks (change only the last number):
#       Notebook 0:  !python 05_extract_wt_radiomics.py 5 0
#       Notebook 1:  !python 05_extract_wt_radiomics.py 5 1
#       ... up to ...
#       Notebook 4:  !python 05_extract_wt_radiomics.py 5 4
#   Single-machine full run (no sharding):
#       !python 05_extract_wt_radiomics.py 1 0
#
# IMPORTANT: finish all normalization shards (04) before starting extraction,
# and finish all extraction shards before running the combine step (06).
# =============================================================================

# --- CONFIG: edit these paths ---
PREPROCESSED_DIR = '/content/drive/MyDrive/bsd/11_radiomics_for_gliomas/02_train/trial_80/preprocessed'
FEATURES_DIR = '/content/drive/MyDrive/bsd/11_radiomics_for_gliomas/02_train/trial_80/features'
PER_CASE_DIR = os.path.join(FEATURES_DIR, 'per_case')

sequences_to_extract = ['flair', 't1ce', 't1', 't2']
SMOKE_CASE = 'BT0791'

# --- Parse shard arguments. No args = smoke test on one case. ---
if len(sys.argv) >= 3:
    NUM_SHARDS = int(sys.argv[1])
    SHARD_INDEX = int(sys.argv[2])
    SMOKE_TEST = False
else:
    NUM_SHARDS = 1
    SHARD_INDEX = 0
    SMOKE_TEST = True

if not os.path.exists(PER_CASE_DIR):
    os.makedirs(PER_CASE_DIR)

# --- Extractor configuration ---
settings = {
    'binCount': 64,
    'normalize': False,
    'resampledPixelSpacing': None,
    'interpolator': 'sitkBSpline',
    'label': 1
}

extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

extractor.disableAllFeatures()
extractor.enableFeatureClassByName('firstorder')
extractor.enableFeatureClassByName('glcm')
extractor.enableFeatureClassByName('glszm')
extractor.enableFeatureClassByName('glrlm')
extractor.enableFeatureClassByName('gldm')

extractor.enableImageTypeByName('Original')
extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0]})
extractor.enableImageTypeByName('Wavelet')

print("Extractor settings:")
print(extractor.settings)
print("")

# --- Collect case folders (sorted, so sharding is identical in every notebook) ---
all_entries = sorted(os.listdir(PREPROCESSED_DIR))
case_folders = []
for entry in all_entries:
    entry_path = os.path.join(PREPROCESSED_DIR, entry)
    if not os.path.isdir(entry_path):
        continue
    if entry.startswith('_'):
        continue
    case_folders.append(entry)

if SMOKE_TEST:
    case_folders = [SMOKE_CASE]
    print(f"SMOKE TEST MODE: extracting only {SMOKE_CASE}, nothing will be written.\n")
else:
    print(f"Shard {SHARD_INDEX} of {NUM_SHARDS}. Total case folders: {len(case_folders)}\n")

# --- Main loop: case, then sequence ---
for case_index, case_folder in enumerate(case_folders):

    # Sharding: this notebook only handles cases whose index matches its shard
    if (not SMOKE_TEST) and (case_index % NUM_SHARDS != SHARD_INDEX):
        continue

    per_case_path = os.path.join(PER_CASE_DIR, f"{case_folder}.csv")

    # Resume: skip if this case already has an output file
    if (not SMOKE_TEST) and os.path.isfile(per_case_path):
        print(f"Skipping (already done): {case_folder}")
        continue

    case_path = os.path.join(PREPROCESSED_DIR, case_folder)
    print(f"Extracting features for: {case_folder}")

    # Assign cohort from the folder name
    if 'EGD' in case_folder:
        cohort = 'EGD'
    elif 'TCGA' in case_folder:
        cohort = 'TCGA'
    elif 'UCSF' in case_folder:
        cohort = 'UCSF'
    elif case_folder.startswith('BT'):
        cohort = 'UTSW'
    else:
        cohort = 'Unknown'

    # Read the mask, guarded
    mask_path = os.path.join(case_path, 'seg.nii.gz')
    if not os.path.exists(mask_path):
        print(f"  -> Skipping. No seg.nii.gz found.")
        continue

    try:
        sitk_mask = sitk.ReadImage(mask_path)
    except Exception as e:
        print(f"  -> Skipping. Could not read mask: {e}")
        continue

    # Force whole-tumor binary mask (labels 1/2/4 -> 1)
    sitk_mask_binary = sitk.Greater(sitk_mask, 0)
    mask_array = sitk.GetArrayFromImage(sitk_mask_binary)

    # Guard: skip empty masks
    if int(mask_array.sum()) == 0:
        print(f"  -> Skipping. Mask is empty (0 tumor voxels).")
        continue

    # Start the patient row
    patient_data = {
        'Case_ID': case_folder,
        'Cohort': cohort
    }

    # Extract each sequence from the normalized image
    for seq in sequences_to_extract:

        seq_file_path = None
        for file_name in os.listdir(case_path):
            if file_name.startswith(f"{seq}_norm") and file_name.endswith('.nii.gz'):
                seq_file_path = os.path.join(case_path, file_name)
                break

        if seq_file_path is None:
            print(f"  -> WARNING: missing {seq}_norm file.")
            continue

        try:
            sitk_img = sitk.ReadImage(seq_file_path)
        except Exception as e:
            print(f"  -> ERROR reading {seq}: {e}")
            continue

        # Guard: skip this sequence if any ROI voxel is non-finite
        img_array = sitk.GetArrayFromImage(sitk_img)
        roi_values = img_array[mask_array > 0]
        if not np.isfinite(roi_values).all():
            print(f"  -> WARNING: {seq} has non-finite voxels in ROI, skipping {seq}.")
            continue

        try:
            results = extractor.execute(sitk_img, sitk_mask_binary)
            for key, value in results.items():
                if not key.startswith('diagnostics_'):
                    patient_data[f"{seq.upper()}_{key}"] = value
        except Exception as e:
            print(f"  -> ERROR on {seq}: {e}")
            continue

    # Smoke test: report and stop before writing
    if SMOKE_TEST:
        feature_keys = [k for k in patient_data if k not in ('Case_ID', 'Cohort')]
        print(f"  -> SMOKE TEST: {len(feature_keys)} features extracted (no hang).")
        print(f"  -> sample keys: {feature_keys[:5]}")
        continue

    # Write this single case to its own CSV. No shared file, so no corruption
    # across the five notebooks, and the file's existence is the resume flag.
    df_patient = pd.DataFrame([patient_data])
    df_patient.to_csv(per_case_path, index=False)
    print(f"  -> wrote {case_folder}.csv")

print("\nSmoke test done." if SMOKE_TEST else f"\nShard {SHARD_INDEX} of {NUM_SHARDS} complete.")
