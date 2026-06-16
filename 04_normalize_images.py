import os
import sys
import numpy as np
import SimpleITK as sitk

# =============================================================================
# WITHIN-BRAIN Z-SCORE NORMALIZATION  (shardable for parallel Colab notebooks)
#
# Reads each existing N4 file, z-scores the brain voxels (voxels > 0), holds the
# background at 0, and writes <seq>_norm.nii.gz next to the N4 file. N4 is NOT
# re-run. Each sequence of each case is normalized by its own brain statistics.
#
# HOW TO RUN
#   Smoke test (one case, prints stats, writes that one _norm):
#       !python 04_normalize_images.py
#   Full run, shard S of N across N notebooks (change only the last number):
#       Notebook 0:  !python 04_normalize_images.py 5 0
#       Notebook 1:  !python 04_normalize_images.py 5 1
#       ... up to ...
#       Notebook 4:  !python 04_normalize_images.py 5 4
#   Single-machine full run (no sharding):
#       !python 04_normalize_images.py 1 0
#
# Resume: a (case, sequence) is skipped if its _norm file already exists, so a
# notebook that times out can simply be rerun.
# =============================================================================

# --- CONFIG: edit this path ---
PREPROCESSED_DIR = '/content/drive/MyDrive/bsd/11_radiomics_for_gliomas/02_train/trial_80/preprocessed'

sequences_to_normalize = ['flair', 't1ce', 't1', 't2']
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
    print(f"SMOKE TEST MODE: normalizing only {SMOKE_CASE}\n")
else:
    print(f"Shard {SHARD_INDEX} of {NUM_SHARDS}. Total case folders: {len(case_folders)}\n")

# --- Main loop: case, then sequence ---
for case_index, case_folder in enumerate(case_folders):

    # Sharding: this notebook only handles cases whose index matches its shard
    if case_index % NUM_SHARDS != SHARD_INDEX:
        continue

    case_path = os.path.join(PREPROCESSED_DIR, case_folder)
    print(f"Normalizing: {case_folder}")

    for seq in sequences_to_normalize:

        out_file_path = os.path.join(case_path, f"{seq}_norm.nii.gz")

        # Resume: skip if already normalized
        if os.path.exists(out_file_path):
            print(f"  -> {seq}: already normalized, skipping.")
            continue

        # Find the N4 file. Handles '<seq>_n4.nii.gz' and EGD's '<seq>_n4_stripped.nii.gz'.
        n4_file_path = None
        for file_name in os.listdir(case_path):
            if file_name.startswith(f"{seq}_n4") and file_name.endswith('.nii.gz'):
                n4_file_path = os.path.join(case_path, file_name)
                break

        if n4_file_path is None:
            print(f"  -> WARNING: no N4 file for {seq}, skipping.")
            continue

        sitk_img = sitk.ReadImage(n4_file_path)
        img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

        brain_mask = img_array > 0
        if int(brain_mask.sum()) == 0:
            print(f"  -> WARNING: {seq} has no positive voxels, skipping.")
            continue

        brain_values = img_array[brain_mask]
        brain_mean = float(brain_values.mean())
        brain_std = float(brain_values.std())

        if brain_std == 0:
            print(f"  -> WARNING: {seq} brain std is 0, skipping.")
            continue

        normalized_array = np.zeros_like(img_array, dtype=np.float32)
        normalized_array[brain_mask] = (img_array[brain_mask] - brain_mean) / brain_std

        normalized_sitk = sitk.GetImageFromArray(normalized_array)
        normalized_sitk.CopyInformation(sitk_img)
        sitk.WriteImage(normalized_sitk, out_file_path)

        print(f"  -> {seq}: brain mean {brain_mean:.2f}, std {brain_std:.2f} -> saved {seq}_norm.nii.gz")

    print("")

print("Smoke test done." if SMOKE_TEST else f"Shard {SHARD_INDEX} of {NUM_SHARDS} complete.")
