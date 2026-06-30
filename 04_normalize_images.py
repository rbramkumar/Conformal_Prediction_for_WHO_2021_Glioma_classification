# Code for z-score normalizations for all 4 cohorts 

import os
import sys
import numpy as np
import SimpleITK as sitk

PREPROCESSED_DIR = '' # removed for manuscript it had the folder name

sequences_to_normalize = ['flair', 't1ce', 't1', 't2']


all_entries = sorted(os.listdir(PREPROCESSED_DIR))
case_folders = []
for entry in all_entries:
    entry_path = os.path.join(PREPROCESSED_DIR, entry)
    if not os.path.isdir(entry_path):
        continue
    if entry.startswith('_'):
        continue
    case_folders.append(entry)

for case_index, case_folder in enumerate(case_folders):


    case_path = os.path.join(PREPROCESSED_DIR, case_folder)
    print(f"Normalizing: {case_folder}")

    for seq in sequences_to_normalize:

        out_file_path = os.path.join(case_path, f"{seq}_norm.nii.gz")

        if os.path.exists(out_file_path):
            continue

        n4_file_path = None
        for file_name in os.listdir(case_path):
            if file_name.startswith(f"{seq}_n4") and file_name.endswith('.nii.gz'):
                n4_file_path = os.path.join(case_path, file_name)
                break

        if n4_file_path is None:
            continue

        sitk_img = sitk.ReadImage(n4_file_path)
        img_array = sitk.GetArrayFromImage(sitk_img).astype(np.float32)

        brain_mask = img_array > 0
        if int(brain_mask.sum()) == 0:
            continue

        brain_values = img_array[brain_mask]
        brain_mean = float(brain_values.mean())
        brain_std = float(brain_values.std())

        if brain_std == 0:
            continue

        normalized_array = np.zeros_like(img_array, dtype=np.float32)
        normalized_array[brain_mask] = (img_array[brain_mask] - brain_mean) / brain_std

        normalized_sitk = sitk.GetImageFromArray(normalized_array)
        normalized_sitk.CopyInformation(sitk_img)
        sitk.WriteImage(normalized_sitk, out_file_path)
