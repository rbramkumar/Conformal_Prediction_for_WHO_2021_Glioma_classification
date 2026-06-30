import os
import sys
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


PREPROCESSED_DIR = '' # removed for manuscript 
FEATURES_DIR = '' # removed for manuscript
PER_CASE_DIR = os.path.join(FEATURES_DIR, 'per_case')

sequences_to_extract = ['flair', 't1ce', 't1', 't2']

if not os.path.exists(PER_CASE_DIR):
    os.makedirs(PER_CASE_DIR)


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


    per_case_path = os.path.join(PER_CASE_DIR, f"{case_folder}.csv")

    if os.path.isfile(per_case_path):
        
        continue

    case_path = os.path.join(PREPROCESSED_DIR, case_folder)
    

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

    mask_path = os.path.join(case_path, 'seg.nii.gz')
    if not os.path.exists(mask_path):
        continue

    try:
        sitk_mask = sitk.ReadImage(mask_path)
    except Exception as e:
        continue

    sitk_mask_binary = sitk.Greater(sitk_mask, 0)
    mask_array = sitk.GetArrayFromImage(sitk_mask_binary)

    if int(mask_array.sum()) == 0:
        continue

    patient_data = {
        'Case_ID': case_folder,
        'Cohort': cohort
    }

    for seq in sequences_to_extract:

        seq_file_path = None
        for file_name in os.listdir(case_path):
            if file_name.startswith(f"{seq}_norm") and file_name.endswith('.nii.gz'):
                seq_file_path = os.path.join(case_path, file_name)
                break

        if seq_file_path is None:
            continue

        try:
            sitk_img = sitk.ReadImage(seq_file_path)
        except Exception as e:
            continue

        # Guard: skip this sequence if any ROI voxel is non-finite
        img_array = sitk.GetArrayFromImage(sitk_img)
        roi_values = img_array[mask_array > 0]
        if not np.isfinite(roi_values).all():
            continue

        try:
            results = extractor.execute(sitk_img, sitk_mask_binary)
            for key, value in results.items():
                if not key.startswith('diagnostics_'):
                    patient_data[f"{seq.upper()}_{key}"] = value
        except Exception as e:
            continue

    
    df_patient = pd.DataFrame([patient_data])
    df_patient.to_csv(per_case_path, index=False)
