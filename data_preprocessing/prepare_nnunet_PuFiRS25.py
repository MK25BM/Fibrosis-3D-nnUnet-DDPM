#!/usr/bin/env python3
"""
Complete PuFiRS25 dataset preparation for nnUNet V2:
1. Copy from PuFiRS25 to nnUNet_raw (using mapping.csv to match images to labels)
2. Reorganize into 70 train / 25 val / 25 test split
3. Create splits_final.json for nnUNet fold structure
"""
import os
import shutil
import json
import random
import pandas as pd
from pathlib import Path

def copy_from_pufirs(src_base, dst_base):
    """
    Copy images and labels from PuFiRS25 to nnUNet_raw using mapping.csv
    Images: already named correctly (new names)
    Labels: need to be renamed (old names → new names)
    """
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    
    print("\n" + "="*70)
    print("STEP 1: Copy data from PuFiRS25 using mapping.csv")
    print("="*70)
    
    # Copy dataset.json
    src_json = src_base / "dataset.json"
    dst_json = dst_base / "dataset.json"
    if src_json.exists():
        os.makedirs(dst_base, exist_ok=True)
        shutil.copy2(src_json, dst_json)
        print(f"✓ Copied dataset.json")
    
    # Create directories
    for d in ["imagesTr", "labelsTr", "imagesVal", "labelsVal", "imagesTs", "labelsTs"]:
        (dst_base / d).mkdir(parents=True, exist_ok=True)
    
    # Process training set
    print("\nTraining set (imagesTr/labelsTr):")
    mapping_df = pd.read_csv(src_base / "imagesTr" / "mapping.csv")
    
    for idx, row in mapping_df.iterrows():
        new_name = row['new']
        old_name = row['old']
        
        # Image: copy with new name (already correct)
        src_img = src_base / "imagesTr" / f"{new_name}_0000.nii.gz"
        dst_img = dst_base / "imagesTr" / f"{new_name}_0000.nii.gz"
        
        # Label: copy with old name but rename to new name
        src_lbl = src_base / "labelsTr" / f"{old_name}.nii.gz"
        dst_lbl = dst_base / "labelsTr" / f"{new_name}.nii.gz"
        
        if src_img.exists() and src_lbl.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
    
    print(f"  ✓ Copied {len(mapping_df)} training cases")
    
    # Process validation set
    print("\nValidation set (imagesVal/labelsVal):")
    mapping_df = pd.read_csv(src_base / "imagesVal" / "mapping.csv")
    
    for idx, row in mapping_df.iterrows():
        new_name = row['new']
        old_name = row['old']
        
        src_img = src_base / "imagesVal" / f"{new_name}_0000.nii.gz"
        dst_img = dst_base / "imagesVal" / f"{new_name}_0000.nii.gz"
        
        src_lbl = src_base / "labelsVal" / f"{old_name}.nii.gz"
        dst_lbl = dst_base / "labelsVal" / f"{new_name}.nii.gz"
        
        if src_img.exists() and src_lbl.exists():
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
    
    print(f"  ✓ Copied {len(mapping_df)} validation cases")
    print("\n✓ Data copy complete!")


def reorganize_splits(src_pufirs, dst_base, seed=42):
    """
    Reorganize 120 cases into 70 train + 25 val + 25 test.
    Copy directly from PuFiRS25 source.
    
    nnUNet V2 structure:
    - imagesTr/ + labelsTr/ = 70 + 25 = 95 cases
    - imagesTs/ + labelsTs/ = 25 cases (holdout test)
    - splits_final.json = defines internal fold: 70 train, 25 val
    """
    
    src_pufirs = Path(src_pufirs)
    dst_base = Path(dst_base)
    
    print("\n" + "="*70)
    print("STEP 2: Reorganize into 70/25/25 split")
    print("="*70)
    
    # Source directories (from PuFiRS25)
    src_img_tr = src_pufirs / "imagesTr"
    src_lbl_tr = src_pufirs / "labelsTr"
    src_img_val = src_pufirs / "imagesVal"
    src_lbl_val = src_pufirs / "labelsVal"
    
    # Destination directories (nnUNet_raw)
    dst_img_tr = dst_base / "imagesTr"
    dst_lbl_tr = dst_base / "labelsTr"
    dst_img_test = dst_base / "imagesTs"
    dst_lbl_test = dst_base / "labelsTs"
    
    # Get all case IDs from PuFiRS25
    train_cases = sorted([f.replace("_0000.nii.gz", "") for f in os.listdir(src_img_tr) if f.endswith("_0000.nii.gz")])
    val_cases = sorted([f.replace("_0000.nii.gz", "") for f in os.listdir(src_img_val) if f.endswith("_0000.nii.gz")])
    
    all_cases = train_cases + val_cases
    
    print(f"\nSource dataset (PuFiRS25):")
    print(f"  imagesTr/: {len(train_cases)} cases")
    print(f"  imagesVal/: {len(val_cases)} cases")
    print(f"  Total: {len(all_cases)} cases")
    
    # Shuffle and split into 70 + 25 + 25
    random.Random(seed).shuffle(all_cases)
    
    new_train_cases = all_cases[:70]
    new_val_cases = all_cases[70:95]
    new_test_cases = all_cases[95:120]
    
    print(f"\nNew split (70/25/25):")
    print(f"  Train: {len(new_train_cases)} cases")
    print(f"  Val:   {len(new_val_cases)} cases")
    print(f"  Test:  {len(new_test_cases)} cases")
    
    # Clear destination directories
    for d in [dst_img_tr, dst_lbl_tr, dst_img_test, dst_lbl_test]:
        for f in d.glob("*.nii.gz"):
            f.unlink()
    
    # Copy case from PuFiRS25 to nnUNet_raw
    def copy_case_from_pufirs(case_id, src_split, dst_img_dir, dst_lbl_dir):
        """Copy a case from PuFiRS25 source"""
        img_file = f"{case_id}_0000.nii.gz"
        lbl_file = f"{case_id}.nii.gz"
        
        if src_split == "train":
            src_img = src_img_tr / img_file
            src_lbl = src_lbl_tr / lbl_file
        else:  # val
            src_img = src_img_val / img_file
            src_lbl = src_lbl_val / lbl_file
        
        if src_img.exists() and src_lbl.exists():
            shutil.copy2(src_img, dst_img_dir / img_file)
            shutil.copy2(src_lbl, dst_lbl_dir / lbl_file)
            return True
        return False
    
    # Determine which split each case came from
    train_split = {case: "train" for case in train_cases}
    val_split = {case: "val" for case in val_cases}
    case_source = {**train_split, **val_split}
    
    # Copy training cases to nnUNet
    for case in new_train_cases:
        copy_case_from_pufirs(case, case_source[case], dst_img_tr, dst_lbl_tr)
    print(f"\n✓ Copied {len(new_train_cases)} training cases to imagesTr/labelsTr")
    
    # Copy validation cases to nnUNet (stay in imagesTr/labelsTr for nnUNet)
    for case in new_val_cases:
        copy_case_from_pufirs(case, case_source[case], dst_img_tr, dst_lbl_tr)
    print(f"✓ Copied {len(new_val_cases)} validation cases to imagesTr/labelsTr")
    
    # Copy test cases
    for case in new_test_cases:
        copy_case_from_pufirs(case, case_source[case], dst_img_test, dst_lbl_test)
    print(f"✓ Copied {len(new_test_cases)} test cases to imagesTs/labelsTs")
    
    # Create splits_final.json
    splits_final = [
        {
            "train": [case for case in new_train_cases],
            "val": [case for case in new_val_cases]
        }
    ]
    
    splits_file = dst_base / "splits_final.json"
    with open(splits_file, 'w') as f:
        json.dump(splits_final, f, indent=2)
    
    print(f"\n✓ Created splits_final.json")
    
    # Update dataset.json with correct numTraining
    dataset_file = dst_base / "dataset.json"
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        dataset['numTraining'] = len(new_train_cases) + len(new_val_cases)
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Updated dataset.json: numTraining={dataset['numTraining']}")
    
    print("\n" + "="*70)
    print("✅ Dataset preparation complete!")
    print("="*70)
    print(f"\nFinal structure:")
    print(f"  imagesTr/ + labelsTr/: {len(new_train_cases) + len(new_val_cases)} cases")
    print(f"    ├─ Training:   {len(new_train_cases)} cases")
    print(f"    └─ Validation: {len(new_val_cases)} cases")
    print(f"  imagesTs/ + labelsTs/: {len(new_test_cases)} cases (holdout test)")
    print(f"  splits_final.json: Defined for nnUNet fold 0")
    print(f"\nNext step: Run nnUNet preprocessing")
    print(f"  nnUNetv2_plan_and_preprocess -d 25 -c 3d --clean")


if __name__ == "__main__":
    src_pufirs = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/PuFiRS25")
    dst_nnunet = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25")
    
    # Step 1: Copy from PuFiRS25
    copy_from_pufirs(src_pufirs, dst_nnunet)
    
    # Step 2: Reorganize splits (copies directly from PuFiRS25)
    reorganize_splits(src_pufirs, dst_nnunet, seed=42)
