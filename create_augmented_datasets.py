#!/usr/bin/env python3
"""
Create augmented datasets with different synthetic-to-real ratios.
Uses generated synthetic images to create 3 new nnUNet datasets:
- Dataset025_PuFiRS25_Aug0.5: 70 real + 35 synthetic = 105 cases
- Dataset025_PuFiRS25_Aug1.0: 70 real + 70 synthetic = 140 cases
- Dataset025_PuFiRS25_Aug2.0: 70 real + 140 synthetic = 210 cases
"""

import os
import json
import shutil
import nibabel as nib
from pathlib import Path
from collections import defaultdict

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_case_id_from_filename(filename):
    """Extract case ID from nifti filename"""
    return filename.replace('.nii.gz', '')

def create_augmented_dataset(base_dataset_name, aug_ratio, num_synthetic_to_use):
    """Create an augmented dataset with specified ratio of synthetic images"""
    
    # Paths
    base_dataset_folder = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25"
    synthetic_images_folder = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2"
    synthetic_masks_folder = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_masks_v2"
    synthetic_metadata_file = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_metadata_batch1.json"
    
    aug_dataset_folder = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/{base_dataset_name}"
    
    print(f"\n{'='*70}")
    print(f"Creating {base_dataset_name}")
    print(f"  Ratio: Aug{aug_ratio:.1f}")
    print(f"  Total cases: 70 real + {num_synthetic_to_use} synthetic = {70 + num_synthetic_to_use}")
    print(f"{'='*70}")
    
    # Create directories
    imagestr_dir = os.path.join(aug_dataset_folder, 'imagesTr')
    labelstr_dir = os.path.join(aug_dataset_folder, 'labelsTr')
    imagests_dir = os.path.join(aug_dataset_folder, 'imagesTs')
    labelsts_dir = os.path.join(aug_dataset_folder, 'labelsTs')
    
    for d in [imagestr_dir, labelstr_dir, imagests_dir, labelsts_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Copy real training images and masks
    print("\nCopying real training images...")
    real_images_dir = os.path.join(base_dataset_folder, 'imagesTr')
    real_masks_dir = os.path.join(base_dataset_folder, 'labelsTr')
    
    real_cases = sorted([f for f in os.listdir(real_images_dir) if f.endswith('_0000.nii.gz')])
    for i, img_file in enumerate(real_cases, 1):
        case_id = img_file.replace('_0000.nii.gz', '')
        mask_file = img_file.replace('_0000.nii.gz', '.nii.gz')
        
        src_img = os.path.join(real_images_dir, img_file)
        dst_img = os.path.join(imagestr_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        src_mask = os.path.join(real_masks_dir, mask_file)
        dst_mask = os.path.join(labelstr_dir, mask_file)
        shutil.copy2(src_mask, dst_mask)
        
        if i % 10 == 0:
            print(f"  Copied {i}/{len(real_cases)} real cases")
    
    print(f"  ✓ All {len(real_cases)} real training cases copied")
    
    # Copy synthetic images
    if num_synthetic_to_use > 0:
        print(f"\nCopying {num_synthetic_to_use} synthetic images...")
        
        # Load metadata to track sources
        synthetic_metadata = load_json(synthetic_metadata_file) if os.path.exists(synthetic_metadata_file) else {}
        
        synthetic_files = sorted([f for f in os.listdir(synthetic_images_folder) 
                                 if f.endswith('_0000.nii.gz')])[:num_synthetic_to_use]
        
        for i, img_file in enumerate(synthetic_files, 1):
            case_id = img_file.replace('_0000.nii.gz', '')
            mask_file = img_file.replace('_0000.nii.gz', '.nii.gz')
            
            src_img = os.path.join(synthetic_images_folder, img_file)
            dst_img = os.path.join(imagestr_dir, img_file)
            shutil.copy2(src_img, dst_img)
            
            src_mask = os.path.join(synthetic_masks_folder, mask_file)
            dst_mask = os.path.join(labelstr_dir, mask_file)
            shutil.copy2(src_mask, dst_mask)
            
            if i % 20 == 0:
                print(f"  Copied {i}/{len(synthetic_files)} synthetic cases")
        
        print(f"  ✓ All {len(synthetic_files)} synthetic cases copied")
    
    # Copy test set (same for all augmented versions)
    print("\nCopying test set...")
    test_images_dir = os.path.join(base_dataset_folder, 'imagesTs')
    test_masks_dir = os.path.join(base_dataset_folder, 'labelsTs')
    
    test_cases = sorted([f for f in os.listdir(test_images_dir) if f.endswith('_0000.nii.gz')])
    for img_file in test_cases:
        mask_file = img_file.replace('_0000.nii.gz', '.nii.gz')
        
        src_img = os.path.join(test_images_dir, img_file)
        dst_img = os.path.join(imagests_dir, img_file)
        shutil.copy2(src_img, dst_img)
        
        src_mask = os.path.join(test_masks_dir, mask_file)
        dst_mask = os.path.join(labelsts_dir, mask_file)
        shutil.copy2(src_mask, dst_mask)
    
    print(f"  ✓ Test set copied ({len(test_cases)} cases)")
    
    # Create dataset.json
    print("\nGenerating dataset.json...")
    base_dataset_json = load_json(os.path.join(base_dataset_folder, 'dataset.json'))
    
    # Update with new case count
    total_training_cases = len(real_cases) + num_synthetic_to_use
    
    aug_dataset_json = {
        **base_dataset_json,
        "name": base_dataset_name,
        "numTraining": total_training_cases,
        "description": f"PuFiRS25 with {num_synthetic_to_use} synthetic augmentations (ratio {aug_ratio:.1f})"
    }
    
    dataset_json_path = os.path.join(aug_dataset_folder, 'dataset.json')
    save_json(aug_dataset_json, dataset_json_path)
    print(f"  ✓ dataset.json created")
    
    # Create splits_final.json with updated training cases
    print("\nGenerating splits_final.json...")
    base_splits = load_json(os.path.join(base_dataset_folder, 'splits_final.json'))
    
    # The original splits have 70 real training cases + 25 validation cases
    # For augmented datasets, we keep the same validation split but add synthetic to training
    original_train = set(base_splits[0]['train'])
    original_val = set(base_splits[0]['val'])
    
    # Get list of all training cases (real + synthetic)
    all_real_case_ids = [f.replace('_0000.nii.gz', '') for f in real_cases]
    all_synthetic_case_ids = [f.replace('_0000.nii.gz', '') 
                             for f in sorted([f for f in os.listdir(synthetic_images_folder) 
                                            if f.endswith('_0000.nii.gz')])[:num_synthetic_to_use]]
    
    # Create new splits: all training cases go to train fold, validation stays in val
    aug_splits = [{
        "train": all_real_case_ids + all_synthetic_case_ids,
        "val": list(original_val)
    }]
    
    splits_path = os.path.join(aug_dataset_folder, 'splits_final.json')
    save_json(aug_splits, splits_path)
    print(f"  ✓ splits_final.json created ({len(aug_splits[0]['train'])} train, {len(aug_splits[0]['val'])} val)")
    
    # Create mapping CSV files (for reference)
    print("\nGenerating mapping files...")
    
    # Training mapping
    mapping_tr = []
    mapping_tr.extend([(f.replace('_0000.nii.gz', ''), 'real') for f in real_cases])
    mapping_tr.extend([(f.replace('_0000.nii.gz', ''), 'synthetic') 
                       for f in sorted([f for f in os.listdir(synthetic_images_folder) 
                                      if f.endswith('_0000.nii.gz')])[:num_synthetic_to_use]])
    
    with open(os.path.join(aug_dataset_folder, 'training_cases.csv'), 'w') as f:
        f.write("case_id,source\n")
        for case_id, source in mapping_tr:
            f.write(f"{case_id},{source}\n")
    
    print(f"  ✓ Mapping files created")
    
    print(f"\n✓ {base_dataset_name} created successfully!")
    print(f"  Location: {aug_dataset_folder}")
    print(f"  Training cases: {total_training_cases} (70 real + {num_synthetic_to_use} synthetic)")
    print(f"  Test cases: {len(test_cases)}")
    
    return aug_dataset_folder

def main():
    # Check if synthetic images exist
    synthetic_images_folder = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2"
    
    if not os.path.exists(synthetic_images_folder):
        print(f"ERROR: Synthetic images folder not found: {synthetic_images_folder}")
        print("Please ensure the DDPM generation job has completed first.")
        return
    
    synthetic_files = [f for f in os.listdir(synthetic_images_folder) if f.endswith('_0000.nii.gz')]
    print(f"\nFound {len(synthetic_files)} synthetic images")
    
    if len(synthetic_files) < 140:
        print(f"WARNING: Expected 140 synthetic images, found {len(synthetic_files)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create three augmented datasets
    configs = [
        ("Dataset025_PuFiRS25_Aug0.5", 0.5, 35),
        ("Dataset025_PuFiRS25_Aug1.0", 1.0, 70),
        ("Dataset025_PuFiRS25_Aug2.0", 2.0, 140),
    ]
    
    created_datasets = []
    for dataset_name, ratio, num_synthetic in configs:
        try:
            folder = create_augmented_dataset(dataset_name, ratio, num_synthetic)
            created_datasets.append((dataset_name, folder))
        except Exception as e:
            print(f"ERROR creating {dataset_name}: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Augmented Datasets Created")
    print(f"{'='*70}\n")
    
    for dataset_name, folder in created_datasets:
        img_count = len([f for f in os.listdir(os.path.join(folder, 'imagesTr')) 
                        if f.endswith('.nii.gz')])
        print(f"✓ {dataset_name}: {img_count} training cases")
    
    print(f"\nNext steps:")
    print(f"1. Run preprocessing: nnUNetv2_plan_and_preprocess -d <dataset_id> -c 3d_fullres")
    print(f"2. Train models: nnUNetv2_train <dataset_id> 3d_fullres 0 -tr nnUNetTrainer")
    print(f"3. Evaluate: Use evaluate_baseline_direct.py for each model")

if __name__ == "__main__":
    main()
