import json
import os

WORKSPACE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
BASELINE_SPLITS_PATH = os.path.join(WORKSPACE, "nnUNet_preprocessed/Dataset025_PuFiRS25/splits_final.json")

def main():
    print("🛠️ Rebuilding Custom Appended Cross-Validation Splits Layouts...")
    print("──────────────────────────────────────────────────────────────────────────")
    
    # 1. Load the master baseline splits file
    with open(BASELINE_SPLITS_PATH, 'r') as f:
        baseline_splits = json.load(f)

    # 2. Build the exact file token lists based on your raw folder contents
    # Dataset 026 (Aug0.5): First 35 consecutive files
    syn_pool_026 = [f"synthetic_{i:04d}" for i in range(35)]
    
    # Dataset 027 (Aug1.0): First 70 consecutive files
    syn_pool_027 = [f"synthetic_{i:04d}" for i in range(70)]
    
    # Dataset 028 (Aug2.0): First 136 consecutive files + your 4 custom appended files
    syn_pool_028 = [f"synthetic_{i:04d}" for i in range(136)] + [
        "synthetic_0272", "synthetic_0273", "synthetic_0274", "synthetic_0275"
    ]

    dataset_mappings = {
        "Dataset026_PuFiRS25_Aug0.5": syn_pool_026,
        "Dataset027_PuFiRS25_Aug1.0": syn_pool_027,
        "Dataset028_PuFiRS25_Aug2.0": syn_pool_028
    }

    # 3. Apply the updated lists to each dataset directory
    for ds_folder, allowed_synthetic_pool in dataset_mappings.items():
        ds_preprocessed_path = os.path.join(WORKSPACE, "nnUNet_preprocessed", ds_folder)
        if not os.path.exists(ds_preprocessed_path):
            continue
            
        new_splits = []
        
        for fold_data in baseline_splits:
            base_train = list(fold_data['train'])
            base_val = list(fold_data['val'])
            
            # Combine your 70 baseline real training cases with the exact verified token pool
            augmented_train_pool = list(base_train) + allowed_synthetic_pool
            
            new_splits.append({
                "train": sorted(list(set(augmented_train_pool))),
                "val": sorted(list(set(base_val)))  # Validation set remains pristine and un-leaked
            })
            
        target_json_path = os.path.join(ds_preprocessed_path, "splits_final.json")
        with open(target_json_path, 'w') as f:
            json.dump(new_splits, f, indent=4)
            
        print(f"✅ {ds_folder} -> Mapped {len(allowed_synthetic_pool)} verified files inside split array.")

if __name__ == "__main__":
    main()
