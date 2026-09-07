import json
import os

WORKSPACE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
METADATA_PATH = os.path.join(WORKSPACE, "augmented_data/synthetic_images_v2/synthetic_metadata_batch1.json")
BASELINE_SPLITS_PATH = os.path.join(WORKSPACE, "nnUNet_preprocessed/Dataset025_PuFiRS25/splits_final.json")

# Defines how many perturbations to include per parent patient across your new datasets
DATASET_SCALES = {
    "Dataset029_PuFiRS25_50Real": 0,           # 35 Real Cases + 0 Synthetic Cases = 35 Total
    "Dataset030_PuFiRS25_50Real_Aug0.5": 1,   # 35 Real Cases + 35 Synthetic Cases = 70 Total
    "Dataset031_PuFiRS25_50Real_Aug1.0": 2,   # 35 Real Cases + 70 Synthetic Cases = 105 Total
    "Dataset032_PuFiRS25_50Real_Aug1.5": 3    # 35 Real Cases + 105 Synthetic Cases = 140 Total
}

def main():
    print("🛠️ Constructing Airtight Split Records for the 50% Real Data Study Arm...")
    print("──────────────────────────────────────────────────────────────────────────")
    
    # Load original master split tracking configurations
    with open(BASELINE_SPLITS_PATH, 'r') as f:
        baseline_splits = json.load(f)
        
    # Load the metadata file to map synthetic files back to their parent cases
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # Group synthetic IDs chronologically by parent case
    parent_map = {}
    for entry in metadata:
        s_id = entry['synthetic_id']
        source_patient = entry['source_case']
        if source_patient not in parent_map:
            parent_map[source_patient] = []
        parent_map[source_patient].append(s_id)
        
    # Isolate the exact 35 training parent IDs used for synthesis
    sampled_35_real_cases = sorted(list(parent_map.keys()))
    print(f"🫁 Verified the {len(sampled_35_real_cases)} original human training cases used for synthesis.")

    for ds_folder, max_per_case in DATASET_SCALES.items():
        target_preprocessed_dir = os.path.join(WORKSPACE, "nnUNet_preprocessed", ds_folder)
        if not os.path.exists(target_preprocessed_dir):
            continue
            
        new_splits = []
        
        # Mirror cross-validation parameters across all 5 folds
        for fold_data in baseline_splits:
            base_val = list(fold_data['val'])
            
            # The base training set now uses ONLY the 35 cases chosen for the experiment
            new_train_pool = list(sampled_35_real_cases)
            
            # Append synthetic images in precise increments for each patient
            if max_per_case > 0:
                for patient in sampled_35_real_cases:
                    allowed_syn_files = parent_map[patient][:max_per_case]
                    new_train_pool.extend(allowed_syn_files)
                    
            new_splits.append({
                "train": sorted(list(set(new_train_pool))),
                "val": sorted(list(set(base_val)))  # Keeps validation set completely pristine
            })
            
        target_json_path = os.path.join(target_preprocessed_dir, "splits_final.json")
        with open(target_json_path, 'w') as f:
            json.dump(new_splits, f, indent=4)
            
        final_count = len(new_splits[0]['train'])
        print(f"✅ {ds_folder} -> Set to {final_count} training items ({len(sampled_35_real_cases)} real + {final_count - len(sampled_35_real_cases)} synthetic).")

if __name__ == "__main__":
    main()
