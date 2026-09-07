import os
import json
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import monai.transforms as transforms

# Define target paths manually matching your nnUNet grid configuration
SPLITS_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed/Dataset025_PuFiRS25/splits_final.json"
MASK_FOLDER = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTr"
BBOX_JSON = "pufirs_bboxes.json"

# Output folder for storage
PERTURBED_MASK_OUT = "evaluation/perturbed_masks/"
os.makedirs(PERTURBED_MASK_OUT, exist_ok=True)

def find_case(bboxes, patient_id, fold_num=0, sample_idx=0):
    """Safely extracts matching box from JSON registry."""
    found_bbox = next((item['bounding_box'] for item in bboxes 
                      if item['patient_id'] == patient_id 
                      and item['fold'] == str(fold_num).zfill(2) 
                      and item['sample_idx'] == str(sample_idx).zfill(2)), None)
    if found_bbox is None:
        return None
    return [found_bbox['width']['start'], found_bbox['width']['end'],
            found_bbox['height']['start'], found_bbox['height']['end'],
            found_bbox['depth']['start'], found_bbox['depth']['end']]

def main():
    # 1. Load your training cases split validation arrays
    with open(SPLITS_PATH, 'r') as f:
        train_case_ids = json.load(f)[0]['train']
        
    with open(BBOX_JSON, 'r') as f:
        bboxes = json.load(f)

    # 2. Define the perturbation sequence using MONAI transforms
    # We use RandAffine and RandFlip to change shape and size dynamically
    perturbation_transform = transforms.Compose([
        transforms.AddChanneld(keys=["mask"]), # Shape becomes: [1, W, H, D]
        transforms.RandAffined(
            keys=["mask"],
            prob=1.0,
            rotate_range=(0.1, 0.1, 0.1),       # Small rotations
            scale_range=(0.15, 0.15, 0.15),     # ±15% size change
            translate_range=(3, 3, 2),          # Pixel grid translations
            mode="nearest"
        ),
        transforms.SqueezeDimd(keys=["mask"], dim=0) # Back to original [W, H, D]
    ])

    print(f"🧬 Loaded tracking matrix split: targetting {len(train_case_ids)} valid fibrosis training profiles.")

    for patient_id in tqdm(train_case_ids):
        mask_path = os.path.join(MASK_FOLDER, f"{patient_id}.nii.gz")
        if not os.path.exists(mask_path):
            continue
            
        # Load label volume safely
        ref = nib.load(mask_path)
        ref_img = ref.get_fdata()

        # Skip empty profiles early 
        if np.sum(ref_img >= 1) == 0:
            continue

        # Look up bounding box
        bbox = find_case(bboxes, patient_id, fold_num=0, sample_idx=0)
        if bbox is None:
            continue

        # Isolate the original region block matrix
        cropped_mask = np.copy(ref_img[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]])

        # Execute data transformations dictionaries
        data_dict = {"mask": cropped_mask}
        perturbed_dict = perturbation_transform(data_dict)
        perturbed_crop = perturbed_dict["mask"].astype(ref_img.dtype)

        # Build a modified background canvas mask volume copy
        new_mask_volume = np.copy(ref_img)
        new_mask_volume[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = perturbed_crop

        # Write the newly constructed mask volume out to disk space
        out_name = f"perturbed_mask_{patient_id}.nii.gz"
        out_path = os.path.join(PERTURBED_MASK_OUT, out_name)
        
        nifti_mask = nib.Nifti1Image(new_mask_volume, affine=ref.affine, header=ref.header)
        nib.save(nifti_mask, out_path)

    print(f"🎉 Perturbation complete. New masks stored safely in: {PERTURBED_MASK_OUT}")

if __name__ == "__main__":
    main()
