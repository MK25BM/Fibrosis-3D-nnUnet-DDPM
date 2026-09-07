import os
import nibabel as nib
import numpy as np

# --- Configuration Paths ---
SYNTHETIC_BASE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/checkpoint_outputs"
TARGET_PATIENT_ID = "PuFiRS25_00034"
OUTPUT_DIR = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/native_patches"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("✂️ Extracting raw, 128x128x128 native patches for direct texture comparison...")
    
    target_stages = [1, 5, 10, 15, 20]
    
    for stage in target_stages:
        stage_path = os.path.join(SYNTHETIC_BASE, f"stage_{stage}", f"aug_synthetic_{TARGET_PATIENT_ID}.nii.gz")
        if not os.path.exists(stage_path): continue
        
        vol_obj = nib.load(stage_path)
        vol_data = vol_obj.get_fdata()
        
        # ✅ FIX: Extract a true, unwarped 128-cube from the core disease region
        # This matches the exact resolution space where your model trained
        native_patch = vol_data[100:228, 150:278, 50:178]
        
        out_name = os.path.join(OUTPUT_DIR, f"native_patch_stage_{stage}.nii.gz")
        nib.save(nib.Nifti1Image(native_patch.astype(np.float32), np.eye(4)), out_name)
        print(f"✅ Saved 128x128x128 native patch: {out_name}")

if __name__ == "__main__":
    main()
