import os
import glob
import json

WORKSPACE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
RAW_BASE = os.path.join(WORKSPACE, "nnUNet_raw")
PREPROCESSED_BASE = os.path.join(WORKSPACE, "nnUNet_preprocessed")

DATASETS = [
    "Dataset029_PuFiRS25_50Real",
    "Dataset030_PuFiRS25_50Real_Aug0.5",
    "Dataset031_PuFiRS25_50Real_Aug1.0",
    "Dataset032_PuFiRS25_50Real_Aug1.5"
]

def main():
    print("🧹 Cleaning out wildcard links and hard-locking physical files to target split sizes...")
    print("──────────────────────────────────────────────────────────────────────────")
    
    for ds in DATASETS:
        images_dir = os.path.join(RAW_BASE, ds, "imagesTr")
        labels_dir = os.path.join(RAW_BASE, ds, "labelsTr")
        
        # 1. Completely clear out the incorrect wildcard links
        if os.path.exists(images_dir):
            for f in glob.glob(os.path.join(images_dir, "*")): os.remove(f)
        if os.path.exists(labels_dir):
            for f in glob.glob(os.path.join(labels_dir, "*")): os.remove(f)
            
        # 2. Read your corrected split configuration file to get the exact target cases
        splits_path = os.path.join(PREPROCESSED_BASE, ds, "splits_final.json")
        with open(splits_path, 'r') as f:
            splits = json.load(f)
            
        # Compile the comprehensive active case pool (Train + Val)
        required_cases = set()
        for fold in splits:
            required_cases.update(fold['train'])
            required_cases.update(fold['val'])
            
        # 3. Create strict symlinks for ONLY the required files
        for case in sorted(list(required_cases)):
            if "synthetic_" in case:
                src_ct = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset028_PuFiRS25_Aug2.0/imagesTr/{case}_0000.nii.gz"
                src_lbl = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset028_PuFiRS25_Aug2.0/labelsTr/{case}.nii.gz"
            else:
                src_ct = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/imagesTr/{case}_0000.nii.gz"
                src_lbl = f"/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTr/{case}.nii.gz"
                
            if os.path.exists(src_ct) and os.path.exists(src_lbl):
                os.symlink(src_ct, os.path.join(images_dir, f"{case}_0000.nii.gz"))
                os.symlink(src_lbl, os.path.join(labels_dir, f"{case}.nii.gz"))

        # Print the resulting physical file count to confirm success
        final_count = len(os.listdir(labels_dir))
        print(f"✅ {ds} -> Raw folder locked at exactly {final_count} files (Real + Synthetic).")

if __name__ == "__main__":
    main()
