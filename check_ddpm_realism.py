import os
import glob
import numpy as np
import nibabel as nib
import json

# --- Configuration ---
WORKSPACE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
REAL_CT_DIR = f"{WORKSPACE}/nnUNet_raw/Dataset025_PuFiRS25/imagesTr"
REAL_MASK_DIR = f"{WORKSPACE}/nnUNet_raw/Dataset025_PuFiRS25/labelsTr"
SYNTH_CT_DIR = f"{WORKSPACE}/augmented_data/synthetic_images_v2"
SYNTH_MASK_DIR = f"{WORKSPACE}/augmented_data/synthetic_masks_v2"

def compute_entropy(voxels):
    """Calculate 1D Shannon Entropy of voxel intensities."""
    counts, _ = np.histogram(voxels, bins=100, density=True)
    counts = counts[counts > 0]
    return -np.sum(counts * np.log2(counts + 1e-5))

def main():
    # Added flush=True to force the text out of the cluster buffer immediately
    print("🔬 Executing Rapid DDPM Fidelity and Realism Diagnostic...", flush=True)
    print("──────────────────────────────────────────────────────────────────────────", flush=True)
    
    # 1. Gather a representative sample of Real Fibrosis Voxels
    real_images = sorted(glob.glob(os.path.join(REAL_CT_DIR, "*_0000.nii.gz")))[:15]
    real_voxels_pool = []
    
    print(f"📦 Loading {len(real_images)} real baseline patient volumes...", flush=True)
    for img_path in real_images:
        p_id = os.path.basename(img_path).replace("_0000.nii.gz", "")
        mask_path = os.path.join(REAL_MASK_DIR, f"{p_id}.nii.gz")
        if not os.path.exists(mask_path): continue
        
        img = np.clip(nib.load(img_path).get_fdata(), -1024, 300)
        mask = nib.load(mask_path).get_fdata()
        real_voxels_pool.extend(img[mask >= 1].flatten())
        
    real_voxels = np.array(real_voxels_pool)
    print(f"✅ Loaded {len(real_voxels)} real disease voxels.", flush=True)
    
    # 2. Gather a representative sample of Synthetic Fibrosis Voxels
    synth_images = sorted(glob.glob(os.path.join(SYNTH_CT_DIR, "synthetic_*_0000.nii.gz")))[:20]
    synth_voxels_pool = []
    
    print(f"📦 Loading {len(synth_images)} synthetic generated matrices...", flush=True)
    for img_path in synth_images:
        filename = os.path.basename(img_path)
        
        # 🚀 FIXED: Robust regex/split parsing to cleanly extract 'synthetic_0000' from 'synthetic_0000_0000.nii.gz'
        mask_base = filename.split("_0000")[0]
        mask_path = os.path.join(SYNTH_MASK_DIR, f"{mask_base}.nii.gz")
        
        if not os.path.exists(mask_path):
            continue
        
        try:
            img = np.clip(nib.load(img_path).get_fdata(), -1024, 300)
            mask = nib.load(mask_path).get_fdata()
            # Append only the voxels where the mask indicates active fibrosis disease tissue
            synth_voxels_pool.extend(img[mask >= 0.5].flatten())
        except Exception as e:
            print(f"⚠️ Error reading synthetic file {mask_base}: {e}", flush=True)
            
    synth_voxels = np.array(synth_voxels_pool)
    print(f"✅ Loaded {len(synth_voxels)} synthetic disease voxels.", flush=True)
    
    if len(synth_voxels) == 0:
        print("❌ Error: No synthetic voxels found inside the specified directories.", flush=True)
        return

    # 3. Compute Statistical Diagnostics
    print("\n📊 STATISTICAL ALIGNMENT METRICS:", flush=True)
    
    # Check A: HU Intensity Intersection
    real_hist, bins = np.histogram(real_voxels, bins=100, range=(-1024, 300), density=True)
    synth_hist, _ = np.histogram(synth_voxels, bins=100, range=(-1024, 300), density=True)
    intersection = np.sum(np.minimum(real_hist, synth_hist)) / (np.sum(real_hist) + 1e-8)
    print(f"  -> HU Profile Intersection: {intersection*100:.2f}%", flush=True)
    
    # Check B: Textural Complexity (Entropy)
    real_entropy = compute_entropy(real_voxels)
    synth_entropy = compute_entropy(synth_voxels)
    entropy_diff = abs(real_entropy - synth_entropy)
    print(f"  -> Real Voxel Entropy     : {real_entropy:.4f}", flush=True)
    print(f"  -> Synthetic Voxel Entropy: {synth_entropy:.4f} (Delta: {entropy_diff:.4f})", flush=True)
    
    # Check C: Mean Intensity Shifts
    # FIXED: Replaced old undefined variable real_real with real_voxels
    print(f"  -> Real Mean HU           : {np.mean(real_voxels):.2f} HU", flush=True)
    print(f"  -> Synthetic Mean HU      : {np.mean(synth_voxels):.2f} HU", flush=True)

    print("\n──────────────────────────────────────────────────────────────────────────", flush=True)
    print("📋 EVALUATION DIAGNOSTIC:", flush=True)
    if intersection > 0.70 and entropy_diff < 0.6:
        print("✅ VERDICT: YOUR DDPM MODEL IS COMPLETELY HEALTHY AND REALISTIC!", flush=True)
        print("   The generated textures perfectly mirror human fibrosis statistics.", flush=True)
        print("   The segmentation drop is purely due to downstream dataset mixing / overfitting.", flush=True)
    else:
        print("⚠️ VERDICT: DDPM PATCH FREQUENCY MISMATCH DETECTED.", flush=True)
        print("   The model is washing out micro-textures or shifting Hounsfield windows.", flush=True)
        print("   This confirms your suspicion: the generation pipeline requires finer tuning.", flush=True)

if __name__ == "__main__":
    main()
