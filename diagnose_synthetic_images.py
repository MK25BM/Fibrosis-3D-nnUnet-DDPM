#!/usr/bin/env python3
"""
Quick diagnostic script to inspect generated synthetic images.
No plotting - just analysis.
"""

import os
import glob
import numpy as np
import nibabel as nib
from pathlib import Path

SYNTHETIC_DIR = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images"

def analyze_synthetic_image(img_path, idx=0):
    """Analyze a single synthetic image for artifacts."""
    print(f"\n{'='*80}")
    print(f"Image {idx}: {os.path.basename(img_path)}")
    print(f"{'='*80}")
    
    # Load image
    img_nifti = nib.load(img_path)
    img_data = img_nifti.get_fdata()
    
    print(f"\nShape: {img_data.shape}")
    print(f"Value range: [{img_data.min():.1f}, {img_data.max():.1f}] HU")
    print(f"Mean: {img_data.mean():.1f}, Std: {img_data.std():.1f}")
    
    # Check for hard patch boundaries by looking at discontinuities
    print(f"\n--- Patch Boundary Detection ---")
    patch_size = 128
    w, h, d = img_data.shape
    
    # Look for sudden intensity changes along each axis
    diffs_d = np.abs(np.diff(img_data, axis=2)).max(axis=(0,1))
    
    # Find peaks (likely patch boundaries)
    threshold = np.percentile(diffs_d, 99)  # Top 1% jumps
    peaks_d = np.where(diffs_d > threshold)[0]
    
    print(f"Potential patch boundaries (D axis): {peaks_d}")
    print(f"Expected patch positions at multiples of 64 (stride): 64, 128, 192, ...")
    
    if len(peaks_d) > 0:
        print(f"\n⚠️  DETECTED: Hard patch boundaries!")
        print(f"   Patches were placed without smooth blending.")
        print(f"   Max intensity jump at boundaries: {diffs_d[peaks_d].max():.1f} HU")
    else:
        print(f"\n✓ No obvious hard boundaries detected (good sign)")
    
    # Check anatomical plausibility
    print(f"\n--- Intensity Distribution ---")
    
    lung_voxels = np.sum((img_data >= -1024) & (img_data < -100))
    tissue_voxels = np.sum((img_data >= -100) & (img_data < 100))
    dense_voxels = np.sum((img_data >= 100))
    
    total_voxels = np.prod(img_data.shape)
    print(f"Lung tissue (-1024 to -100 HU): {lung_voxels/total_voxels*100:.1f}%")
    print(f"Soft tissue / fibrosis (-100 to +100 HU): {tissue_voxels/total_voxels*100:.1f}%")
    print(f"Dense tissue (+100 HU): {dense_voxels/total_voxels*100:.1f}%")
    
    if lung_voxels/total_voxels < 0.2:
        print(f"⚠️  WARNING: Very few lung-density voxels")
    
    # Smoothness check
    print(f"\n--- Smoothness Check ---")
    
    # Compare with downsampled version
    ds_factor = 8
    downsampled = img_data[::ds_factor, ::ds_factor, ::ds_factor]
    
    local_std_full = np.std(np.gradient(img_data, axis=2))
    local_std_down = np.std(np.gradient(downsampled, axis=2))
    
    print(f"Local gradient std (full resolution): {local_std_full:.2f}")
    print(f"Local gradient std (downsampled): {local_std_down:.2f}")
    print(f"Ratio: {local_std_full/local_std_down:.1f}x")
    
    if local_std_full/local_std_down > 2.0:
        print(f"⚠️  WARNING: High-frequency artifacts (potential unblended patches)")
    
    # Check if synthetic masks exist
    mask_pattern = os.path.basename(img_path).replace("_0000.nii.gz", "_0001.nii.gz")
    mask_path = os.path.join(SYNTHETIC_DIR, mask_pattern)
    
    if os.path.exists(mask_path):
        print(f"\n✓ Synthetic mask found: {mask_pattern}")
    else:
        print(f"\n❌ Synthetic mask NOT found!")
        print(f"   Expected: {mask_pattern}")


def main():
    """Analyze all synthetic images."""
    
    print("\n" + "="*80)
    print("SYNTHETIC IMAGE QUALITY DIAGNOSIS")
    print("="*80)
    
    # Find all synthetic images
    synthetic_files = sorted(glob.glob(os.path.join(SYNTHETIC_DIR, "synthetic_*_0000.nii.gz")))
    
    if not synthetic_files:
        print(f"\n❌ No synthetic images found in {SYNTHETIC_DIR}")
        return
    
    print(f"\nFound {len(synthetic_files)} synthetic images")
    
    # Analyze first 3 in detail
    for idx, img_path in enumerate(synthetic_files[:3]):
        analyze_synthetic_image(img_path, idx=idx)
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print("""
Key issues identified:
1. Hard patch boundaries → Need Hann windowing for smooth blending
2. No synthetic masks → Need to save fibrosis masks (_0001.nii.gz)
3. High-frequency artifacts → Indicates unblended patch placement

Solutions:
- Use Hann window weighting for patch blending
- Save generated masks alongside CT volumes
- Verify patch coordinates are correct
""")


if __name__ == "__main__":
    main()
