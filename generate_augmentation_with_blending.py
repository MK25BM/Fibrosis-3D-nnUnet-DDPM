#!/usr/bin/env python3
"""
Augmentation wrapper using sample.py's proven blending pipeline.

Instead of reimplementing blending, this reuses sample.py's native patch blending
which has proper Hann windowing and generates both CT and masks.
"""

import os
import sys
import json
import glob
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from typing import Tuple
from scipy.ndimage import binary_erosion, binary_dilation, gaussian_filter
from tqdm import tqdm
from scipy.ndimage import map_coordinates
import random

# Import DDPM modules
LUNG_DDPM_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS"
if LUNG_DDPM_PATH not in sys.path:
    sys.path.insert(0, LUNG_DDPM_PATH)

from diffusion_model.trainer import GaussianDiffusion
from diffusion_model.unet import create_model

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "workspace": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM",
    "ct_folder": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/imagesTr",
    "mask_folder": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTr",
    "ddpm_checkpoint": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-21T131022/model-23.pt",
    "bbox_json": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/pufirs_bboxes_train.json",  # FIXED: Training bboxes!
    "output_ct_dir": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2",
    "output_mask_dir": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_masks_v2",
    
    # DDPM parameters (match training)
    "input_size": 128,
    "depth_size": 128,
    "num_channels": 64,
    "num_res_blocks": 1,
    "num_class_labels": 2,
    "timesteps": 250,
    "mix_from": 250,
    "patch_stride": 64,  # 50% overlap for smooth blending
    
    # Sampling config
    "num_cases_to_sample": 35,  # All 35 of 70 training cases (50% of training set)
    "perturbations_per_case": 4,  # 35 × 4 = 140 synthetic images
    "batch_number": 1,  # Single batch for simplicity
    "seed": 42,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def label2masks(masked_img):
    """Convert binary mask to one-hot format for DDPM conditioning."""
    result_img = np.zeros(masked_img.shape + (1,))
    result_img[masked_img >= 1, 0] = 1.0
    return result_img


def perturb_mask(mask: np.ndarray, perturbation_type: str) -> np.ndarray:
    """Apply morphological and spatial perturbation to fibrosis mask."""
    
    if perturbation_type == "erosion":
        return binary_erosion(mask, iterations=1).astype(mask.dtype)
    
    elif perturbation_type == "dilation":
        return binary_dilation(mask, iterations=1).astype(mask.dtype)
    
    elif perturbation_type == "gaussian_smooth":
        # Smooth the mask boundary (soften edges)
        return (gaussian_filter(mask.astype(float), sigma=1.0) > 0.5).astype(mask.dtype)
    
    elif perturbation_type == "noise":
        # Add small random noise and binarize
        noisy = mask.astype(float) + np.random.normal(0, 0.1, mask.shape)
        return (noisy > 0.5).astype(mask.dtype)
    
    elif perturbation_type == "elastic_transform":
        # Elastic deformation (MONAI-like): spatial warping
        # Create displacement field
        alpha = 30  # Deformation strength
        sigma = 5   # Smoothness
        shape = mask.shape
        
        # Random displacement fields
        dx = np.random.randn(*shape) * alpha
        dy = np.random.randn(*shape) * alpha
        dz = np.random.randn(*shape) * alpha
        
        # Smooth displacement fields
        dx = gaussian_filter(dx, sigma=sigma)
        dy = gaussian_filter(dy, sigma=sigma)
        dz = gaussian_filter(dz, sigma=sigma)
        
        # Apply displacement to coordinates
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        x_deformed = x + dx
        y_deformed = y + dy
        z_deformed = z + dz
        
        # Clamp coordinates to valid range
        x_deformed = np.clip(x_deformed, 0, shape[0]-1)
        y_deformed = np.clip(y_deformed, 0, shape[1]-1)
        z_deformed = np.clip(z_deformed, 0, shape[2]-1)
        
        # Interpolate mask at deformed coordinates
        try:
            deformed = map_coordinates(mask.astype(float), [x_deformed, y_deformed, z_deformed], order=1, mode='constant', cval=0)
            return (deformed > 0.5).astype(mask.dtype)
        except:
            return mask  # Fallback if interpolation fails
    
    elif perturbation_type == "random_scale_translate":
        # Random scale and translation
        scale = np.random.uniform(0.8, 1.2)  # 80-120% zoom
        tx = np.random.randint(-10, 11)  # ±10 voxel translation
        ty = np.random.randint(-10, 11)
        tz = np.random.randint(-10, 11)
        
        shape = mask.shape
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        # Center, scale, translate
        center = np.array(shape) / 2
        x_centered = (x - center[0]) / scale + center[0] + tx
        y_centered = (y - center[1]) / scale + center[1] + ty
        z_centered = (z - center[2]) / scale + center[2] + tz
        
        # Clamp and interpolate
        x_centered = np.clip(x_centered, 0, shape[0]-1)
        y_centered = np.clip(y_centered, 0, shape[1]-1)
        z_centered = np.clip(z_centered, 0, shape[2]-1)
        
        try:
            transformed = map_coordinates(mask.astype(float), [x_centered, y_centered, z_centered], order=1, mode='constant', cval=0)
            return (transformed > 0.5).astype(mask.dtype)
        except:
            return mask
    
    elif perturbation_type == "combined":
        # Random combination of above perturbations
        ops = ['erosion', 'dilation', 'gaussian_smooth', 'elastic_transform']
        op = np.random.choice(ops)
        return perturb_mask(mask, op)
    
    return mask  # No perturbation


def generate_synthetic_with_blending(
    diffusion_model,
    ct_img: np.ndarray,
    mask_img: np.ndarray,
    bbox: dict,
    patch_size: int = 128,
    stride: int = 64,
    max_patches: int = 3  # Sample up to 3 patches per case for diversity
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic image using sample.py's Hann windowing blending approach.
    
    Returns: (synthetic_ct, synthetic_mask)
    """
    
    w_start, w_end = bbox['width']['start'], bbox['width']['end']
    h_start, h_end = bbox['height']['start'], bbox['height']['end']
    d_start, d_end = bbox['depth']['start'], bbox['depth']['end']
    
    # Initialize canvases with blending weights (like sample.py)
    synthetic_accum = np.zeros_like(ct_img, dtype=np.float32)
    weight_accum = np.zeros_like(ct_img, dtype=np.float32)
    mask_accum = np.zeros_like(mask_img, dtype=np.float32)
    mask_weight_accum = np.zeros_like(mask_img, dtype=np.float32)
    
    # Build Hann window for smooth blending (from sample.py, lines 89-92)
    ramp = np.linspace(0, 1, patch_size)
    ramp = np.minimum(ramp, ramp[::-1])
    ramp = ramp / (ramp.max() + 1e-5)
    w_mask = ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :]
    
    # Collect all valid patches first
    valid_patches = []
    for w in range(w_start, w_end - patch_size + 1, stride):
        for h in range(h_start, h_end - patch_size + 1, stride):
            for d in range(d_start, d_end - patch_size + 1, stride):
                
                local_mask = mask_img[w:w+patch_size, h:h+patch_size, d:d+patch_size]
                
                # Only sample patches with sufficient fibrosis (>50 voxels)
                if np.sum(local_mask >= 1) > 50:
                    valid_patches.append((w, h, d, local_mask.copy()))
    
    # LIMIT: Only process first N patches to avoid hanging
    if len(valid_patches) == 0:
        # No valid patches - return original
        return ct_img.copy(), mask_img.copy()
    
    valid_patches = valid_patches[:max_patches]
    
    # Process patches
    for patch_idx, (w, h, d, local_mask) in enumerate(valid_patches):
        
        # Prepare condition mask (from sample.py, line 96-98)
        binary_mask = (local_mask >= 1).astype(int)
        mask_one_hot = label2masks(binary_mask)
        input_tensor = torch.tensor(mask_one_hot).float().permute(3, 0, 1, 2).transpose(3, 1).unsqueeze(0).cuda()
        input_tensor = (input_tensor * 2) - 1
        
        # Prepare CT (from sample.py, line 100-105)
        local_ct = ct_img[w:w+patch_size, h:h+patch_size, d:d+patch_size]
        local_ct = np.clip(local_ct, -1024, 300)
        local_ct = (local_ct - (-1024)) / (300 - (-1024))
        ct_tensor = torch.tensor(local_ct).float().unsqueeze(0).transpose(3, 1).unsqueeze(0).cuda()
        ct_tensor = (ct_tensor * 2) - 1
        
        # Sample from DDPM (from sample.py, line 107-113)
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                sampled_tensor = diffusion_model.sample_dpm_solver(
                    x_start=ct_tensor,
                    batch_size=1,
                    condition_tensors=input_tensor,
                    mix_from=250
                )
        
        # Denormalize (from sample.py, line 114-117)
        sample_img = np.squeeze(sampled_tensor.cpu().numpy())
        sample_img = (sample_img + 1) / 2
        sample_img = (sample_img * (300 - (-1024))) + (-1024)
        sample_img_oriented = np.transpose(sample_img, (1, 0, 2))
        
        # Accumulate with Hann windowing (from sample.py, line 119-122)
        synthetic_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += (sample_img_oriented * w_mask)
        weight_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += w_mask
        
        # Also accumulate mask (same blending)
        mask_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += (local_mask * w_mask)
        mask_weight_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += w_mask
    
    # Final re-composition (from sample.py, line 124-130)
    final_ct = np.copy(ct_img).astype(np.float32)
    final_mask = np.copy(mask_img).astype(np.float32)
    modified = weight_accum > 1e-4
    
    # Normalize overlapping regions
    normalized_synthetic = np.zeros_like(ct_img, dtype=np.float32)
    normalized_synthetic[modified] = synthetic_accum[modified] / weight_accum[modified]
    
    normalized_mask = np.zeros_like(mask_img, dtype=np.float32)
    normalized_mask[modified] = mask_accum[modified] / mask_weight_accum[modified]
    
    # Blend back into original
    final_ct[modified] = (normalized_synthetic[modified] * weight_accum[modified]) + \
                         (ct_img[modified] * (1.0 - weight_accum[modified]))
    
    final_mask[modified] = (normalized_mask[modified] * mask_weight_accum[modified]) + \
                           (mask_img[modified] * (1.0 - mask_weight_accum[modified]))
    
    return final_ct, final_mask


def main():
    print("\n" + "="*80)
    print("AUGMENTATION WITH PROPER BLENDING (Using sample.py approach)")
    print("="*80)
    
    # Create output directories
    os.makedirs(CONFIG["output_ct_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_mask_dir"], exist_ok=True)
    
    # Load DDPM model
    print("\nLoading DDPM model...")
    model = create_model(
        image_size=CONFIG["input_size"],
        num_channels=CONFIG["num_channels"],
        num_res_blocks=CONFIG["num_res_blocks"],
        in_channels=CONFIG["num_class_labels"],
        out_channels=1,
        channel_mult=(1, 2, 3, 4),
        attention_resolutions="16"
    ).cuda()
    
    diffusion = GaussianDiffusion(
        model,
        image_size=CONFIG["input_size"],
        depth_size=CONFIG["depth_size"],
        timesteps=CONFIG["timesteps"],
        loss_type='l1',
        channels=1
    ).cuda()
    
    checkpoint = torch.load(CONFIG["ddpm_checkpoint"], map_location='cuda')
    diffusion.load_state_dict(checkpoint['ema'])
    print("✓ DDPM model loaded")
    
    # Load bbox information
    with open(CONFIG["bbox_json"], 'r') as f:
        bboxes = json.load(f)
    
    # 🔒 ENFORCED SPLIT LOOKUP: Read the master baseline splits final file explicitly first
    splits_path = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed/Dataset025_PuFiRS25/splits_final.json"
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        official_train_ids = sorted(list(set(splits[0]['train'])))  # The strict 70 training IDs
    
    # Generate file paths strictly from the approved training IDs pool only
    ct_files = []
    mask_files = []
    for p_id in official_train_ids:
        target_ct = os.path.join(CONFIG["ct_folder"], f"{p_id}_0000.nii.gz")
        target_mask = os.path.join(CONFIG["mask_folder"], f"{p_id}.nii.gz")
        
        # Verify files exist on your cluster arrays before appending to targets
        if os.path.exists(target_ct) and os.path.exists(target_mask):
            ct_files.append(target_ct)
            mask_files.append(target_mask)

    print(f"Found {len(ct_files)} verified training case paths in imagesTr/labelsTr matching split files.")
    
    # Randomly select subset indices out of the verified clean training list only
    np.random.seed(CONFIG["seed"])
    num_to_sample = min(CONFIG["num_cases_to_sample"], len(ct_files))
    selected_indices = np.random.choice(len(ct_files), num_to_sample, replace=False)
    
    print(f"🎯 Selected {len(selected_indices)} unique training cases for synthetic texture generation.")
    
    synthetic_count = 0
    synthetic_metadata = []  # Track which case + perturbation each synthetic image came from
    
    for case_idx, idx in enumerate(tqdm(selected_indices, desc="Processing cases")):
        
        # Extract patient ID from the clean pool
        ct_file_base = os.path.basename(ct_files[idx])
        patient_id = ct_file_base.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        
        # Load CT and mask natively
        ct_nifti = nib.load(ct_files[idx])
        ct_img = ct_nifti.get_fdata().astype(np.float32)
        mask_img = nib.load(mask_files[idx]).get_fdata()
        
        # Find bounding box
        bbox_entry = next((b for b in bboxes if b['patient_id'] == patient_id), None)
        if bbox_entry is None:
            print(f"⚠️ Warning: Missing bbox coordinates layout for training case {patient_id}. Skipping.")
            continue
        
        bbox = bbox_entry['bounding_box']
        
        # Generate perturbed versions
        perturbation_types = [
            'erosion', 'dilation', 'gaussian_smooth', 'noise',
            'elastic_transform', 'random_scale_translate', 'combined'
        ]
        
        for perturb_idx, perturb_type in enumerate(perturbation_types[:CONFIG["perturbations_per_case"]]):
            
            # Perturb mask
            perturbed_mask = perturb_mask(mask_img, perturb_type)
            
            try:
                # Generate synthetic with blending (3 patches per case for diversity)
                print(f"    Generating synthetic ({perturb_type})...", end=" ", flush=True)
                synthetic_ct, synthetic_mask = generate_synthetic_with_blending(
                    diffusion,
                    ct_img,
                    perturbed_mask,
                    bbox,
                    patch_size=CONFIG["input_size"],
                    stride=CONFIG["patch_stride"],
                    max_patches=3  # Sample up to 3 patches per case
                )
                print("✓")
                
                # Save outputs with descriptive names
                ct_out_path = os.path.join(
                    CONFIG["output_ct_dir"],
                    f"synthetic_{synthetic_count:04d}_0000.nii.gz"
                )
                mask_out_path = os.path.join(
                    CONFIG["output_mask_dir"],
                    f"synthetic_{synthetic_count:04d}.nii.gz"
                )
                
                nib.save(nib.Nifti1Image(synthetic_ct.astype(np.float32), ct_nifti.affine, ct_nifti.header), ct_out_path)
                nib.save(nib.Nifti1Image(synthetic_mask.astype(np.float32), ct_nifti.affine, ct_nifti.header), mask_out_path)
                
                # Track metadata for traceability
                synthetic_metadata.append({
                    "synthetic_id": f"synthetic_{synthetic_count:04d}",
                    "source_case": patient_id,
                    "perturbation_type": perturb_type,
                    "batch_number": CONFIG.get("batch_number", 1),
                    "ct_file": ct_out_path,
                    "mask_file": mask_out_path
                })
                
                synthetic_count += 1
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Clear GPU memory after each case to prevent OOM
        torch.cuda.empty_cache()
        del ct_img, mask_img
    
    # Save summary
    summary = {
        "total_generated": synthetic_count,
        "method": "DDPM-inpainting-with-hann-blending (sample.py approach)",
        "patch_size": CONFIG["input_size"],
        "patch_stride": CONFIG["patch_stride"],
        "training_cases_sampled": len(selected_indices),
        "perturbations_per_case": CONFIG["perturbations_per_case"],
        "output_ct_directory": CONFIG["output_ct_dir"],
        "output_mask_directory": CONFIG["output_mask_dir"]
    }
    
    # Save summary and metadata mapping
    summary = {
        "total_generated": synthetic_count,
        "batch_number": CONFIG.get("batch_number", 1),
        "method": "DDPM-inpainting-with-hann-blending (sample.py approach)",
        "patch_size": CONFIG["input_size"],
        "patch_stride": CONFIG["patch_stride"],
        "training_cases_sampled": len(selected_indices),
        "perturbations_per_case": CONFIG["perturbations_per_case"],
        "perturbation_types_used": [
            'erosion', 'dilation', 'gaussian_smooth', 'noise',
            'elastic_transform', 'random_scale_translate', 'combined'
        ],
        "output_ct_directory": CONFIG["output_ct_dir"],
        "output_mask_directory": CONFIG["output_mask_dir"]
    }
    
    summary_path = os.path.join(CONFIG["output_ct_dir"], "generation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed metadata mapping for traceability
    metadata_path = os.path.join(CONFIG["output_ct_dir"], f"synthetic_metadata_batch{CONFIG.get('batch_number', 1)}.json")
    with open(metadata_path, 'w') as f:
        json.dump(synthetic_metadata, f, indent=2)
    
    print(f"\n✓ Generated {synthetic_count} synthetic images with proper blending")
    print(f"✓ CT volumes: {CONFIG['output_ct_dir']}")
    print(f"✓ Masks: {CONFIG['output_mask_dir']}")
    print(f"✓ Summary: {summary_path}")
    print(f"✓ Metadata mapping: {metadata_path}")
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
