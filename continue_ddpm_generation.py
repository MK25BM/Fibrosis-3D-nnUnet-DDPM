#!/usr/bin/env python3
"""
Continue DDPM synthetic generation from where batch job left off.
Generates remaining synthetic images to reach target of 140.
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
    "bbox_json": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/pufirs_bboxes_train.json",
    "output_ct_dir": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2",
    "output_mask_dir": "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_masks_v2",
    
    # DDPM parameters
    "input_size": 128,
    "depth_size": 128,
    "num_channels": 64,
    "num_res_blocks": 1,
    "num_class_labels": 2,
    "timesteps": 250,
    "mix_from": 250,
    "patch_stride": 64,
    
    # Continue from batch 1
    "target_total": 140,
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
        return (gaussian_filter(mask.astype(float), sigma=1.0) > 0.5).astype(mask.dtype)
    
    elif perturbation_type == "noise":
        noisy = mask.astype(float) + np.random.normal(0, 0.1, mask.shape)
        return (noisy > 0.5).astype(mask.dtype)
    
    elif perturbation_type == "elastic_transform":
        alpha = 30
        sigma = 5
        shape = mask.shape
        
        dx = np.random.randn(*shape) * alpha
        dy = np.random.randn(*shape) * alpha
        dz = np.random.randn(*shape) * alpha
        
        dx = gaussian_filter(dx, sigma=sigma)
        dy = gaussian_filter(dy, sigma=sigma)
        dz = gaussian_filter(dz, sigma=sigma)
        
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        x_deformed = x + dx
        y_deformed = y + dy
        z_deformed = z + dz
        
        x_deformed = np.clip(x_deformed, 0, shape[0]-1)
        y_deformed = np.clip(y_deformed, 0, shape[1]-1)
        z_deformed = np.clip(z_deformed, 0, shape[2]-1)
        
        try:
            deformed = map_coordinates(mask.astype(float), [x_deformed, y_deformed, z_deformed], order=1, mode='constant', cval=0)
            return (deformed > 0.5).astype(mask.dtype)
        except:
            return mask
    
    elif perturbation_type == "random_scale_translate":
        scale = np.random.uniform(0.8, 1.2)
        tx = np.random.randint(-10, 11)
        ty = np.random.randint(-10, 11)
        tz = np.random.randint(-10, 11)
        
        shape = mask.shape
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        
        center = np.array(shape) / 2
        x_centered = (x - center[0]) / scale + center[0] + tx
        y_centered = (y - center[1]) / scale + center[1] + ty
        z_centered = (z - center[2]) / scale + center[2] + tz
        
        x_centered = np.clip(x_centered, 0, shape[0]-1)
        y_centered = np.clip(y_centered, 0, shape[1]-1)
        z_centered = np.clip(z_centered, 0, shape[2]-1)
        
        try:
            transformed = map_coordinates(mask.astype(float), [x_centered, y_centered, z_centered], order=1, mode='constant', cval=0)
            return (transformed > 0.5).astype(mask.dtype)
        except:
            return mask
    
    elif perturbation_type == "combined":
        ops = ['erosion', 'dilation', 'gaussian_smooth', 'elastic_transform']
        op = np.random.choice(ops)
        return perturb_mask(mask, op)
    
    return mask


def generate_synthetic_with_blending(
    diffusion_model,
    ct_img: np.ndarray,
    mask_img: np.ndarray,
    bbox: dict,
    patch_size: int = 128,
    stride: int = 64,
    max_patches: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic image using Hann windowing blending."""
    
    w_start, w_end = bbox['width']['start'], bbox['width']['end']
    h_start, h_end = bbox['height']['start'], bbox['height']['end']
    d_start, d_end = bbox['depth']['start'], bbox['depth']['end']
    
    synthetic_accum = np.zeros_like(ct_img, dtype=np.float32)
    weight_accum = np.zeros_like(ct_img, dtype=np.float32)
    mask_accum = np.zeros_like(mask_img, dtype=np.float32)
    mask_weight_accum = np.zeros_like(mask_img, dtype=np.float32)
    
    ramp = np.linspace(0, 1, patch_size)
    ramp = np.minimum(ramp, ramp[::-1])
    ramp = ramp / (ramp.max() + 1e-5)
    w_mask = ramp[:, None, None] * ramp[None, :, None] * ramp[None, None, :]
    
    valid_patches = []
    for w in range(w_start, w_end - patch_size + 1, stride):
        for h in range(h_start, h_end - patch_size + 1, stride):
            for d in range(d_start, d_end - patch_size + 1, stride):
                
                local_mask = mask_img[w:w+patch_size, h:h+patch_size, d:d+patch_size]
                
                if np.sum(local_mask >= 1) > 50:
                    valid_patches.append((w, h, d, local_mask.copy()))
    
    if len(valid_patches) == 0:
        return ct_img.copy(), mask_img.copy()
    
    valid_patches = valid_patches[:max_patches]
    
    for patch_idx, (w, h, d, local_mask) in enumerate(valid_patches):
        
        binary_mask = (local_mask >= 1).astype(int)
        mask_one_hot = label2masks(binary_mask)
        input_tensor = torch.tensor(mask_one_hot).float().permute(3, 0, 1, 2).transpose(3, 1).unsqueeze(0).cuda()
        input_tensor = (input_tensor * 2) - 1
        
        local_ct = ct_img[w:w+patch_size, h:h+patch_size, d:d+patch_size]
        local_ct = np.clip(local_ct, -1024, 300)
        local_ct = (local_ct - (-1024)) / (300 - (-1024))
        ct_tensor = torch.tensor(local_ct).float().unsqueeze(0).transpose(3, 1).unsqueeze(0).cuda()
        ct_tensor = (ct_tensor * 2) - 1
        
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                sampled_tensor = diffusion_model.sample_dpm_solver(
                    x_start=ct_tensor,
                    batch_size=1,
                    condition_tensors=input_tensor,
                    mix_from=250
                )
        
        sample_img = np.squeeze(sampled_tensor.cpu().numpy())
        sample_img = (sample_img + 1) / 2
        sample_img = (sample_img * (300 - (-1024))) + (-1024)
        sample_img_oriented = np.transpose(sample_img, (1, 0, 2))
        
        synthetic_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += (sample_img_oriented * w_mask)
        weight_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += w_mask
        
        mask_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += (local_mask * w_mask)
        mask_weight_accum[w:w+patch_size, h:h+patch_size, d:d+patch_size] += w_mask
    
    final_ct = np.copy(ct_img).astype(np.float32)
    final_mask = np.copy(mask_img).astype(np.float32)
    modified = weight_accum > 1e-4
    
    normalized_synthetic = np.zeros_like(ct_img, dtype=np.float32)
    normalized_synthetic[modified] = synthetic_accum[modified] / weight_accum[modified]
    
    normalized_mask = np.zeros_like(mask_img, dtype=np.float32)
    normalized_mask[modified] = mask_accum[modified] / mask_weight_accum[modified]
    
    final_ct[modified] = (normalized_synthetic[modified] * weight_accum[modified]) + \
                         (ct_img[modified] * (1.0 - weight_accum[modified]))
    
    final_mask[modified] = (normalized_mask[modified] * mask_weight_accum[modified]) + \
                           (mask_img[modified] * (1.0 - mask_weight_accum[modified]))
    
    return final_ct, final_mask


def get_next_synthetic_id():
    """Get the next available synthetic ID based on existing files."""
    ct_dir = CONFIG["output_ct_dir"]
    existing_files = glob.glob(os.path.join(ct_dir, "synthetic_*.nii.gz"))
    
    if not existing_files:
        return 0
    
    # Extract numeric IDs and get max
    ids = []
    for f in existing_files:
        try:
            base = os.path.basename(f)
            num_str = base.split('_')[1]
            ids.append(int(num_str))
        except:
            continue
    
    return max(ids) + 1 if ids else 0


def main():
    print("\n" + "="*80)
    print("CONTINUING DDPM SYNTHETIC GENERATION (Interactive)")
    print("="*80)
    
    # Check existing progress
    existing_ct = len(glob.glob(os.path.join(CONFIG["output_ct_dir"], "synthetic_*.nii.gz")))
    existing_mask = len(glob.glob(os.path.join(CONFIG["output_mask_dir"], "synthetic_*.nii.gz")))
    
    print(f"\nExisting synthetic images: {existing_ct}")
    print(f"Existing synthetic masks: {existing_mask}")
    print(f"Target: {CONFIG['target_total']}")
    print(f"Remaining: {CONFIG['target_total'] - existing_ct}")
    
    if existing_ct >= CONFIG['target_total']:
        print("\n✓ Already reached target! No additional generation needed.")
        return
    
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
    
    # Load official train/val split
    splits_path = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed/Dataset025_PuFiRS25/splits_final.json"
    with open(splits_path, 'r') as f:
        splits = json.load(f)
        official_train_ids = set(splits[0]['train'])
    
    # Get list of all images
    ct_files = sorted(glob.glob(os.path.join(CONFIG["ct_folder"], "*_0000.nii.gz")))
    mask_files = sorted(glob.glob(os.path.join(CONFIG["mask_folder"], "*.nii.gz")))
    
    # Filter to official training cases only
    valid_indices = []
    for idx, ct_file in enumerate(ct_files):
        patient_id = os.path.basename(ct_file).replace("_0000.nii.gz", "")
        if patient_id in official_train_ids:
            valid_indices.append(idx)
    
    print(f"Found {len(valid_indices)} official TRAINING cases")
    
    # Randomly select from training set
    np.random.seed(CONFIG["seed"])
    selected_indices = np.random.choice(
        valid_indices,
        min(35, len(valid_indices)),
        replace=False
    )
    
    print(f"Selected {len(selected_indices)} cases for generation")
    
    # Get starting synthetic ID
    start_id = get_next_synthetic_id()
    synthetic_count = existing_ct
    
    print(f"\nStarting from synthetic ID: {start_id}")
    print(f"Current count: {synthetic_count}/{CONFIG['target_total']}")
    
    synthetic_metadata = []
    remaining = CONFIG['target_total'] - existing_ct
    
    if remaining <= 0:
        print("\n✓ Already at target!")
        return
    
    print(f"\n⚠️  Only generating {remaining} additional images to reach target")
    
    # Only generate enough to reach target - simple approach: take last case + next perturbations
    # or continue from where we left off
    perturbation_types = ['erosion', 'dilation', 'gaussian_smooth', 'noise', 'elastic_transform', 'random_scale_translate', 'combined']
    
    # Since we don't know exactly which case/perturbation we left off at, generate from a specific case
    # to ensure we get exactly the right number
    target_case_idx = -1  # Start from last case
    idx = selected_indices[target_case_idx]
    
    ct_file = os.path.basename(ct_files[idx])
    patient_id = ct_file.replace("_0000.nii.gz", "").replace(".nii.gz", "")
    
    # Load CT and mask
    ct_nifti = nib.load(ct_files[idx])
    ct_img = ct_nifti.get_fdata().astype(np.float32)
    mask_img = nib.load(mask_files[idx]).get_fdata()
    
    bbox_entry = next((b for b in bboxes if b['patient_id'] == patient_id), None)
    if bbox_entry is None:
        print(f"ERROR: Could not find bbox for {patient_id}")
        return
    
    bbox = bbox_entry['bounding_box']
    
    # Generate only the remaining perturbations
    for perturb_idx, perturb_type in enumerate(perturbation_types):
        
        if synthetic_count >= CONFIG['target_total']:
            print(f"\n✓ Reached target of {CONFIG['target_total']} synthetic images!")
            break
        
        perturbed_mask = perturb_mask(mask_img, perturb_type)
        
        try:
            print(f"[{synthetic_count+1}/{CONFIG['target_total']}] {patient_id} ({perturb_type})...", end=" ", flush=True)
            
            synthetic_ct, synthetic_mask = generate_synthetic_with_blending(
                diffusion,
                ct_img,
                perturbed_mask,
                bbox,
                patch_size=CONFIG["input_size"],
                stride=CONFIG["patch_stride"],
                max_patches=3
            )
            
            # Save outputs
            ct_out_path = os.path.join(
                CONFIG["output_ct_dir"],
                f"synthetic_{start_id + synthetic_count:04d}_0000.nii.gz"
            )
            mask_out_path = os.path.join(
                CONFIG["output_mask_dir"],
                f"synthetic_{start_id + synthetic_count:04d}.nii.gz"
            )
            
            nib.save(nib.Nifti1Image(synthetic_ct.astype(np.float32), ct_nifti.affine, ct_nifti.header), ct_out_path)
            nib.save(nib.Nifti1Image(synthetic_mask.astype(np.float32), ct_nifti.affine, ct_nifti.header), mask_out_path)
            
            synthetic_metadata.append({
                "synthetic_id": f"synthetic_{start_id + synthetic_count:04d}",
                "source_case": patient_id,
                "perturbation_type": perturb_type,
                "batch_number": 2,  # Continuation batch
                "ct_file": ct_out_path,
                "mask_file": mask_out_path
            })
            
            synthetic_count += 1
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            continue
    
    torch.cuda.empty_cache()
    del ct_img, mask_img
    
    # Save continuation metadata
    if synthetic_metadata:
        metadata_path = os.path.join(CONFIG["output_ct_dir"], "synthetic_metadata_batch2_continuation.json")
        with open(metadata_path, 'w') as f:
            json.dump(synthetic_metadata, f, indent=2)
        
        print(f"\n✓ Continuation metadata: {metadata_path}")
    
    print(f"\n✓ Continuation complete!")
    print(f"✓ Total synthetic images: {synthetic_count}/{CONFIG['target_total']}")
    print(f"✓ CT volumes: {CONFIG['output_ct_dir']}")
    print(f"✓ Masks: {CONFIG['output_mask_dir']}")


if __name__ == "__main__":
    main()
