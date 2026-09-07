#!/usr/bin/env python3
"""
Direct evaluation of baseline model without using nnUNetv2_predict CLI.
Load the checkpoint directly and run inference.
"""
import os
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from collections import defaultdict
import scipy.stats as stats
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json

# Add nnUNet to path
import sys
sys.path.insert(0, '/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Metrics computation
def compute_dice(pred, gt):
    """Compute DICE coefficient"""
    intersection = np.sum(pred * gt)
    if (np.sum(pred) + np.sum(gt)) == 0:
        return 1.0 if np.array_equal(pred, gt) else 0.0
    dice = 2 * intersection / (np.sum(pred) + np.sum(gt))
    return dice

def compute_sensitivity(pred, gt):
    """Compute sensitivity (recall) = TP / (TP + FN)"""
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def compute_specificity(pred, gt):
    """Compute specificity = TN / (TN + FP)"""
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))
    if (tn + fp) == 0:
        return 0.0
    return tn / (tn + fp)

def compute_iou(pred, gt):
    """Compute Intersection over Union"""
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    if union == 0:
        return 1.0 if np.array_equal(pred, gt) else 0.0
    return intersection / union

def compute_hausdorff_distance(pred, gt):
    """Compute Hausdorff distance between binary masks"""
    try:
        from scipy.spatial.distance import directed_hausdorff
        
        # Get coordinates of True values
        pred_coords = np.argwhere(pred > 0)
        gt_coords = np.argwhere(gt > 0)
        
        if len(pred_coords) == 0 or len(gt_coords) == 0:
            if np.array_equal(pred, gt):
                return 0.0
            else:
                return np.inf
        
        # Compute directed Hausdorff distances
        hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
        hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
        
        return max(hd1, hd2)
    except:
        return np.nan

def compute_bootstrap_ci(values, num_bootstraps=2000, alpha=0.95, seed=42):
    """
    Computes empirical 95% Confidence Intervals using bootstrapping.
    Handles non-normal clinical metric distributions reliably.
    """
    vals = np.array(values)
    # Filter out any lingering infinite values or NaNs safely
    vals = vals[np.isfinite(vals)]
    n = len(vals)
    
    if n == 0:
        return np.nan, np.nan
        
    rng = np.random.default_rng(seed)
    boot_means = []
    
    # Resample with replacement num_bootstraps times
    for _ in range(num_bootstraps):
        boot_sample = rng.choice(vals, size=n, replace=True)
        boot_means.append(np.mean(boot_sample))
        
    # Isolate lower and upper empirical percentile boundaries
    low_percentile = ((1.0 - alpha) / 2.0) * 100
    high_percentile = (alpha + ((1.0 - alpha) / 2.0)) * 100
    
    ci_lower = np.percentile(boot_means, low_percentile)
    ci_upper = np.percentile(boot_means, high_percentile)
    
    return float(ci_lower), float(ci_upper)

def run_inference_direct(model_folder, test_images_dir, output_pred_dir, resume=True):
    """Run inference using nnUNetPredictor directly with resume capability"""
    print(f"\n{'='*70}")
    print("Running inference using nnUNetPredictor...")
    print(f"{'='*70}")
    print(f"Model folder: {model_folder}")
    print(f"Test images: {test_images_dir}")
    print(f"Output dir: {output_pred_dir}")
    
    os.makedirs(output_pred_dir, exist_ok=True)
    
    # Check which cases have already been predicted
    already_predicted = set()
    if resume and os.path.exists(output_pred_dir):
        already_predicted = set([f.stem.replace('.nii', '') for f in Path(output_pred_dir).glob("*.nii.gz")])
        if already_predicted:
            print(f"\nResume mode: Found {len(already_predicted)} already predicted cases")
            print(f"Examples: {sorted(list(already_predicted))[:5]}")
    
    # Initialize predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    
    # Initialize from trained model
    print("\nInitializing from trained model...")
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=[0],
        checkpoint_name='checkpoint_best.pth'
    )
    
    # Get test images and filter out already predicted ones
    test_images = sorted([f for f in Path(test_images_dir).glob("*_0000.nii.gz")])
    remaining_images = [f for f in test_images if f.stem.replace('_0000', '') not in already_predicted]
    # remaining_images = [] #### only when eval summary
    if len(remaining_images) == 0:
        print("\n✓ All cases already predicted!")
        return output_pred_dir
    
    print(f"\nCases to predict: {len(remaining_images)}/{len(test_images)} (skipping {len(already_predicted)} already done)")
    
    # Run predictions using predict_from_files
    print("\nRunning predictions...")
    
    # Create temporary directory for remaining images
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for img_file in remaining_images:
            # Create symlinks to remaining images
            (tmpdir / img_file.name).symlink_to(img_file)
        
        predictor.predict_from_files(
            list_of_lists_or_source_folder=str(tmpdir),
            output_folder_or_list_of_truncated_output_files=str(output_pred_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=8,
            num_processes_segmentation_export=8
        )
    
    print("\n✓ Inference complete!")
    return output_pred_dir

def evaluate_predictions(pred_dir, gt_dir):
    """Evaluate predictions against ground truth"""
    print(f"\n{'='*70}")
    print("Evaluating predictions...")
    print(f"{'='*70}\n")
    
    pred_files = sorted(Path(pred_dir).glob("*.nii.gz"))
    
    results = defaultdict(list)
    case_results = {}
    
    for pred_file in pred_files:
        case_name = pred_file.stem.replace(".nii", "")
        gt_file = Path(gt_dir) / f"{case_name}.nii.gz"
        
        if not gt_file.exists():
            print(f"⚠ GT not found for {case_name}, skipping")
            continue
        
        # Load images
        pred_img = nib.load(pred_file)
        gt_img = nib.load(gt_file)
        
        pred = (pred_img.get_fdata() > 0.5).astype(np.uint8)
        gt = (gt_img.get_fdata() > 0).astype(np.uint8)
        
        # Compute metrics
        dice = compute_dice(pred, gt)
        sensitivity = compute_sensitivity(pred, gt)
        specificity = compute_specificity(pred, gt)
        iou = compute_iou(pred, gt)
        hd = compute_hausdorff_distance(pred, gt)
        
        # Volume metrics
        pred_volume = np.sum(pred)
        gt_volume = np.sum(gt)
        volume_ratio = pred_volume / gt_volume if gt_volume > 0 else 0
        
        case_results[case_name] = {
            "DICE": float(dice),
            "Sensitivity": float(sensitivity),
            "Specificity": float(specificity),
            "IoU": float(iou),
            "Hausdorff_Distance": float(hd) if not np.isnan(hd) else None,
            "Pred_Volume": int(pred_volume),
            "GT_Volume": int(gt_volume),
            "Volume_Ratio": float(volume_ratio)
        }
        
        results['DICE'].append(dice)
        results['Sensitivity'].append(sensitivity)
        results['Specificity'].append(specificity)
        results['IoU'].append(iou)
        if not np.isnan(hd):
            results['Hausdorff_Distance'].append(hd)
        
        print(f"{case_name:30s} | DICE: {dice:.4f} | Sens: {sensitivity:.4f} | Spec: {specificity:.4f} | IoU: {iou:.4f}")
    
    # Aggregate metrics
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}\n")
    
    summary = {}
    for metric, values in results.items():
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            ci_lower, ci_upper = compute_bootstrap_ci(values, num_bootstraps=2000)
            
            summary[metric] = {
                "mean": float(mean),
                "std": float(std),
                "min": float(min_val),
                "max": float(max_val),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n": len(values)
            }
            
            print(f"{metric:20s}: {mean:.4f} [95% CI: {ci_lower:.4f} - {ci_upper:.4f}] | ± {std:.4f} (min: {min_val:.4f}, max: {max_val:.4f})")
    
    return summary, case_results

def main():
    # Paths
    model_folder = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results/Dataset025_PuFiRS25/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    test_images_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/imagesTs")
    test_labels_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTs")
    output_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/test_predictions_baseline_curriculum")
    
    # Verify test set exists
    test_cases = list(test_images_dir.glob("*.nii.gz"))
    print(f"\nFound {len(test_cases)} test cases in {test_images_dir}")
    
    # Run inference with resume capability
    pred_dir = run_inference_direct(model_folder, test_images_dir, output_dir, resume=True)
    
    # Evaluate
    summary, case_results = evaluate_predictions(pred_dir, test_labels_dir)
    
    # Save results
    results_file = output_dir / "evaluation_results.json"
    os.makedirs(output_dir, exist_ok=True)
    
    full_results = {
        "summary": summary,
        "case_results": case_results,
        "test_set": str(test_images_dir),
        "gt_set": str(test_labels_dir),
        "model_folder": str(model_folder),
        "dataset_id": 25
    }
    
    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    return summary

if __name__ == "__main__":
    main()
