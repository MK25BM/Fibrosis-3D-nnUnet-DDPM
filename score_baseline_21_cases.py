#!/usr/bin/env python3
"""
Score the 21 baseline predictions we have completed.
Does NOT attempt re-prediction.
Just computes metrics on existing files.
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import distance_transform_edt
import nibabel as nib
from collections import defaultdict
import sys

# Paths
PRED_DIR = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/test_predictions_baseline")
GT_DIR = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/PuFiRS25/labelsTs")
RESULTS_JSON = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/baseline_metrics_21cases.json")

def compute_dice(pred, gt):
    """Compute Dice coefficient."""
    intersection = np.sum(pred * gt)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)
    return float(dice)

def compute_sensitivity(pred, gt):
    """Compute sensitivity (true positive rate)."""
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    sensitivity = tp / (tp + fn + 1e-8)
    return float(sensitivity)

def compute_specificity(pred, gt):
    """Compute specificity (true negative rate)."""
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))
    specificity = tn / (tn + fp + 1e-8)
    return float(specificity)

def compute_iou(pred, gt):
    """Compute Intersection over Union."""
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    iou = intersection / (union + 1e-8)
    return float(iou)

def compute_hausdorff_distance(pred, gt):
    """Compute Hausdorff distance (95th percentile)."""
    if np.sum(pred) == 0 or np.sum(gt) == 0:
        return float('nan')
    
    try:
        # Distance from pred to gt
        pred_to_gt = distance_transform_edt(1 - gt)
        d1 = np.max(pred_to_gt[pred > 0]) if np.sum(pred) > 0 else 0
        
        # Distance from gt to pred
        gt_to_pred = distance_transform_edt(1 - pred)
        d2 = np.max(gt_to_pred[gt > 0]) if np.sum(gt) > 0 else 0
        
        hd = max(d1, d2)
        return float(hd)
    except:
        return float('nan')

def main():
    print("=" * 70)
    print("BASELINE EVALUATION - 21 Completed Test Cases")
    print("=" * 70)
    
    # Find prediction files
    pred_files = sorted(PRED_DIR.glob("*.nii.gz"))
    print(f"\nFound {len(pred_files)} prediction files to score")
    
    if len(pred_files) == 0:
        print("ERROR: No prediction files found!")
        sys.exit(1)
    
    case_results = {}
    metrics_dict = defaultdict(list)
    
    print("\nProcessing cases:")
    print("-" * 70)
    
    for i, pred_file in enumerate(pred_files, 1):
        case_name = pred_file.stem.replace(".nii", "")
        gt_file = GT_DIR / f"{case_name}.nii.gz"
        
        # Check if ground truth exists
        if not gt_file.exists():
            print(f"{i:2d}. {case_name}: SKIPPED (no ground truth)")
            continue
        
        try:
            # Load images
            pred_img = nib.load(pred_file)
            gt_img = nib.load(gt_file)
            
            pred_data = pred_img.get_fdata()
            gt_data = gt_img.get_fdata()
            
            # Threshold predictions at 0.5
            pred_binary = (pred_data > 0.5).astype(np.uint8)
            gt_binary = (gt_data > 0).astype(np.uint8)
            
            # Compute metrics
            dice = compute_dice(pred_binary, gt_binary)
            sensitivity = compute_sensitivity(pred_binary, gt_binary)
            specificity = compute_specificity(pred_binary, gt_binary)
            iou = compute_iou(pred_binary, gt_binary)
            hd = compute_hausdorff_distance(pred_binary, gt_binary)
            
            # Store results
            case_results[case_name] = {
                "dice": dice,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "iou": iou,
                "hausdorff_distance": hd
            }
            
            # Accumulate metrics
            metrics_dict["dice"].append(dice)
            metrics_dict["sensitivity"].append(sensitivity)
            metrics_dict["specificity"].append(specificity)
            metrics_dict["iou"].append(iou)
            metrics_dict["hausdorff_distance"].append(hd)
            
            # Print per-case result
            print(f"{i:2d}. {case_name}: DICE={dice:.4f}, Sens={sensitivity:.4f}, Spec={specificity:.4f}")
            
        except Exception as e:
            print(f"{i:2d}. {case_name}: ERROR - {str(e)}")
    
    # Compute summary statistics
    print("-" * 70)
    summary = {}
    for metric, values in metrics_dict.items():
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n_cases": len(values)
            }
    
    # Print summary
    print("\nSUMMARY STATISTICS (21 test cases):")
    print("-" * 70)
    for metric, stats in summary.items():
        print(f"{metric:20s}: {stats['mean']:.4f} ± {stats['std']:.4f} (min={stats['min']:.4f}, max={stats['max']:.4f})")
    
    # Save to JSON
    output = {
        "timestamp": str(Path(__file__).stat().st_mtime),
        "n_cases_evaluated": len(case_results),
        "case_results": case_results,
        "summary_statistics": summary
    }
    
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_JSON}")
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
