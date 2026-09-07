#!/usr/bin/env python3
"""
Quick evaluation of nnUNet predictions on test set.
Can be used independently if predictions are already generated.
"""
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import argparse

def compute_metrics(pred, gt):
    """Compute all metrics for a prediction"""
    # Binarize
    pred_bin = (pred > 0.5).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    
    # DICE
    intersection = np.sum(pred_bin * gt_bin)
    if (np.sum(pred_bin) + np.sum(gt_bin)) == 0:
        dice = 1.0 if np.array_equal(pred_bin, gt_bin) else 0.0
    else:
        dice = 2 * intersection / (np.sum(pred_bin) + np.sum(gt_bin))
    
    # Sensitivity
    tp = np.sum(pred_bin * gt_bin)
    fn = np.sum((1 - pred_bin) * gt_bin)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity
    tn = np.sum((1 - pred_bin) * (1 - gt_bin))
    fp = np.sum(pred_bin * (1 - gt_bin))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # IoU
    union = np.sum(np.logical_or(pred_bin, gt_bin))
    iou = intersection / union if union > 0 else (1.0 if np.array_equal(pred_bin, gt_bin) else 0.0)
    
    # Volume metrics
    pred_volume = np.sum(pred_bin)
    gt_volume = np.sum(gt_bin)
    volume_ratio = pred_volume / gt_volume if gt_volume > 0 else 0
    
    return {
        "DICE": float(dice),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "IoU": float(iou),
        "Pred_Volume": int(pred_volume),
        "GT_Volume": int(gt_volume),
        "Volume_Ratio": float(volume_ratio)
    }

def evaluate_folder(pred_dir, gt_dir, output_json=None):
    """Evaluate predictions in a folder"""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    
    results = defaultdict(list)
    case_results = {}
    
    print(f"\n{'='*80}")
    print(f"Evaluating {len(pred_files)} predictions")
    print(f"{'='*80}\n")
    print(f"{'Case':<30} | {'DICE':>8} | {'Sens':>8} | {'Spec':>8} | {'IoU':>8} | {'Vol Ratio':>10}")
    print(f"{'-'*80}")
    
    for pred_file in pred_files:
        case_name = pred_file.stem.replace(".nii", "")
        gt_file = gt_dir / f"{case_name}.nii.gz"
        
        if not gt_file.exists():
            print(f"⚠ GT not found for {case_name}")
            continue
        
        # Load and compute metrics
        pred = nib.load(pred_file).get_fdata()
        gt = nib.load(gt_file).get_fdata()
        
        metrics = compute_metrics(pred, gt)
        case_results[case_name] = metrics
        
        for key, val in metrics.items():
            if key != "Pred_Volume" and key != "GT_Volume":
                results[key].append(val)
        
        print(f"{case_name:<30} | {metrics['DICE']:>8.4f} | {metrics['Sensitivity']:>8.4f} | "
              f"{metrics['Specificity']:>8.4f} | {metrics['IoU']:>8.4f} | {metrics['Volume_Ratio']:>10.2f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    summary = {}
    for metric in sorted(results.keys()):
        values = results[metric]
        mean = np.mean(values)
        std = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        summary[metric] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min_val),
            "max": float(max_val),
            "n": len(values)
        }
        
        print(f"{metric:<20}: {mean:7.4f} ± {std:6.4f}  (min: {min_val:7.4f}, max: {max_val:7.4f})")
    
    # Save results
    if output_json:
        results_data = {
            "summary": summary,
            "case_results": case_results,
            "pred_dir": str(pred_dir),
            "gt_dir": str(gt_dir)
        }
        with open(output_json, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✓ Results saved to {output_json}")
    
    return summary, case_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate nnUNet predictions")
    parser.add_argument("--pred-dir", type=str, 
                       default="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/test_predictions_baseline",
                       help="Directory with predictions")
    parser.add_argument("--gt-dir", type=str,
                       default="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTs",
                       help="Directory with ground truth labels")
    parser.add_argument("--output-json", type=str,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    evaluate_folder(args.pred_dir, args.gt_dir, args.output_json)
