#!/usr/bin/env python3
"""
Compare baseline and 3 augmented models on the same test set.
Evaluates all 4 models (baseline + Aug0.5, Aug1.0, Aug2.0) on 25 holdout test cases.
Generates comparison report with DICE improvements.
"""

import os
import json
import numpy as np
import nibabel as nib
import torch
from pathlib import Path
from collections import defaultdict
from scipy.spatial.distance import directed_hausdorff
import sys

sys.path.insert(0, '/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Metrics computation
def compute_dice(pred, gt):
    intersection = np.sum(pred * gt)
    dice = 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)
    return float(dice)

def compute_sensitivity(pred, gt):
    tp = np.sum(pred * gt)
    fn = np.sum((1 - pred) * gt)
    return float(tp / (tp + fn + 1e-8))

def compute_specificity(pred, gt):
    tn = np.sum((1 - pred) * (1 - gt))
    fp = np.sum(pred * (1 - gt))
    return float(tn / (tn + fp + 1e-8))

def compute_iou(pred, gt):
    intersection = np.sum(pred * gt)
    union = np.sum(np.logical_or(pred, gt))
    return float(intersection / (union + 1e-8))

def evaluate_model(model_name, model_folder, test_images_dir, test_labels_dir, output_pred_dir):
    """Evaluate a single model on test set"""
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*70}")
    print(f"Model: {model_folder}")
    
    os.makedirs(output_pred_dir, exist_ok=True)
    
    # Initialize predictor
    """ print("Initializing predictor...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        allow_tqdm=True
    )
    
    # Initialize from trained model
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=[0],
        checkpoint_name='checkpoint_best.pth'
    )
    
    # Check for already predicted cases
    already_predicted = set([f.stem.replace('.nii', '') for f in Path(output_pred_dir).glob("*.nii.gz")])
    test_images = sorted([f for f in Path(test_images_dir).glob("*_0000.nii.gz")])
    remaining_images = [f for f in test_images if f.stem.replace('_0000', '') not in already_predicted]

    if remaining_images:
        print(f"Predicting on {len(remaining_images)} remaining test cases...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for img_file in remaining_images:
                (tmpdir / img_file.name).symlink_to(img_file)
            
            predictor.predict_from_files(
                list_of_lists_or_source_folder=str(tmpdir),
                output_folder_or_list_of_truncated_output_files=str(output_pred_dir),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=8,
                num_processes_segmentation_export=8
            )
    else:
        print(f"All {len(already_predicted)} test cases already predicted")
    """
    print(f"All test cases already predicted")
    # Evaluate predictions
    print(f"\nEvaluating metrics...")
    pred_files = sorted(Path(output_pred_dir).glob("*.nii.gz"))
    
    results = defaultdict(list)
    case_results = {}
    
    for pred_file in pred_files:
        case_name = pred_file.stem.replace(".nii", "")
        gt_file = Path(test_labels_dir) / f"{case_name}.nii.gz"
        
        if not gt_file.exists():
            continue
        
        # Load and threshold
        pred = (nib.load(pred_file).get_fdata() > 0.5).astype(np.uint8)
        gt = (nib.load(gt_file).get_fdata() > 0).astype(np.uint8)
        
        # Compute metrics
        dice = compute_dice(pred, gt)
        sensitivity = compute_sensitivity(pred, gt)
        specificity = compute_specificity(pred, gt)
        iou = compute_iou(pred, gt)
        
        case_results[case_name] = {
            "DICE": dice,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "IoU": iou
        }
        
        results['DICE'].append(dice)
        results['Sensitivity'].append(sensitivity)
        results['Specificity'].append(specificity)
        results['IoU'].append(iou)
    
    # Summary statistics
    summary = {}
    for metric, values in results.items():
        if values:
            summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values)
            }
    
    print(f"\nResults ({summary['DICE']['n']} cases evaluated):")
    print(f"  DICE:        {summary['DICE']['mean']:.4f} ± {summary['DICE']['std']:.4f}")
    print(f"  Sensitivity: {summary['Sensitivity']['mean']:.4f} ± {summary['Sensitivity']['std']:.4f}")
    print(f"  Specificity: {summary['Specificity']['mean']:.4f} ± {summary['Specificity']['std']:.4f}")
    print(f"  IoU:         {summary['IoU']['mean']:.4f} ± {summary['IoU']['std']:.4f}")
    
    torch.cuda.empty_cache()
    
    return summary, case_results

def main():
    print("\n" + "="*70)
    print("AUGMENTED MODEL COMPARISON STUDY")
    print("="*70)
    
    # Paths
    test_images_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/imagesTs")
    test_labels_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTs")
    results_dir = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation")
    
    base_results_path = Path("/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results")
    
    # Models to compare
    models = [
        ("Baseline (No Augmentation)", 
         base_results_path / "Dataset025_PuFiRS25/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
         results_dir / "test_predictions_baseline"),
        
        ("Aug0.5 (70 real + 35 synthetic)",
         base_results_path / "Dataset026_PuFiRS25_Aug0.5/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
         results_dir / "test_predictions_Aug0.5"),
        
        ("Aug1.0 (70 real + 70 synthetic)",
         base_results_path / "Dataset027_PuFiRS25_Aug1.0/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
         results_dir / "test_predictions_Aug1.0"),
        
        ("Aug2.0 (70 real + 140 synthetic)",
         base_results_path / "Dataset028_PuFiRS25_Aug2.0/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres",
         results_dir / "test_predictions_Aug2.0"),
    ]
    
    # Evaluate all models
    all_results = {}
    for model_name, model_folder, pred_dir in models:
        if not model_folder.exists():
            print(f"\n⚠️  Model not found: {model_folder}")
            continue
        
        summary, case_results = evaluate_model(model_name, str(model_folder), test_images_dir, test_labels_dir, pred_dir)
        all_results[model_name] = (summary, case_results)
    
    # Generate comparison report
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    comparison_data = []
    for model_name, (summary, _) in all_results.items():
        dice_mean = summary['DICE']['mean']
        dice_std = summary['DICE']['std']
        comparison_data.append({
            'Model': model_name,
            'DICE_Mean': dice_mean,
            'DICE_Std': dice_std,
            'Sensitivity': summary['Sensitivity']['mean'],
            'Specificity': summary['Specificity']['mean'],
            'IoU': summary['IoU']['mean']
        })
        print(f"{model_name:45s} | DICE: {dice_mean:.4f} ± {dice_std:.4f}")
    
    # Calculate DICE improvements
    if len(comparison_data) > 1:
        baseline_dice = comparison_data[0]['DICE_Mean']
        print(f"\nDICE Improvement vs Baseline:")
        for row in comparison_data[1:]:
            improvement = (row['DICE_Mean'] - baseline_dice) * 100
            print(f"  {row['Model']:43s}: {improvement:+.2f}%")
    
    # Save results
    results_json = results_dir / "augmented_models_comparison.json"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_json, 'w') as f:
        json.dump({
            'comparison': comparison_data,
            'detailed_results': {k: v[1] for k, v in all_results.items()}
        }, f, indent=2)
    
    print(f"\n✓ Comparison results saved to: {results_json}")
    
    # Save CSV for easy viewing
    csv_path = results_dir / "augmented_models_comparison.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
        writer.writeheader()
        writer.writerows(comparison_data)
    
    print(f"✓ CSV report saved to: {csv_path}")
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
