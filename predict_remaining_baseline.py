#!/usr/bin/env python3
"""
Predict remaining 4 test cases for baseline model to achieve 100% coverage.
"""

import os
import sys
import torch
from pathlib import Path

# Add nnUNet to path
sys.path.insert(0, '/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet')

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Configuration
DATASET_NAME = "Dataset025_PuFiRS25"
PROJECT_DIR = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"

# Set environment variables if not already set
os.environ.setdefault("nnUNet_raw", f"{PROJECT_DIR}/nnUNet_raw")
os.environ.setdefault("nnUNet_results", f"{PROJECT_DIR}/nnUNet_results")
os.environ.setdefault("nnUNet_preprocessed", f"{PROJECT_DIR}/nnUNet_preprocessed")

TEST_IMAGES_DIR = Path(f"{PROJECT_DIR}/nnUNet_raw/{DATASET_NAME}/imagesTs")
PREDICTIONS_DIR = Path(f"{PROJECT_DIR}/evaluation/test_predictions_baseline")
MODEL_FOLDER = Path(f"{PROJECT_DIR}/nnUNet_results/{DATASET_NAME}/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")

REMAINING_CASES = ["PuFiRS25_00107", "PuFiRS25_00112", "PuFiRS25_00117", "PuFiRS25_00118"]

def predict_case(predictor, case_id):
    """Predict segmentation for a single test case."""
    
    image_file = TEST_IMAGES_DIR / f"{case_id}_0000.nii.gz"
    output_file = PREDICTIONS_DIR / f"{case_id}.nii.gz"
    
    if not image_file.exists():
        print(f"  ERROR: Image file not found: {image_file}")
        return False
    
    if output_file.exists():
        print(f"  ✓ Already predicted (skipping)")
        return True
    
    try:
        # Create a temp directory for this case's prediction
        temp_case_dir = PREDICTIONS_DIR / f"temp_{case_id}"
        temp_case_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the image to temp directory with standard nnUNet naming
        import shutil
        temp_image_file = temp_case_dir / f"{case_id}_0000.nii.gz"
        shutil.copy(str(image_file), str(temp_image_file))
        
        # Predict using the correct API (directory -> directory)
        predictor.predict_from_files(
            str(temp_case_dir),
            str(PREDICTIONS_DIR)
        )
        
        # Rename output file to remove _0000 suffix if needed
        temp_output = PREDICTIONS_DIR / f"{case_id}_0000.nii.gz"
        if temp_output.exists() and not output_file.exists():
            temp_output.rename(output_file)
        
        # Clean up temp directory
        shutil.rmtree(str(temp_case_dir), ignore_errors=True)
        
        if output_file.exists():
            print(f"  ✓ Prediction saved")
            return True
        else:
            print(f"  ERROR: Output file not created")
            return False
    
    except Exception as e:
        print(f"  ERROR during prediction: {str(e)}")
        return False

def main():
    print("\n" + "="*70)
    print("PREDICTING REMAINING 4 BASELINE TEST CASES")
    print("="*70)
    print(f"Cases to predict: {', '.join(REMAINING_CASES)}")
    print()
    
    # Verify paths
    print(f"Model folder: {MODEL_FOLDER}")
    if not MODEL_FOLDER.exists():
        print(f"ERROR: Model folder not found!")
        sys.exit(1)
    
    print(f"Test images: {TEST_IMAGES_DIR}")
    print(f"Output dir: {PREDICTIONS_DIR}")
    print()
    
    # Initialize predictor
    print("Initializing nnUNetPredictor...")
    try:
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
        print("Loading model checkpoint...")
        predictor.initialize_from_trained_model_folder(
            MODEL_FOLDER,
            use_folds=[0],
            checkpoint_name='checkpoint_best.pth'
        )
        
        print("✓ Model loaded successfully\n")
        
        # Predict remaining cases
        print("="*70)
        print("PREDICTING CASES")
        print("="*70 + "\n")
        
        success_count = 0
        for i, case_id in enumerate(REMAINING_CASES, 1):
            print(f"[{i}/4] {case_id}...", end=" ")
            if predict_case(predictor, case_id):
                success_count += 1
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print()
        print("="*70)
        print(f"✓ Completed {success_count}/{len(REMAINING_CASES)} predictions")
        print("="*70)
        
        # Verify final count
        completed = len(list(PREDICTIONS_DIR.glob("*.nii.gz"))) - 1  # -1 for dataset.json, etc
        print(f"\nFinal baseline prediction count: {completed}/25")
        
        return success_count == len(REMAINING_CASES)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
