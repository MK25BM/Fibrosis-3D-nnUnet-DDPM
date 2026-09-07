#!/bin/bash
#SBATCH --job-name=class_sweep
#SBATCH --partition=workq
#SBATCH --account=brics.u6dm
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sweep_%j.log

export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

source ~/miniforge3/bin/activate ild_ddpm

# Run the validation tool over your best historical checkpoint state
echo "Evaluating per-class performance for best checkpoint..."
nnUNetv2_train 25 3d_fullres 0 -p nnUNetResEncUNetLPlans --val --val_best
