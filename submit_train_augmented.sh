#!/bin/bash
#SBATCH --account=brics.u6dm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --job-name=train_augmented
#SBATCH --output=logs/train_augmented_%j.log
#SBATCH --error=logs/train_augmented_%j.err
#SBATCH --open-mode=append

set -e

# Activate conda environment
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# Set thread limits (matching baseline config)
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

# Limit background workers to prevent RAM spikes
export nnUNet_n_proc_da=2    
export nnUNet_num_raw_images_background_loaders=2

# PyTorch memory management
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# Set nnUNet environment variables
export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

mkdir -p logs

echo "=========================================="
echo "Training Augmented Models"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM

echo "[1/3] Training Aug0.5 (70 real + 35 synthetic)..."
nnUNetv2_train Dataset026_PuFiRS25_Aug0.5 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer 2>&1 | tee logs/train_aug0.5.log

echo ""
echo "[2/3] Training Aug1.0 (70 real + 70 synthetic)..."
nnUNetv2_train Dataset027_PuFiRS25_Aug1.0 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer 2>&1 | tee logs/train_aug1.0.log

echo ""
echo "[3/3] Training Aug2.0 (70 real + 140 synthetic)..."
nnUNetv2_train Dataset028_PuFiRS25_Aug2.0 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer 2>&1 | tee logs/train_aug2.0.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Models saved to:"
echo "  Aug0.5: $nnUNet_results/Dataset026_PuFiRS25_Aug0.5/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/"
echo "  Aug1.0: $nnUNet_results/Dataset027_PuFiRS25_Aug1.0/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/"
echo "  Aug2.0: $nnUNet_results/Dataset028_PuFiRS25_Aug2.0/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/"
echo ""
echo "Next: Evaluate all models and compare DICE:"
echo "  python compare_augmented_models.py"
