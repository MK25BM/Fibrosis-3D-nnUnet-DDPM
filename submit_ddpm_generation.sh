#!/bin/bash
#SBATCH --account=brics.u6dm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=ddpm_synthetic_gen
#SBATCH --output=logs/ddpm_synthetic_gen_%j.log

set -e

# Activate conda environment
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# Set thread limits to reduce memory pressure
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Generating 140 DDPM Synthetic Images"
echo "=========================================="
echo "Batch 1: 10 cases × 3 perturbations = 30 images"
echo "Batch 2: 25 cases × 4 perturbations = 110 images"
echo "Total: 35 cases, 140 synthetic images"
echo ""
echo "Start time: $(date)"
echo "=========================================="
echo ""

cd /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM

# Run full generation (all 35 cases, 4 perturbations each = 140 total)
# Note: We'll update the script to use all 35 cases and 4 perturbations
python generate_augmentation_with_blending.py

echo ""
echo "=========================================="
echo "Generation complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Results saved to:"
echo "  CT: /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2/"
echo "  Masks: /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_masks_v2/"
