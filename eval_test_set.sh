#!/bin/bash
#SBATCH --account=brics.u6dm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_pufirs_baseline
#SBATCH --output=logs/eval_pufirs_baseline_%j.log

set -e

# Activate conda environment
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# Set nnUNet environment variables
export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

# Set thread limits
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Evaluating nnUNet baseline on test set"
echo "=========================================="
echo "Test images: $nnUNet_raw/Dataset025_PuFiRS25/imagesTs"
echo "Test labels: $nnUNet_raw/Dataset025_PuFiRS25/labelsTs"
echo ""

cd /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM

# Run evaluation
python evaluate_test_set.py

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "=========================================="
echo "Results saved to: evaluation/test_predictions_baseline/evaluation_results.json"
