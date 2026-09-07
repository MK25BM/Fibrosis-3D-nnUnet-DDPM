#!/bin/bash
#SBATCH --account=brics.u6dm
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --job-name=preprocess_augmented
#SBATCH --output=logs/preprocess_augmented_%j.log

set -e

# Activate conda environment
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# Set nnUNet environment variables
export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

# Set thread limits
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "Preprocessing Augmented Datasets"
echo "=========================================="
echo "Start time: $(date)"
echo ""

cd /projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM

# Preprocess each augmented dataset
echo "Preprocessing Dataset025_PuFiRS25_Aug0.5..."
nnUNetv2_plan_and_preprocess -d Dataset025_PuFiRS25_Aug0.5 -c 3d_fullres -np 8 --clean

echo ""
echo "Preprocessing Dataset025_PuFiRS25_Aug1.0..."
nnUNetv2_plan_and_preprocess -d Dataset025_PuFiRS25_Aug1.0 -c 3d_fullres -np 8 --clean

echo ""
echo "Preprocessing Dataset025_PuFiRS25_Aug2.0..."
nnUNetv2_plan_and_preprocess -d Dataset025_PuFiRS25_Aug2.0 -c 3d_fullres -np 8 --clean

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Next: Train augmented models with:"
echo "  nnUNetv2_train Dataset025_PuFiRS25_Aug0.5 3d_fullres 0"
echo "  nnUNetv2_train Dataset025_PuFiRS25_Aug1.0 3d_fullres 0"
echo "  nnUNetv2_train Dataset025_PuFiRS25_Aug2.0 3d_fullres 0"
