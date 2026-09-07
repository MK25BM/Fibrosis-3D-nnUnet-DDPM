#!/bin/bash
#SBATCH --job-name=pufirs_train
#SBATCH --partition=workq
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:40:00
#SBATCH --open-mode=append

# Training script for PuFiRS25 with nnUNet V2
# Usage: sbatch train_pufirs.sh
# Or:    ./train_pufirs.sh (for local execution)

set -e

# ─── Clean environment ───
module purge

source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# ─── Configuration Block ───
DATASET_ID=32
CONFIG="3d_fullres"              # Note: Changed from '3d' to '3d_fullres' for nnU-Net v2 compatibility
FOLD=0
PLANNER_NAME="nnUNetResEncUNetLPlans"
TRAINER_CLASS="nnUNetTrainer"    # Default trainer class for nnU-Net v2
CUDA_VISIBLE_DEVICES=0           # Sets the targeted GPU device ID

# ─── Prevent system-level library thread bloat ───
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# ─── Limit background workers to prevent RAM spikes ───
export nnUNet_n_proc_da=2    
export nnUNet_num_raw_images_background_loaders=2

# ─── PyTorch memory management ───
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# ─── nnU-Net environment variables ───
export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

mkdir -p logs


# Launch the training command using the configured variables
echo "--------------------------------------------------------"
echo "Starting nnU-Net Training Session..."
echo "Dataset ID:    $DATASET_ID"
echo "Configuration: $CONFIG"
echo "Fold:          $FOLD"
echo "Planner:       $PLANNER_NAME"
echo "Trainer Class: $TRAINER_CLASS"
echo "Target GPU:    $CUDA_VISIBLE_DEVICES"
echo "--------------------------------------------------------"

nnUNetv2_train $DATASET_ID $CONFIG $FOLD -p $PLANNER_NAME -tr $TRAINER_CLASS --c 

echo "Training session complete!"