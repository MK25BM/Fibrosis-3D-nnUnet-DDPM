#!/bin/bash
#SBATCH --job-name=nnunet_preprocess
#SBATCH --partition=workq       
#SBATCH --account=brics.u6dm           # Links to your project group allocation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        
#SBATCH --mem=32G                
#SBATCH --time=00:45:00          
#SBATCH --output=nnunet_preproc_%j.log

# 1. Clear any system-inherited modules
module purge

# 2. Set your mandatory nnU-Net folder paths
export nnUNet_raw="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw"
export nnUNet_preprocessed="/projects/u6dm/mk25bm.u6dm/nnUNet_preprocessed"
export nnUNet_results="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_results"

# 3. Explicitly activate your custom conda environment
source ~/miniforge3/bin/activate ild_ddpm

# 4. Restrict internal math thread pools to match our 8 CPUs
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 5. Run preprocessing across 4 concurrent file extraction workers
echo "Starting nnU-Net dataset planning and preprocessing..."
nnUNetv2_plan_and_preprocess -d 25 -c 3d --clean -npfp 4 -np 4
echo "Preprocessing complete!"
