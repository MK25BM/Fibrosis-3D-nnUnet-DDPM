#!/bin/bash
################################################################################
# Interactive Augmentation Pipeline Runner
# Run this with: srun --account=brics.u6dm --partition=workq --reservation=interactive \
#                     --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 \
#                     --mem=45G --time=04:00:00 --pty bash
# Then inside the interactive session: bash run_augmentation_interactive.sh
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Augmented Segmentation Pipeline${NC}"
echo -e "${BLUE}Interactive Mode${NC}"
echo -e "${BLUE}========================================${NC}"

WORK_DIR="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
cd "$WORK_DIR"

# Setup conda environment
echo -e "\n${YELLOW}Setting up environment...${NC}"
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate ild_ddpm

# Verify GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || echo "GPU check skipped"

# Verify key files exist
echo -e "\n${YELLOW}Verifying required files...${NC}"
files_to_check=(
    "nnUNet_results/Dataset025_PuFiRS25/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_best.pth"
    "Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-21T131022/model-23.pt"
    "nnUNet_raw/Dataset025_PuFiRS25/imagesTr"
    "nnUNet_raw/Dataset025_PuFiRS25/labelsTr"
)

for file in "${files_to_check[@]}"; do
    if [ -e "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗ MISSING${NC}: $file"
        exit 1
    fi
done

# Setup environment variables
export nnUNet_raw="$WORK_DIR/nnUNet_raw"
export nnUNet_preprocessed="$WORK_DIR/nnUNet_preprocessed"
export nnUNet_results="$WORK_DIR/nnUNet_results"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo -e "\n${GREEN}Environment variables set:${NC}"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"

# Phase 1: Generate synthetic images from perturbed masks
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}PHASE 1: Synthetic Image Generation${NC}"
echo -e "${BLUE}========================================${NC}"

if [ ! -d "augmented_data/synthetic_images" ]; then
    echo -e "${YELLOW}Generating synthetic images and augmented datasets...${NC}"
    python augmented_segmentation_pipeline_simple.py
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Augmentation pipeline complete${NC}"
    else
        echo -e "${RED}✗ Failed to run augmentation pipeline${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Augmented data already exists, skipping generation${NC}"
    echo "  To regenerate, run: rm -rf augmented_data/"
fi

# All augmentation phases are handled by augmented_segmentation_pipeline_simple.py
# which creates datasets with different synthetic-to-real ratios

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Augmentation pipeline complete!${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "\n${YELLOW}Augmented datasets created:${NC}"
for ratio in 0.0 0.5 1.0 2.0; do
    dataset_name="Dataset025_PuFiRS25_Aug${ratio}"
    if [ -d "nnUNet_raw/$dataset_name" ]; then
        echo -e "${GREEN}✓${NC} $dataset_name"
        tr_count=$(ls nnUNet_raw/$dataset_name/imagesTr/*.nii.gz 2>/dev/null | wc -l)
        echo "    Training cases: $tr_count"
    fi
done

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "  1. Run nnUNet preprocessing for each dataset:"
echo "     for ratio in 0.0 0.5 1.0 2.0; do"
echo "       nnUNetv2_plan_and_preprocess -d 25 -c 3d --clean"
echo "     done"
echo ""
echo "  2. Train augmented models:"
echo "     sbatch train_pufirs_aug0.0.sh"
echo "     sbatch train_pufirs_aug1.0.sh"
echo ""
echo "  3. Evaluate on holdout test set:"
echo "     python evaluate_augmented_models.py"
echo ""
echo "  4. Analyze results:"
echo "     python analyze_augmentation_results.py"

echo -e "\n${YELLOW}Output directory:${NC}"
ls -lh augmented_data/ 2>/dev/null && echo "" || echo "augmented_data created"

echo -e "\n${GREEN}Done!${NC}\n"
