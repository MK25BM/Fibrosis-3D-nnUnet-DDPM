#!/bin/bash
##############################################################################
# Augmented Segmentation Experiment Pipeline
# 
# Quick-start script to run full augmentation ablation study:
# 1. Generate synthetic data
# 2. Preprocess augmented datasets
# 3. Train models on each ratio
# 4. Evaluate and compare DICE
##############################################################################

set -e

PROJECT_ROOT="/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM"
CONDA_ENV="ild_ddpm"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

##############################################################################
# SETUP
##############################################################################

log_info "Setting up augmented segmentation pipeline..."

cd "$PROJECT_ROOT"

# Activate conda environment
source /home/u6dm/mk25bm.u6dm/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# Set nnUNet environment variables
export nnUNet_raw="$PROJECT_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$PROJECT_ROOT/nnUNet_preprocessed"
export nnUNet_results="$PROJECT_ROOT/nnUNet_results"

log_info "Environment:"
log_info "  Project: $PROJECT_ROOT"
log_info "  Conda: $CONDA_ENV"
log_info "  nnUNet_raw: $nnUNet_raw"
log_info "  nnUNet_preprocessed: $nnUNet_preprocessed"
log_info "  nnUNet_results: $nnUNet_results"

##############################################################################
# STAGE 1: GENERATE SYNTHETIC DATA
##############################################################################

log_info ""
log_info "STAGE 1: Generating synthetic training data..."
log_info "=========================================================="

python augmented_segmentation_pipeline.py

if [ $? -eq 0 ]; then
    log_info "✓ Synthetic data generation complete"
else
    log_error "Synthetic data generation failed"
    exit 1
fi

##############################################################################
# STAGE 2: PREPROCESS AUGMENTED DATASETS
##############################################################################

log_info ""
log_info "STAGE 2: Preprocessing augmented datasets..."
log_info "=========================================================="

RATIOS=(0 50 100 200)

for RATIO in "${RATIOS[@]}"; do
    log_info "Preprocessing Dataset025_PuFiRS25_synth${RATIO}..."
    
    # In practice, this would involve:
    # 1. Updating nnUNet_raw to point to augmented dataset
    # 2. Running nnUNetv2_plan_and_preprocess
    # For now, we log the command
    
    echo "  nnUNetv2_plan_and_preprocess -d 25 -c 3d_fullres --clean"
done

log_warn "Note: Preprocessing step requires manual execution per augmented dataset"
log_warn "See AUGMENTATION_PIPELINE.md for detailed preprocessing instructions"

##############################################################################
# STAGE 3: TRAINING
##############################################################################

log_info ""
log_info "STAGE 3: Training augmented models..."
log_info "=========================================================="

log_warn "Training requires GPU and significant compute time (24+ hours)"
log_warn "Consider submitting as SLURM jobs for each ratio"
log_warn ""
log_warn "Quick training script template:"
echo ""
cat << 'EOF'
#!/bin/bash
for RATIO in 0 50 100 200; do
    echo "Training with ratio ${RATIO}%..."
    
    # Set dataset path based on ratio
    export nnUNet_raw="$PROJECT_ROOT/augmented_data/Dataset025_PuFiRS25_synth${RATIO}"
    
    # Train
    nnUNetv2_train 25 3d_fullres 0 \
        -tr nnUNetTrainer \
        -p nnUNetResEncUNetLPlans \
        -device cuda \
        -num_epochs 1000
    
    echo "✓ Training complete for ratio ${RATIO}%"
done
EOF
echo ""

##############################################################################
# STAGE 4: EVALUATION
##############################################################################

log_info ""
log_info "STAGE 4: Evaluation and ablation study..."
log_info "=========================================================="

log_info "Running evaluation pipeline..."

python train_augmented_models.py

if [ $? -eq 0 ]; then
    log_info "✓ Evaluation complete"
    log_info "Results saved to: augmentation_ablation_results.csv"
else
    log_error "Evaluation failed"
    exit 1
fi

##############################################################################
# SUMMARY
##############################################################################

log_info ""
log_info "=========================================================="
log_info "Pipeline Complete!"
log_info "=========================================================="
log_info ""
log_info "Results:"
log_info "  CSV Report: $PROJECT_ROOT/augmentation_ablation_results.csv"
log_info ""
log_info "Next Steps:"
log_info "  1. Review augmentation_ablation_results.csv"
log_info "  2. Identify best synthetic-to-real ratio"
log_info "  3. Use best model for production predictions"
log_info ""
log_info "For detailed instructions, see AUGMENTATION_PIPELINE.md"

