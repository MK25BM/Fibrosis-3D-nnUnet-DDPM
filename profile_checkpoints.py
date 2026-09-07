import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy

# Set thread limits safely for background compute nodes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# --- Fixed Configuration Paths ---
RUN_DIRS = [
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-17T205411", # 0: model-1 to 3
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-18T065004", # 1: model-4 to 6
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-18T112817", # 2: model-7 to 9
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-18T205856", # 3: model-10 to 13
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-19T142525", # 4: model-14 to 17
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-20T181210", # 5: model-18 to 20
    "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/Lung-DDPM-PLUS/runs/Lung-DDPM+/Lung-DDPM+_fold_PuFiRS25_26-08-21T131022"  # 6: model-21 to 22
]
CT_FOLDER = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/imagesTr"
MASK_FOLDER = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset025_PuFiRS25/labelsTr"
SYNTHETIC_BASE = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/checkpoint_outputs"
TARGET_PATIENT_ID = "PuFiRS25_00034"

def extract_loss_from_specific_folder(run_folder, milestone_step):
    """Extract scalar loss from a single target directory log timeline."""
    event_files = sorted(glob.glob(os.path.join(run_folder, "events.out.tfevents.*")))
    if not event_files: return None
    try:
        ea = event_accumulator.EventAccumulator(event_files[-1], size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags().get('scalars', [])
        loss_tag = [tag for tag in tags if 'loss' in tag.lower()]
        if loss_tag:
            for events in ea.Scalars(loss_tag[-1]):
                if events.step == milestone_step:
                    return events.value
    except Exception: pass
    return None

def main():
    print(f"🎯 Initiating Synchronized Checkpoint Profiler ({TARGET_PATIENT_ID})...")
    print("──────────────────────────────────────────────────────────────────────────")

    real_ct_path = os.path.join(CT_FOLDER, f"{TARGET_PATIENT_ID}_0000.nii.gz")
    if not os.path.exists(real_ct_path): real_ct_path = os.path.join(CT_FOLDER, f"{TARGET_PATIENT_ID}.nii.gz")
    real_volume = nib.load(real_ct_path).get_fdata()
    real_volume = np.clip(real_volume, -1024, 300)
    
    ref_mask = nib.load(os.path.join(MASK_FOLDER, f"{TARGET_PATIENT_ID}.nii.gz")).get_fdata()
    disease_indices = ref_mask >= 1
    
    gt_voxels = real_volume[disease_indices]
    min_hu, max_hu = -1024.0, 300.0
    gt_norm = np.clip((gt_voxels - min_hu) / (max_hu - min_hu), 0.0, 1.0)
    
    records = []
    
    for stage_idx in range(1, 24):
        sample_path = os.path.join(SYNTHETIC_BASE, f"stage_{stage_idx}", f"aug_synthetic_{TARGET_PATIENT_ID}.nii.gz")
        if not os.path.exists(sample_path): continue
            
        # ✅ HARDCODED DIRECT FIX: Map each stage index to its true directory index
        if stage_idx <= 3:    f_idx = 0
        elif stage_idx <= 6:  f_idx = 1
        elif stage_idx <= 9:  f_idx = 2
        elif stage_idx <= 13: f_idx = 3
        elif stage_idx <= 17: f_idx = 4
        elif stage_idx <= 20: f_idx = 5 
        else: f_idx = 6  # Default fallback for any unexpected stage index
            
        target_step = stage_idx * 500
        tb_loss = extract_loss_from_specific_folder(RUN_DIRS[f_idx], target_step)
        
        try:
            synthetic_volume = nib.load(sample_path).get_fdata()
            synthetic_volume = np.clip(synthetic_volume, -1024, 300)
            syn_voxels = synthetic_volume[disease_indices]
            syn_norm = np.clip((syn_voxels - min_hu) / (max_hu - min_hu), 0.0, 1.0)
            
            calc_ssim = np.corrcoef(gt_norm, syn_norm)[0, 1]
            mse_val = np.mean((gt_norm - syn_norm) ** 2)
            calc_rmse = np.sqrt(mse_val)
            calc_psnr = psnr(gt_norm, syn_norm, data_range=1.0) if mse_val > 0 else 99.0
            patch_entropy = shannon_entropy(syn_voxels)
            
            records.append({
                "Checkpoint": f"Stage-{stage_idx}",
                "Step": target_step,
                "Run Loss": round(tb_loss, 6) if tb_loss else "N/A",
                "Mask SSIM": round(calc_ssim, 4),
                "Mask PSNR": round(calc_psnr, 2),
                "Mask RMSE": round(calc_rmse, 4),
                "Mask Entropy": round(patch_entropy, 4)
            })
        except Exception as e: print(f"⚠️ Error parsing stage_{stage_idx}: {e}")
            
    df = pd.DataFrame(records)
    print(f"\n📊 UNIFIED ARCHITECTURE RE-COMPOSITION REPORT ({TARGET_PATIENT_ID}):")
    if not df.empty: print(df.to_string(index=False))
    print("──────────────────────────────────────────────────────────────────────────")

if __name__ == "__main__": main()
