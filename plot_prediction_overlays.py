import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Configuration Paths ---
HRCT_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset028_PuFiRS25_Aug2.0/imagesTs/PuFiRS25_00112_0000.nii.gz"
if not os.path.exists(HRCT_PATH):
    HRCT_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset028_PuFiRS25_Aug2.0/imagesTs/PuFiRS25_00112.nii.gz"

GT_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/nnUNet_raw/Dataset028_PuFiRS25_Aug2.0/labelsTs/PuFiRS25_00112.nii.gz"
PRED_PATH = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/evaluation/test_predictions_Aug2.0/PuFiRS25_00112.nii.gz"
OUTPUT_PANEL = "prediction_comparison_panel_00112.png"
OUTPUT_GIF = "prediction_comparison_animation_00112.gif"

def main():
    print("📦 Loading 3D volumetric NIfTI data streams...", flush=True)
    
    # Load and orient NIfTI volumes (transpose to match standard axial view)
    hrct_vol = np.transpose(nib.load(HRCT_PATH).get_fdata(), (1, 0, 2))
    gt_vol = np.transpose(nib.load(GT_PATH).get_fdata(), (1, 0, 2))
    pred_vol = np.transpose(nib.load(PRED_PATH).get_fdata(), (1, 0, 2))
    
    # Enforce standard clinical HU windowing [-1024, 300] for maximum tissue contrast
    hrct_vol = np.clip(hrct_vol, -1024, 300)
    
    # Isolate slices containing active fibrosis to avoid printing empty slices
    active_slices = np.where(np.sum(gt_vol >= 1, axis=(0, 1)) > 500)[0]
    if len(active_slices) == 0:
        active_slices = np.arange(hrct_vol.shape[2])
        
    # Select three evenly distributed slices across the disease span (Lower, Middle, Upper)
    slices_to_plot = [
        active_slices[int(len(active_slices) * 0.25)],
        active_slices[int(len(active_slices) * 0.50)],
        active_slices[int(len(active_slices) * 0.75)]
    ]
    
    print(f"🎨 Generating static 3-slice panel grid for slices: {slices_to_plot}...", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # Set up translucent segmentation overlay mask properties
    mask_color = np.array([255, 0, 0], dtype=np.uint8) # Red
    alpha = 0.4
    
    for col_idx, slice_idx in enumerate(slices_to_plot):
        ct_slice = hrct_vol[:, :, slice_idx]
        gt_slice = gt_vol[:, :, slice_idx] >= 1
        pred_slice = pred_vol[:, :, slice_idx] >= 1
        
        # Convert grayscale CT slice to RGB canvas
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        ct_rgb_gt = np.stack([ct_norm]*3, axis=-1)
        ct_rgb_pred = np.stack([ct_norm]*3, axis=-1)
        
        # Apply translucent blending seams
        ct_rgb_gt[gt_slice] = (1 - alpha) * ct_rgb_gt[gt_slice] + alpha * (mask_color / 255.0)
        ct_rgb_pred[pred_slice] = (1 - alpha) * ct_rgb_pred[pred_slice] + alpha * (mask_color / 255.0)
        
        # Row 0: Ground Truth
        axes[0, col_idx].imshow(ct_rgb_gt)
        axes[0, col_idx].axis('off')
        if col_idx == 0:
            axes[0, col_idx].text(15, 35, "GROUND TRUTH", color='white', fontsize=12, fontweight='bold')
        axes[0, col_idx].set_title(f"Axial Slice {slice_idx}", color='black', fontsize=10)
            
        # Row 1: nnU-Net Aug2.0 Predictions
        axes[1, col_idx].imshow(ct_rgb_pred)
        axes[1, col_idx].axis('off')
        if col_idx == 0:
            axes[1, col_idx].text(15, 35, "AUG 2.0x PREDICTION", color='white', fontsize=12, fontweight='bold')

    plt.savefig(OUTPUT_PANEL, bbox_inches='tight', pad_inches=0.1)
    print(f"✅ Static paper-ready panel grid saved successfully: {OUTPUT_PANEL}", flush=True)

    # --- Volumetric GIF Animation Block ---
    print(f"🎬 Compiling animated volumetric GIF across {len(active_slices)} slices...", flush=True)
    fig_anim, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(12, 6))
    fig_anim.patch.set_facecolor('black')
    
    ims = []
    for slice_idx in active_slices[::2]: # Skip every second slice to minimize GIF file overhead size
        ct_slice = hrct_vol[:, :, slice_idx]
        gt_slice = gt_vol[:, :, slice_idx] >= 1
        pred_slice = pred_vol[:, :, slice_idx] >= 1
        
        ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
        rgb_gt = np.stack([ct_norm]*3, axis=-1)
        rgb_pred = np.stack([ct_norm]*3, axis=-1)
        
        rgb_gt[gt_slice] = (1 - alpha) * rgb_gt[gt_slice] + alpha * (mask_color / 255.0)
        rgb_pred[pred_slice] = (1 - alpha) * rgb_pred[pred_slice] + alpha * (mask_color / 255.0)
        
        # Render twin subplots frame
        im_gt = ax_gt.imshow(rgb_gt, animated=True)
        im_pred = ax_pred.imshow(rgb_pred, animated=True)
        
        if len(ims) == 0:
            ax_gt.axis('off')
            ax_pred.axis('off')
            ax_gt.set_title("Ground Truth", color='white', fontsize=14, fontweight='bold')
            ax_pred.set_title("Aug 2.0x Prediction", color='white', fontsize=14, fontweight='bold')
            
        txt = ax_gt.text(0.5, 0.95, f"Slice {slice_idx}", color='white', transform=ax_gt.transAxes, ha='center', fontsize=12)
        ims.append([im_gt, im_pred, txt])
        
    ani = animation.ArtistAnimation(fig_anim, ims, interval=80, blit=False, repeat_delay=1000)
    ani.save(OUTPUT_GIF, writer='pillow', fps=12, bitrate=2000)
    print(f"✅ Volumetric comparison animation compiled: {OUTPUT_GIF}", flush=True)

if __name__ == "__main__":
    main()
