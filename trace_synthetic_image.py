#!/usr/bin/env python3
"""
Trace a synthetic image back to its source training case and perturbation.
Usage: python trace_synthetic_image.py synthetic_0005
"""

import json
import sys
import os
from pathlib import Path

def trace_image(synthetic_id: str):
    """Find which training case and perturbation a synthetic image came from."""
    
    metadata_dir = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2"
    
    # Find all metadata files
    metadata_files = sorted(Path(metadata_dir).glob("synthetic_metadata_batch*.json"))
    
    if not metadata_files:
        print(f"❌ No metadata files found in {metadata_dir}")
        return
    
    # Search through all metadata files
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for entry in metadata:
            if entry['synthetic_id'] == synthetic_id:
                print(f"\n✓ Found synthetic image: {synthetic_id}")
                print(f"  Source training case: {entry['source_case']}")
                print(f"  Perturbation type: {entry['perturbation_type']}")
                print(f"  Batch number: {entry['batch_number']}")
                print(f"  CT file: {entry['ct_file']}")
                print(f"  Mask file: {entry['mask_file']}")
                return
    
    print(f"❌ Synthetic image '{synthetic_id}' not found in metadata")

def list_all(batch_num: int = None):
    """List all synthetic images and their sources."""
    
    metadata_dir = "/projects/u6dm/mk25bm.u6dm/Fibrosis-3D-nnUnet-DDPM/augmented_data/synthetic_images_v2"
    metadata_files = sorted(Path(metadata_dir).glob("synthetic_metadata_batch*.json"))
    
    print(f"\n{'Synthetic ID':<20} {'Source Case':<20} {'Perturbation':<25} {'Batch':<8}")
    print("-" * 75)
    
    for metadata_file in metadata_files:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for entry in metadata:
            if batch_num is None or entry['batch_number'] == batch_num:
                print(f"{entry['synthetic_id']:<20} {entry['source_case']:<20} {entry['perturbation_type']:<25} {entry['batch_number']:<8}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python trace_synthetic_image.py <synthetic_id>    # Trace a single image")
        print("  python trace_synthetic_image.py --list             # List all images")
        print("  python trace_synthetic_image.py --list 1           # List batch 1 only")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        batch = int(sys.argv[2]) if len(sys.argv) > 2 else None
        list_all(batch)
    else:
        trace_image(sys.argv[1])
