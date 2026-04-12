"""
Pre-compute degraded images for fast training.
Usage: python scripts/precompute_degradations.py --output_dir ./datasets/FFHQ_LQ

This generates LQ (low quality) images once, then training loads them directly
instead of computing degradations on-the-fly (10x+ speedup).
"""
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from basicsr.utils import tensor2img

from DAEFR.data.ffhq_degradation_dataset import FFHQDegradationDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/DAEFR.yaml',
                        help='Path to config file with degradation parameters')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pre-computed LQ images')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'validation'],
                        help='Which split to process')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to generate (default: all)')
    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)
    
    # Get dataset config for the specified split
    data_params = config.data.params
    dataset_config = data_params[args.split]['params']
    
    # Create dataset
    dataset = FFHQDegradationDataset(dataset_config)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of samples
    num_samples = args.num_samples or len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    print(f"Generating {num_samples} degraded images...")
    print(f"Source: {dataset_config.dataroot_gt}")
    print(f"Output: {output_dir}")
    print(f"Degradations: blur, downsample, noise, JPEG compression")
    
    # Generate and save
    for i in tqdm(range(num_samples)):
        sample = dataset.getitem_degraded(i)
        
        # Get filename from gt_path
        gt_path = sample['gt_path']
        name = Path(gt_path).stem
        
        # Save LQ image
        lq = tensor2img(sample['lq'])
        output_path = output_dir / f"{name}.png"
        cv2.imwrite(str(output_path), lq)
    
    print(f"\nDone! Generated {num_samples} images in {output_dir}")
    print(f"\nTo use for training, update your config:")
    print(f"  dataroot_lq: {args.output_dir}")


if __name__ == '__main__':
    main()
