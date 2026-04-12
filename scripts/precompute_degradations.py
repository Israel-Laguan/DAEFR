"""
Pre-compute degraded images for fast training.
Usage: python scripts/precompute_degradations.py --output_dir ./datasets/FFHQ_LQ

This generates LQ (low quality) images once, then training loads them directly
instead of computing degradations on-the-fly (10x+ speedup).

Uses multiprocessing with all CPU cores minus 1 for maximum speed.
"""
import os
import sys
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from basicsr.utils import tensor2img


def init_worker(config_path, split):
    """Initialize worker process with dataset."""
    global _dataset
    from DAEFR.data.ffhq_degradation_dataset import FFHQDegradationDataset
    config = OmegaConf.load(config_path)
    data_params = config.data.params
    dataset_config = data_params[split]['params']
    _dataset = FFHQDegradationDataset(dataset_config)


def process_image(idx_output):
    """Process a single image (called by worker pool)."""
    idx, output_dir = idx_output
    sample = _dataset.getitem_degraded(idx)
    
    # Get filename from gt_path
    gt_path = sample['gt_path']
    name = Path(gt_path).stem
    
    # Save LQ image
    lq = tensor2img(sample['lq'])
    output_path = Path(output_dir) / f"{name}.png"
    cv2.imwrite(str(output_path), lq)
    
    return idx


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
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    args = parser.parse_args()

    # Load config to get dataset length
    config = OmegaConf.load(args.config)
    data_params = config.data.params
    dataset_config = data_params[args.split]['params']
    
    from DAEFR.data.ffhq_degradation_dataset import FFHQDegradationDataset
    dataset = FFHQDegradationDataset(dataset_config)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine number of samples
    num_samples = args.num_samples or len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    # Determine number of workers
    num_workers = args.num_workers or max(1, cpu_count() - 1)
    
    print(f"Generating {num_samples} degraded images using {num_workers} workers...")
    print(f"Source: {dataset_config.dataroot_gt}")
    print(f"Output: {output_dir}")
    print(f"Degradations: blur, downsample, noise, JPEG compression")
    print(f"CPU cores available: {cpu_count()}, using: {num_workers}")
    
    # Prepare work items
    work_items = [(i, str(output_dir)) for i in range(num_samples)]
    
    # Process in parallel with progress bar
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.config, args.split)
    ) as pool:
        list(tqdm(
            pool.imap(process_image, work_items),
            total=num_samples,
            desc="Generating LQ images"
        ))
    
    print(f"\nDone! Generated {num_samples} images in {output_dir}")
    print(f"\nTo use for training, update your config:")
    print(f"  dataroot_lq: {args.output_dir}")


if __name__ == '__main__':
    main()
