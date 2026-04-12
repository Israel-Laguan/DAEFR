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
    # Limit OpenCV and OpenMP threads per worker to prevent resource exhaustion
    import cv2
    cv2.setNumThreads(1)  # Single thread per worker
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    from DAEFR.data.ffhq_degradation_dataset import FFHQDegradationDataset
    config = OmegaConf.load(config_path)
    data_params = config.data.params
    dataset_config = data_params[split]['params']
    _dataset = FFHQDegradationDataset(dataset_config)


def process_batch(batch_info):
    """Process a batch of images (called by worker pool).
    
    Args:
        batch_info: tuple of (indices, output_dir, batch_num, total_batches)
    Returns:
        Dict with 'processed' and 'failed' lists
    """
    indices, output_dir, batch_num, total_batches = batch_info
    output_path = Path(output_dir)
    processed = []
    failed = []
    
    for idx in indices:
        try:
            sample = _dataset.getitem_degraded(idx)
            
            # Get filename from gt_path
            gt_path = sample['gt_path']
            name = Path(gt_path).stem
            output_file = output_path / f"{name}.png"
            
            # Skip if already exists (resume support)
            if output_file.exists():
                processed.append(idx)
                continue
            
            # Save LQ image
            lq = tensor2img(sample['lq'])
            cv2.imwrite(str(output_file), lq)
            processed.append(idx)
            
        except Exception as e:
            # Log error and continue with other images in batch
            failed.append((idx, str(e)))
    
    return {'processed': processed, 'failed': failed}


def get_usable_cpu_count():
    """Get actual usable CPU count respecting cgroup/scheduler limits."""
    try:
        # sched_getaffinity respects cgroup limits (best for containers)
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        # Fallback to cpu_count if sched_getaffinity not available
        return os.cpu_count() or 4


def main():
    # Set thread limits at startup for main process too
    import cv2
    cv2.setNumThreads(1)  # Further reduce to 1 thread per process
    # Suppress OpenCV warnings via environment (cv2.setLogLevel not available in older versions)
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
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
                        help='Number of parallel workers (default: min(23, CPU count - 1))')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of images per worker batch (default: 20)')
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
    
    # Check for existing files to support resume
    existing_files = set(f.stem for f in output_dir.glob("*.png"))
    already_done = len(existing_files)
    
    if already_done > 0:
        print(f"📁 Found {already_done} existing LQ images (resuming...)")
    
    # Determine number of workers - use actual usable cores
    usable_cores = get_usable_cpu_count()
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        # Use all usable cores minus 1 for system, capped at 47 (tested: 48 threads available)
        num_workers = min(47, max(1, usable_cores - 1))
    
    remaining = num_samples - already_done
    print(f"Generating {remaining} degraded images using {num_workers} workers...")
    print(f"Source: {dataset_config.dataroot_gt}")
    print(f"Output: {output_dir}")
    print(f"Degradations: blur, downsample, noise, JPEG compression")
    print(f"Detected usable cores: {usable_cores}, using: {num_workers} workers")
    print(f"Note: Each worker limited to 1 thread. Override with --num_workers N")
    
    # Prepare batch work items for efficient multiprocessing
    # Each worker processes a batch of images, reducing overhead
    # Skip images that already exist (resume support)
    batch_size = args.batch_size
    
    work_items = []
    for idx in range(num_samples):
        gt_path = dataset.paths[idx]
        name = Path(gt_path).stem
        if name not in existing_files:
            work_items.append(idx)
    
    # Group into batches
    num_batches = (len(work_items) + batch_size - 1) // batch_size
    batched_work_items = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, len(work_items))
        batch_indices = work_items[start:end]
        batched_work_items.append((batch_indices, str(output_dir), i + 1, num_batches))
    
    work_items = batched_work_items
    
    print(f"Batch size: {batch_size} images per worker call")
    print(f"Total batches: {num_batches} (across {num_workers} workers)")
    
    # Process batches in parallel with progress tracking
    total_processed = 0
    failed_items = []  # Collect failed indices for retry
    
    print("\nProcessing batches (with resume and error handling)...")
    print(f"Progress: 0/{num_batches} batches", end='', flush=True)
    
    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(args.config, args.split)
    ) as pool:
        # Use imap with chunksize for better progress granularity
        # chunksize=1 ensures we get updates as each batch completes
        for i, result in enumerate(pool.imap_unordered(process_batch, work_items, chunksize=1)):
            total_processed += len(result['processed'])
            failed_items.extend(result['failed'])
            # Manual progress update every batch for responsive display
            pct = (i + 1) * 100 // num_batches
            print(f"\rProgress: {i + 1}/{num_batches} ({pct}%) | Images: {total_processed} | Failed: {len(failed_items)}    ", end='', flush=True)
    
    print()  # New line after progress
    
    # Retry failed items once (sequentially to avoid parallel issues)
    if failed_items:
        print(f"\n⚠️  {len(failed_items)} images failed. Retrying sequentially...")
        for idx, error in failed_items:
            try:
                sample = dataset.getitem_degraded(idx)
                gt_path = sample['gt_path']
                name = Path(gt_path).stem
                output_file = output_dir / f"{name}.png"
                
                if not output_file.exists():
                    lq = tensor2img(sample['lq'])
                    cv2.imwrite(str(output_file), lq)
                    total_processed += 1
                    print(f"  ✓ Retry successful: {name}")
                else:
                    total_processed += 1
                    
            except Exception as e:
                print(f"  ✗ Retry failed for index {idx}: {e}")
    
    # Final report
    final_count = len(list(output_dir.glob("*.png")))
    print(f"\n✅ Done! Total images in output: {final_count}")
    print(f"   Newly generated: {total_processed}")
    print(f"   Already existed: {already_done}")
    print(f"\nTo use for training, update your config:")
    print(f"  dataroot_lq: {args.output_dir}")


if __name__ == '__main__':
    main()
