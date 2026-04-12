#!/usr/bin/env python3
"""
Download pre-degraded FFHQ 32x32 dataset from HuggingFace and extract to individual images.

This replaces the need to pre-compute degradations locally. The 32x32 images already have
blur, noise, and JPEG compression baked in. During training, they are upsampled to 512x512
with light noise added for variation.

Usage:
    # Download FFHQ 32x32 (default, ~150 MB)
    python scripts/download_predegraded.py

    # Download FFHQ 64x64 (~878 MB, higher quality)
    python scripts/download_predegraded.py --size 64

    # Custom output directory
    python scripts/download_predegraded.py --output ./datasets/ffhq32
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm


def download_ffhq32(output_dir: str = './datasets/ffhq32'):
    """Download FFHQ 32x32 from HuggingFace and extract individual images."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing = list(output_path.glob('*.png'))
    if len(existing) >= 70000:
        print(f"Already extracted {len(existing)} images in {output_path}. Skipping.")
        return str(output_path)

    print(f"Downloading FFHQ 32x32 from HuggingFace (leellodadi/ffhq32)...")
    print(f"Output: {output_path}")

    dataset = load_dataset("leellodadi/ffhq32", split="train", trust_remote_code=True)

    print(f"Extracting {len(dataset)} images...")

    for i in tqdm(range(len(dataset)), desc="Extracting"):
        item = dataset[i]
        img = item['image']  # PIL Image

        # Save with zero-padded index to match GT naming convention
        img.save(output_path / f"{i:05d}.png")

    count = len(list(output_path.glob('*.png')))
    print(f"Done! Extracted {count} images to {output_path}")
    return str(output_path)


def download_ffhq64(output_dir: str = './datasets/ffhq64'):
    """Download FFHQ 64x64 from HuggingFace and extract individual images."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    existing = list(output_path.glob('*.png'))
    if len(existing) >= 70000:
        print(f"Already extracted {len(existing)} images in {output_path}. Skipping.")
        return str(output_path)

    print(f"Downloading FFHQ 64x64 from HuggingFace (Dmini/FFHQ-64x64)...")
    print(f"Output: {output_path}")

    dataset = load_dataset("Dmini/FFHQ-64x64", split="train", trust_remote_code=True)

    print(f"Extracting {len(dataset)} images...")

    for i in tqdm(range(len(dataset)), desc="Extracting"):
        item = dataset[i]
        img = item['image']  # PIL Image
        img.save(output_path / f"{i:05d}.png")

    count = len(list(output_path.glob('*.png')))
    print(f"Done! Extracted {count} images to {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-degraded FFHQ dataset from HuggingFace"
    )
    parser.add_argument('--size', type=int, default=32, choices=[32, 64],
                        help='Image size: 32 or 64 (default: 32)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: ./datasets/ffhq32 or ./datasets/ffhq64)')

    args = parser.parse_args()

    if args.output is None:
        args.output = f'./datasets/ffhq{args.size}'

    if args.size == 32:
        download_ffhq32(args.output)
    else:
        download_ffhq64(args.output)


if __name__ == '__main__':
    main()
