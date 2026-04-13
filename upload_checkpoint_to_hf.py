#!/usr/bin/env python3
"""
Upload final DAEFR checkpoint to Hugging Face Hub.

Usage:
    # Initialize repo with README (do this before training)
    python upload_checkpoint_to_hf.py --init --repo-id your-username/DAEFR-final
    
    # Upload final checkpoint after training completes
    python upload_checkpoint_to_hf.py --final-checkpoint ./experiments/DAEFR_model.ckpt --repo-id your-username/DAEFR-final
    
    # Or auto-find latest checkpoint:
    python upload_checkpoint_to_hf.py --auto-find --repo-id your-username/DAEFR-final
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# README template for DAEFR model
README_TEMPLATE = """---
tags:
- image-restoration
- face-restoration
- super-resolution
- codeformer
- daefr
- pytorch
- computer-vision
license: apache-2.0
---

# DAEFR: Degradation-Aware Face Restoration

This model checkpoint is for **DAEFR (Degradation-Aware Face Restoration)** - a face restoration model that handles various degradation levels.

## Model Description

DAEFR is a degradation-aware face restoration framework that:
- Uses dual codebooks for high-quality and low-quality face restoration
- Employs an association stage to bridge HQ and LQ domains
- Achieves state-of-the-art results on blind face restoration benchmarks

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download checkpoint
checkpoint_path = hf_hub_download(
    repo_id="{{repo_id}}",
    filename="{{checkpoint_name}}"
)

# Load model
model = torch.load(checkpoint_path, map_location='cpu')
```

## Training Details

- **Epochs**: {epochs}
- **Dataset**: FFHQ 512x512
- **Degradation**: Synthetic blind degradation (blur, noise, JPEG, downsampling)

## Citation

```bibtex
@article{{DAEFR,
  title={{Degradation-Aware Face Restoration}},
  author={{}},
  journal={{}},
  year={{2024}}
}}
```
"""


def init_repo_with_readme(repo_id, token=None, private=False, epochs=100):
    """Initialize repo with README before training starts."""
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    
    api = HfApi(token=token)
    
    # Create repo
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token
        )
        print(f"Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
    
    # Generate README
    readme_content = README_TEMPLATE.format(
        repo_id=repo_id,
        checkpoint_name="DAEFR_model.ckpt",
        epochs=epochs
    )
    
    # Upload README
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode('utf-8'),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"README uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error uploading README: {e}")
        sys.exit(1)


def find_latest_checkpoint(experiments_dir="./experiments"):
    """Find the latest checkpoint by modification time."""
    
    patterns = [
        f"{experiments_dir}/*/checkpoints/*.ckpt",
        f"{experiments_dir}/*/*.ckpt",
        f"{experiments_dir}/*.ckpt",
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        return None
    
    # Sort by modification time (newest first)
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Prefer "last.ckpt" or the most recent one
    for ckpt in all_checkpoints:
        if "last.ckpt" in ckpt:
            return ckpt
    
    return all_checkpoints[0]


def find_best_epoch_checkpoint(experiments_dir="./experiments"):
    """Find checkpoint with highest epoch number (e.g., epoch=000048)."""
    import re
    
    patterns = [
        f"{experiments_dir}/*/checkpoints/*.ckpt",
        f"{experiments_dir}/*/*.ckpt",
        f"{experiments_dir}/*.ckpt",
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        return None
    
    # Extract epoch numbers and find highest
    best_checkpoint = None
    best_epoch = -1
    
    for ckpt in all_checkpoints:
        # Match patterns like epoch=000048 or epoch=000001
        match = re.search(r'epoch[=\-]?(\d+)', os.path.basename(ckpt))
        if match:
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_checkpoint = ckpt
        elif "last.ckpt" in ckpt and best_epoch == -1:
            # Fallback to last.ckpt if no epoch found yet
            best_checkpoint = ckpt
    
    return best_checkpoint


def upload_to_huggingface(checkpoint_path, repo_id, token=None, private=False):
    """Upload checkpoint to Hugging Face Hub."""
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    api = HfApi(token=token)
    
    # Create or get repo
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token
        )
        print(f"Repository ready: {repo_url}")
    except Exception as e:
        print(f"Note: Using existing repository or error creating: {e}")
    
    # Upload the checkpoint
    print(f"Uploading {checkpoint_path.name} to {repo_id}...")
    
    try:
        api.upload_file(
            path_or_fileobj=str(checkpoint_path),
            path_in_repo=checkpoint_path.name,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )
        print(f"Successfully uploaded to https://huggingface.co/{repo_id}")
        
        # Also upload config if it exists nearby
        config_path = checkpoint_path.parent / "DAEFR.yaml"
        if not config_path.exists():
            config_path = Path("./configs/DAEFR.yaml")
        
        if config_path.exists():
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo="DAEFR.yaml",
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"Also uploaded config: {config_path}")
            
    except Exception as e:
        print(f"Error uploading: {e}")
        sys.exit(1)


def upload_all_checkpoints(experiments_dir, repo_id, token=None, private=False, preserve_structure=True):
    """Upload all checkpoints from experiment folder to Hugging Face.
    
    Args:
        experiments_dir: Path to experiment folder containing checkpoints
        repo_id: HuggingFace repo ID
        token: HF token
        private: Make repo private
        preserve_structure: If True, upload to subfolder named after experiments_dir basename
    """
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)
    
    import re
    
    experiments_path = Path(experiments_dir)
    
    # Determine the subfolder name in HF repo
    if preserve_structure:
        # Use the leaf folder name (e.g., "2026-04-13T11-59-20_DAEFR_predegraded")
        hf_subfolder = experiments_path.name
        print(f"Preserving folder structure: uploading to '{hf_subfolder}/' subfolder")
    else:
        hf_subfolder = None
        print("Flat upload: all files at root level")
    
    # Find all checkpoint files
    patterns = [
        f"{experiments_dir}/*/*.ckpt",
        f"{experiments_dir}/*.ckpt",
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        print(f"No checkpoints found in {experiments_dir}")
        sys.exit(1)
    
    # Filter out last.ckpt if we have epoch checkpoints (to avoid duplicates)
    epoch_checkpoints = [c for c in all_checkpoints if re.search(r'epoch[=\-]?\d+', os.path.basename(c))]
    if epoch_checkpoints:
        checkpoints_to_upload = sorted(epoch_checkpoints, key=lambda x: int(re.search(r'epoch[=\-]?(\d+)', os.path.basename(x)).group(1)))
    else:
        checkpoints_to_upload = all_checkpoints
    
    print(f"Found {len(checkpoints_to_upload)} checkpoint(s) to upload:")
    for ckpt in checkpoints_to_upload:
        print(f"  - {os.path.basename(ckpt)}")
    
    api = HfApi(token=token)
    
    # Create or get repo
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=private,
            token=token
        )
        print(f"\nRepository ready: {repo_url}")
    except Exception as e:
        print(f"\nNote: Using existing repository: {e}")
    
    # Upload all checkpoints
    print(f"\nUploading to https://huggingface.co/{repo_id}...")
    
    for checkpoint_path in checkpoints_to_upload:
        ckpt_name = os.path.basename(checkpoint_path)
        
        # Set path in repo (with subfolder if preserving structure)
        if hf_subfolder:
            path_in_repo = f"{hf_subfolder}/{ckpt_name}"
        else:
            path_in_repo = ckpt_name
        
        print(f"\nUploading {ckpt_name}...")
        
        try:
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"  ✓ Uploaded {path_in_repo}")
        except Exception as e:
            print(f"  ✗ Failed to upload {path_in_repo}: {e}")
    
    # Upload config (to subfolder if preserving structure)
    config_path = experiments_path / "DAEFR.yaml"
    if not config_path.exists():
        config_path = Path("./configs/DAEFR.yaml")
    
    if config_path.exists():
        config_name = config_path.name
        if hf_subfolder:
            config_path_in_repo = f"{hf_subfolder}/{config_name}"
        else:
            config_path_in_repo = config_name
        
        print(f"\nUploading config: {config_name}")
        try:
            api.upload_file(
                path_or_fileobj=str(config_path),
                path_in_repo=config_path_in_repo,
                repo_id=repo_id,
                repo_type="model",
                token=token
            )
            print(f"  ✓ Uploaded {config_path_in_repo}")
        except Exception as e:
            print(f"  ✗ Failed to upload config: {e}")
    
    print(f"\n{'='*50}")
    print(f"All uploads complete: https://huggingface.co/{repo_id}")
    if hf_subfolder:
        print(f"Files uploaded to: {hf_subfolder}/")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Upload DAEFR checkpoint to Hugging Face")
    parser.add_argument("--init", action="store_true", 
                        help="Initialize repo with README only (do this before training)")
    parser.add_argument("--checkpoint", "-c", type=str, help="Path to checkpoint file")
    parser.add_argument("--final-checkpoint", "-f", type=str, 
                        default="./experiments/DAEFR_model.ckpt",
                        help="Path to final checkpoint to upload (default: ./experiments/DAEFR_model.ckpt)")
    parser.add_argument("--auto-find", action="store_true", 
                        help="Automatically find latest checkpoint by modification time")
    parser.add_argument("--use-best-epoch", action="store_true",
                        help="Find checkpoint with highest epoch number (e.g., epoch=000048)")
    parser.add_argument("--repo-id", "-r", type=str, required=True, 
                        help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--token", "-t", type=str, default=None, 
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--experiments-dir", type=str, default="./experiments",
                        help="Directory to search for checkpoints")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (for README)")
    parser.add_argument("--upload-all", action="store_true",
                        help="Upload ALL checkpoints from the experiment folder (not just best)")
    parser.add_argument("--preserve-structure", action="store_true", default=True,
                        help="Preserve folder structure when uploading (default: True)")
    parser.add_argument("--flat", action="store_true",
                        help="Upload files flat without folder structure (overrides --preserve-structure)")
    
    args = parser.parse_args()
    
    # Get token from env if not provided
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    
    # Handle --init flag (create repo with README only)
    if args.init:
        print("Initializing repository with README...")
        init_repo_with_readme(args.repo_id, token, args.private, args.epochs)
        print(f"\nRepo ready! After training completes, run:")
        print(f"  python upload_checkpoint_to_hf.py --use-best-epoch --repo-id {args.repo_id}")
        print(f"  # Or upload all: python upload_checkpoint_to_hf.py --upload-all --experiments-dir <folder> --repo-id {args.repo_id}")
        return
    
    # Handle --upload-all flag (upload all checkpoints from folder)
    if args.upload_all:
        print(f"Uploading ALL checkpoints from {args.experiments_dir}...")
        # --flat overrides --preserve-structure
        preserve_structure = args.preserve_structure and not args.flat
        upload_all_checkpoints(args.experiments_dir, args.repo_id, token, args.private, preserve_structure)
        return
    
    # Determine which checkpoint to use (single upload)
    checkpoint = None
    if args.checkpoint:
        checkpoint = args.checkpoint
        print(f"Using specified checkpoint: {checkpoint}")
    elif args.use_best_epoch:
        print(f"Searching for highest epoch checkpoint in {args.experiments_dir}...")
        checkpoint = find_best_epoch_checkpoint(args.experiments_dir)
        if checkpoint:
            # Extract epoch number for display
            import re
            match = re.search(r'epoch[=\-]?(\d+)', os.path.basename(checkpoint))
            if match:
                print(f"Found epoch {int(match.group(1))}: {checkpoint}")
            else:
                print(f"Found: {checkpoint}")
        else:
            print("No checkpoint found!")
            sys.exit(1)
    elif args.auto_find:
        print(f"Searching for latest checkpoint by time in {args.experiments_dir}...")
        checkpoint = find_latest_checkpoint(args.experiments_dir)
        if checkpoint:
            print(f"Found: {checkpoint}")
        else:
            print("No checkpoint found!")
            sys.exit(1)
    elif args.final_checkpoint:
        checkpoint = args.final_checkpoint
        if not os.path.exists(checkpoint):
            print(f"Error: Final checkpoint not found at {checkpoint}")
            print("Training may not be complete yet. Use --use-best-epoch to find latest.")
            sys.exit(1)
        print(f"Using final checkpoint: {checkpoint}")
    else:
        print("Error: Must specify one of --checkpoint, --final-checkpoint, --auto-find, --use-best-epoch, --upload-all, or --init")
        parser.print_help()
        sys.exit(1)
    
    upload_to_huggingface(checkpoint, args.repo_id, token, args.private)


if __name__ == "__main__":
    main()
