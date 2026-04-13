#!/usr/bin/env python3
"""
Sanity check script - Test multiple checkpoints on a single image.

Usage:
    # Test all checkpoints on a HQ image (no degradation)
    python scripts/sanity_check_checkpoints.py \
        --checkpoint-dir ./experiments/2026-04-13T11-59-20_DAEFR_predegraded/ \
        --config configs/DAEFR.yaml \
        --input ./datasets/FFHQ/images512x512_validation/celeba_512_validation/00000000.png \
        --output ./sanity_check_results
    
    # Test on synthetically degraded image (better for showing restoration)
    python scripts/sanity_check_checkpoints.py \
        --checkpoint-dir ./experiments/2026-04-13T11-59-20_DAEFR_predegraded/ \
        --config configs/DAEFR.yaml \
        --input ./datasets/FFHQ/images512x512_validation/celeba_512_validation/00000000.png \
        --output ./sanity_check_results \
        --degrade
    
    # Or use a pre-degraded low-quality image
    python scripts/sanity_check_checkpoints.py \
        --checkpoint-dir ./experiments/2026-04-13T11-59-20_DAEFR_predegraded/ \
        --config configs/DAEFR.yaml \
        --input ./datasets/ffhq32/00000.png \
        --output ./sanity_check_results
"""

import argparse
import os
import sys
import glob
import re
from pathlib import Path

import torch
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from main_DAEFR import instantiate_from_config
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from facexlib.utils.face_restoration_helper import FaceRestoreHelper


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    """Load model from config and state dict."""
    if "ckpt_path" in config.params:
        config.params.ckpt_path = None
    if "ckpt_path_HQ" in config.params:
        config.params.ckpt_path_HQ = None
    if "ckpt_path_LQ" in config.params:
        config.params.ckpt_path_LQ = None
    
    try:
        if "ckpt_path" in config.params.first_stage_config.params:
            config.params.first_stage_config.params.ckpt_path = None
        if "ckpt_path" in config.params.cond_stage_config.params:
            config.params.cond_stage_config.params.ckpt_path = None
    except:
        pass

    model = instantiate_from_config(config)
    if sd is not None:
        state_dict = model.state_dict()
        require_keys = state_dict.keys()
        keys = sd.keys()
        
        un_pretrained_keys = []
        count = 0
        for k in require_keys:
            if k not in keys:
                if k[6:] in keys:
                    state_dict[k] = sd[k[6:]]
                else:
                    un_pretrained_keys.append(k)
            else:
                count = count + 1
                state_dict[k] = sd[k]
        
        print(f'  Loaded {count}/{len(require_keys)} keys')
        model.load_state_dict(state_dict, strict=True)

    if gpu:
        model.cuda()
    if eval_mode:
        model.eval()
    return model


def load_model(checkpoint_path, config_path):
    """Load model from checkpoint and config."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Load checkpoint weights
    pl_sd = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Load model
    model = load_model_from_config(config.model, pl_sd.get("state_dict", pl_sd), gpu=True, eval_mode=True)
    
    return model, config


def apply_degradation(img, blur_kernel_size=21, noise_sigma=15, jpeg_quality=50, downsample_factor=4):
    """Apply synthetic degradation to image to simulate low-quality input."""
    h, w = img.shape[:2]
    
    # Downsample then upsample (simulates low resolution)
    if downsample_factor > 1:
        small_h, small_w = h // downsample_factor, w // downsample_factor
        img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Gaussian blur
    if blur_kernel_size > 1:
        img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    
    # Add noise
    if noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # JPEG compression artifacts
    if jpeg_quality < 100:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)
    
    return img


def run_inference(model, face_helper, input_path, output_dir, suffix=None, degraded_input=None):
    """Run inference on a single image."""
    
    # Read image
    img_name = os.path.basename(input_path)
    basename, ext = os.path.splitext(img_name)
    
    input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if input_img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Resize to 512x512 (model expects this size)
    input_img = cv2.resize(input_img, (512, 512))
    
    # Use degraded version if provided
    if degraded_input is not None:
        input_img = degraded_input
    
    face_helper.clean_all()
    face_helper.cropped_faces = [input_img]
    
    # Prepare input
    cropped_face = face_helper.cropped_faces[0]
    cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to('cuda')
    
    # Inference
    with torch.no_grad():
        output = model(cropped_face_t)
        restored_face = tensor2img(output[0].squeeze(0), rgb2bgr=True, min_max=(-1, 1))
    
    restored_face = restored_face.astype('uint8')
    
    # Save output
    if suffix:
        save_name = f"{basename}_{suffix}.png"
    else:
        save_name = f"{basename}_restored.png"
    
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, restored_face)
    
    return save_path


def find_checkpoints(checkpoint_dir):
    """Find all checkpoint files in directory, sorted by epoch."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for .ckpt files
    all_checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    if not all_checkpoints:
        return []
    
    # Filter out last.ckpt if we have epoch checkpoints
    epoch_checkpoints = [c for c in all_checkpoints if re.search(r'epoch[=\-]?\d+', c.name)]
    
    if epoch_checkpoints:
        # Sort by epoch number
        def get_epoch(ckpt_path):
            match = re.search(r'epoch[=\-]?(\d+)', ckpt_path.name)
            return int(match.group(1)) if match else 0
        
        sorted_checkpoints = sorted(epoch_checkpoints, key=get_epoch)
        return [(str(c), get_epoch(c)) for c in sorted_checkpoints]
    else:
        # Just return all checkpoints without last.ckpt
        return [(str(c), None) for c in all_checkpoints if "last.ckpt" not in c.name]


def main():
    parser = argparse.ArgumentParser(description="Sanity check: Test multiple checkpoints on one image")
    parser.add_argument("--checkpoint-dir", "-d", type=str, required=True,
                        help="Directory containing checkpoint files")
    parser.add_argument("--config", "-c", type=str, default="configs/DAEFR.yaml",
                        help="Path to model config (default: configs/DAEFR.yaml)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input test image")
    parser.add_argument("--output", "-o", type=str, default="./sanity_check_results",
                        help="Output directory for results (default: ./sanity_check_results)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--degrade", action="store_true",
                        help="Apply synthetic degradation to input (blur, noise, JPEG) for better restoration test")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Find checkpoints
    print(f"\n{'='*60}")
    print("SANITY CHECK: Testing Multiple Checkpoints")
    print(f"{'='*60}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Input image: {args.input}")
    print(f"Output dir: {args.output}")
    print(f"{'='*60}\n")
    
    checkpoints = find_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print("No checkpoints found!")
        sys.exit(1)
    
    print(f"Found {len(checkpoints)} checkpoint(s):")
    for ckpt_path, epoch in checkpoints:
        ckpt_name = os.path.basename(ckpt_path)
        print(f"  - {ckpt_name}")
    print()
    
    # Initialize face helper (shared across all checkpoints)
    face_helper = FaceRestoreHelper(
        upscale_factor=1, 
        face_size=512, 
        crop_ratio=(1, 1), 
        det_model='retinaface_resnet50', 
        save_ext='png'
    )
    
    # Prepare degraded input if requested
    degraded_input = None
    if args.degrade:
        print("Applying synthetic degradation to input image...")
        input_img = cv2.imread(args.input, cv2.IMREAD_COLOR)
        input_img = cv2.resize(input_img, (512, 512))
        degraded_input = apply_degradation(input_img)
        
        # Save degraded input for reference
        degraded_path = os.path.join(args.output, "degraded_input.png")
        cv2.imwrite(degraded_path, degraded_input)
        print(f"  ✓ Saved degraded input: {degraded_path}")
        print()
    
    # Process with each checkpoint
    results = []
    
    for ckpt_path, epoch in tqdm(checkpoints, desc="Testing checkpoints"):
        ckpt_name = os.path.basename(ckpt_path)
        
        # Create suffix from checkpoint name (e.g., "epoch_047")
        if epoch is not None:
            suffix = f"epoch_{epoch:03d}"
        else:
            # Use checkpoint filename without extension
            suffix = Path(ckpt_path).stem[:20]  # Truncate long names
        
        print(f"\nProcessing with: {ckpt_name}")
        
        try:
            # Load model
            model, config = load_model(ckpt_path, args.config)
            
            # Run inference (with degraded input if --degrade was used)
            output_path = run_inference(model, face_helper, args.input, args.output, suffix, degraded_input)
            
            results.append({
                'checkpoint': ckpt_name,
                'epoch': epoch,
                'output': output_path,
                'status': 'success'
            })
            
            print(f"  ✓ Saved: {os.path.basename(output_path)}")
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results.append({
                'checkpoint': ckpt_name,
                'epoch': epoch,
                'output': None,
                'status': f'failed: {e}'
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        epoch_info = f"(epoch {r['epoch']})" if r['epoch'] else ""
        print(f"  ✓ {r['checkpoint']} {epoch_info}")
        print(f"    → {os.path.basename(r['output'])}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"  ✗ {r['checkpoint']}: {r['status']}")
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
