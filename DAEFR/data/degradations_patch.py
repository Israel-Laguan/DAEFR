"""
Patched degradations module using PIL for JPEG compression.

PIL provides consistent JPEG compression without OpenCV version issues.
Slightly slower (23ms vs 22ms) but more reliable quality.
"""

import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Re-export basicsr degradations that actually exist in current version
# Only import functions used by ffhq_degradation_dataset.py
from basicsr.data.degradations import (
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_mixed_kernels,
)

# PIL-based JPEG compression (replaces OpenCV)
def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts using PIL.
    
    Args:
        img (ndarray): Input image in range [0, 1] with RGB order.
        quality (int/float): JPG quality. Default: 90.
        
    Returns:
        ndarray: Compressed image in range [0, 1] with RGB order.
    """
    img = np.clip(img, 0, 1).astype(np.float32)
    
    # Convert to uint8 PIL image (RGB)
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    
    # Compress to JPEG in memory buffer
    buffer = BytesIO()
    img_pil.save(buffer, format='JPEG', quality=int(quality))
    buffer.seek(0)
    
    # Decompress back to numpy array
    img_pil = Image.open(buffer)
    img = np.array(img_pil).astype(np.float32) / 255.
    
    return img


# Monkey-patch the random_add_jpg_compression to use our fixed version
import basicsr.data.degradations as _degradations_module
_degradations_module.add_jpg_compression = add_jpg_compression

# Also export the fixed function directly
__all__ = [
    'random_add_gaussian_noise',
    'random_add_jpg_compression',
    'random_mixed_kernels',
    'add_jpg_compression',
]
