"""
Patched degradations module to fix OpenCV 4.12 compatibility issues.

OpenCV 4.12 is stricter about parameter types in cv2.imencode.
The encode_param argument requires Python native int types.
"""

import cv2
import numpy as np
from io import BytesIO

# Re-export basicsr degradations that actually exist in current version
# Only import functions used by ffhq_degradation_dataset.py
from basicsr.data.degradations import (
    random_add_gaussian_noise,
    random_add_jpg_compression,
    random_mixed_kernels,
)

# Patch the add_jpg_compression function to ensure proper int types
def add_jpg_compression(img, quality=90):
    """Add JPG compression artifacts with OpenCV 4.12 compatibility.
    
    Args:
        img (ndarray): Input image in range [0, 1] with RGB order.
        quality (int/float): JPG quality. Default: 90.
        
    Returns:
        ndarray: Compressed image in range [0, 1] with RGB order.
    """
    img = np.clip(img, 0, 1)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Ensure quality is a native Python int (not numpy int)
    # This is the fix for OpenCV 4.12 compatibility
    quality_int = int(quality)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_int]
    
    _, encimg = cv2.imencode('.jpg', img_bgr * 255., encode_param)
    img_bgr = cv2.imdecode(encimg, 1)
    
    # Convert back to RGB and normalize
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    
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
