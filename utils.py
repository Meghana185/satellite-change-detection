import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_and_resize_image(image_path, target_size=None):
    """
    Loads an image from path and optionally resizes it.
    If target_size is provided as (width, height), resizes the image to match.
    Returns a numpy array (RGB).
    """
    # Using PIL for robust reading (handles streamlit uploaded files via BytesIO easily later)
    # but since this might be a file path or a PIL image directly, let's handle paths:
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        # Assuming it's already a PIL image or similar file-like object
        img = Image.open(image_path).convert('RGB')
        
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.BILINEAR)
        
    return np.array(img)

def overlay_change_mask(image_array, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Overlays a boolean change mask onto the original RGB image array.
    Pixels where mask == True will be tinted with the specified color.
    
    Args:
        image_array: numpy array, original RGB image
        mask: boolean numpy array (2D) of the same (height, width)
        alpha: transparency of the overlay (0-1)
        color: RGB tuple for the highlight color (default red)
        
    Returns:
        numpy array, the overlaid RGB image
    """
    overlay = np.copy(image_array)
    # Create a solid color mask
    color_mask = np.zeros_like(image_array, dtype=np.uint8)
    color_mask[:] = color
    
    # Where mask is true, blend the original image with the color
    overlay[mask] = cv2.addWeighted(image_array[mask], 1 - alpha, color_mask[mask], alpha, 0)
    
    return overlay
