import cv2
import numpy as np

from config import PATCH_SIZE

def extract_patches(image: np.ndarray) -> list[np.ndarray]:
    """
    Extracts 5 fixed patches of shape (PATCH_SIZE, PATCH_SIZE) from the input image:
    [Top-Left, Top-Right, Bottom-Left, Bottom-Right, Center].

    If the image dimensions are smaller than PATCH_SIZE, the image is padded
    using reflection padding before patch extraction.

    Args:
        image: Input image as a NumPy array. Can be 2D (grayscale) or 3D (color).

    Returns:
        A list of 5 NumPy arrays representing the patches in order:
        [top_left, top_right, bottom_left, bottom_right, center].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy ndarray.")

    h, w = image.shape[:2]

    # Calculate padding if image dimensions are smaller than PATCH_SIZE
    pad_h = max(0, PATCH_SIZE - h)
    pad_w = max(0, PATCH_SIZE - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # cv2.copyMakeBorder is highly optimized and handles arbitrary channel depths
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101
        )
        h, w = image.shape[:2]

    # O(1) Patch coordinates extraction
    top_left = image[0:PATCH_SIZE, 0:PATCH_SIZE]
    top_right = image[0:PATCH_SIZE, w - PATCH_SIZE:w]
    bottom_left = image[h - PATCH_SIZE:h, 0:PATCH_SIZE]
    bottom_right = image[h - PATCH_SIZE:h, w - PATCH_SIZE:w]

    # Center patch coordinates
    center_y = h // 2
    center_x = w // 2
    start_y = center_y - PATCH_SIZE // 2
    end_y = start_y + PATCH_SIZE
    start_x = center_x - PATCH_SIZE // 2
    end_x = start_x + PATCH_SIZE
    
    center = image[start_y:end_y, start_x:end_x]

    return [top_left, top_right, bottom_left, bottom_right, center]
