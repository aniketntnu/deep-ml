import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    """
    
    # Step 1: Convert to numpy array (handles list input also)
    try:
        img = np.array(image)
    except:
        return -1
    
    # Step 2: Validate shape → must be 3D and last dim = 3
    if img.ndim != 3 or img.shape[2] != 3:
        return -1
    
    H, W, C = img.shape
    
    # Step 3: Check empty dimensions
    if H == 0 or W == 0:
        return -1
    
    # Step 4: Check pixel value range [0,255]
    if np.any(img < 0) or np.any(img > 255):
        return -1
    
    # Step 5: Apply luminosity formula
    # Gray = 0.299R + 0.587G + 0.114B
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    
    gray = 0.299 * R + 0.587 * G + 0.114 * B
    
    # Step 6: Round and convert to integer
    gray = np.round(gray).astype(int)
    
    # Step 7: Convert to list
    return gray.tolist()