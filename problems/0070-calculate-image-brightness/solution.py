import numpy as np

def calculate_brightness(img):
    
    # Step 1: convert to numpy array
    try:
        arr = np.array(img)
    except:
        return -1
    
    # Step 2: check empty or not 2D
    if arr.size == 0 or arr.ndim != 2:
        return -1
    
    # Step 3: check consistent shape (no ragged rows)
    # if ragged → dtype becomes object
    if arr.dtype == object:
        return -1
    
    # Step 4: check pixel range
    if np.any(arr < 0) or np.any(arr > 255):
        return -1
    
    # Step 5: compute mean
    avg = np.mean(arr)
    
    # Step 6: round
    return round(float(avg), 2)