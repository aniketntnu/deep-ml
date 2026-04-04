import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    B, C, H, W = X.shape

    # Initialize running stats if not provided
    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1))
    if running_var is None:
        running_var = np.ones((1, C, 1, 1))

    if training:
        # Step 1: Compute mean over (B, H, W) for each channel
        # shape → (1, C, 1, 1)
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)

        # Step 2: Compute variance over (B, H, W)
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)

        # Step 3: Normalize
        X_norm = (X - batch_mean) / np.sqrt(batch_var + epsilon)

        # Step 4: Update running statistics
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var  = (1 - momentum) * running_var  + momentum * batch_var

    else:
        # Use stored running stats during inference
        X_norm = (X - running_mean) / np.sqrt(running_var + epsilon)

    # Step 5: Scale and shift
    out = gamma * X_norm + beta

    return out, running_mean, running_var