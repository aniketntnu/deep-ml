import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    
    # Step 1: Standardize data (zero mean, unit variance)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    X = (data - mean) / std
    
    # Step 2: Compute covariance matrix (features x features)
    cov = np.cov(X, rowvar=False)
    
    # Step 3: Eigen decomposition (symmetric matrix → use eigh)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Step 4: Sort eigenvalues (descending) and reorder eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top k eigenvectors (principal components)
    components = eigenvectors[:, :k]
    
    # Step 6: Fix sign convention
    for i in range(components.shape[1]):
        vec = components[:, i]
        
        # find first element with abs > 1e-10
        for j in range(len(vec)):
            if abs(vec[j]) > 1e-10:
                if vec[j] < 0:
                    components[:, i] = -vec
                break
    
    # Step 7: Round to 4 decimals
    return np.round(components, 4)