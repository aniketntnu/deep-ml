import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    # Your code here
    

    predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
    logSample = np.log(predicted_probs)

    loss = np.sum(true_labels * logSample ,axis =1)

    avg = np.mean(loss)

    return -1 * avg 