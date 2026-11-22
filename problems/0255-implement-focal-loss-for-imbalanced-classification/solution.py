import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):

    """
    Compute Focal Loss for multi-class classification.
    
    Args:
        y_true: Ground truth labels as class indices (list or 1D array)
        y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weights (optional, list or 1D array of length n_classes)
    
    Returns:
        float: Average focal loss
    """

    """
    Compute Focal Loss for multi-class classification.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    # Get the predicted probability of the true class
    #pt = y_pred[np.arange(len(y_true)), y_true]

    pt = []

    for i in range(len(y_true)):
        true_class = y_true[i]
        probability = y_pred[i][true_class]
        pt.append(probability)

    pt = np.array(pt)


    # Clip probabilities to avoid log(0)
    pt = np.clip(pt, 1e-8, 1.0)

    # Calculate the standard cross-entropy loss
    ce = -np.log(pt)

    # Calculate the focal weight
    focal_weight = (1.0 - pt) ** gamma

    # Calculate focal loss
    loss = focal_weight * ce

    # Apply class weights if alpha is provided
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=float)
        loss = loss * alpha[y_true]

    # Return the average loss
    return float(np.mean(loss))