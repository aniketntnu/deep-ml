import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    """
    Compute log-softmax for a 1D list/array of scores.
    """

    # Convert the input to a NumPy array
    scores = np.array(scores)

    # Find the maximum score for numerical stability
    max_score = np.max(scores)

    # Compute log(sum(exp(scores)))
    # Subtracting max_score prevents exponential overflow
    log_sum_exp = max_score + np.log(
        np.sum(np.exp(scores - max_score))
    )

    # log-softmax = score - log(sum(exp(scores)))
    return scores - log_sum_exp