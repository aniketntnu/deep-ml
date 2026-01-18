import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here

    scores = np.array(scores)

    x_max = np.max(scores)
    exp_x = np.exp(scores-x_max)

    softmax_x = exp_x / sum(exp_x)

    return [float(f"{val:.4f}") for val in softmax_x]