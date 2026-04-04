import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here

	mean = np.mean(X,axis=-1,keepdims=True)
	variance = np.var(X,axis=-1,keepdims=True)

	x_norm = (X-mean)/np.sqrt(variance+epsilon)

	x_norm = gamma * x_norm+ beta

	return x_norm 
