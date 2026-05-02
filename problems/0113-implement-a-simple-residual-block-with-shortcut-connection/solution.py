import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	
	out = np.dot(w1,x)
	out = np.maximum(0, out)

	out = np.dot(w2,out)+x

	out = np.maximum(0, out)

	return out