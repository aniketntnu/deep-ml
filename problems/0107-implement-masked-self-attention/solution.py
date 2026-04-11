import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""

	# Step 1: Get dimension for scaling
	d_k = K.shape[1]

	# Step 2: Compute attention scores (Q * K^T / sqrt(d_k))
	scores = np.dot(Q, K.T) / np.sqrt(d_k)

	# Step 3: Apply mask
	# mask has 0 for allowed and -inf for blocked positions
	scores = scores + mask

	# Step 4: Apply softmax row-wise (with numerical stability)
	scores = scores - np.max(scores, axis=1, keepdims=True)
	exp_scores = np.exp(scores)
	attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

	# Step 5: Compute output (attention_weights * V)
	output = np.dot(attention_weights, V)

	return output