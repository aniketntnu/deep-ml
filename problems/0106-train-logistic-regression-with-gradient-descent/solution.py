import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression,
	optimizing parameters with Binary Cross Entropy loss.
	"""

	# X shape: (m, n)
	# y shape: (m,)
	m, n = X.shape                          # m = number of samples, n = number of features

	# prepend bias column (intercept)
	# before: X -> (m, n)
	# after:  X -> (m, n+1)
	X = np.hstack((np.ones((m, 1)), X))

	# reshape y into column vector
	# before: y -> (m,)
	# after:  y -> (m, 1)
	y = y.reshape(-1, 1)

	# initialize weights (including bias)
	# w shape: (n+1, 1)
	w = np.zeros((n + 1, 1))

	losses = []                             # list to store scalar loss values

	for _ in range(iterations):

		# linear model
		# X shape: (m, n+1)
		# w shape: (n+1, 1)
		# z shape: (m, 1)
		z = X @ w

		# sigmoid activation
		# z shape: (m, 1)
		# y_hat shape: (m, 1)
		y_hat = 1 / (1 + np.exp(-z))

		# B