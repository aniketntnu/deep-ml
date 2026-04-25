import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
	
	# Step 1: initialize parameters
	x = x0.astype(float).copy()
	
	# First moment (mean of gradients)
	m = np.zeros_like(x)
	
	# Second moment (mean of squared gradients)
	v = np.zeros_like(x)
	
	for t in range(1, num_iterations + 1):
		
		# Step 2: compute gradient
		g = grad(x)
		
		# Step 3: update biased first moment
		m = beta1 * m + (1 - beta1) * g
		
		# Step 4: update biased second moment
		v = beta2 * v + (1 - beta2) * (g ** 2)
		
		# Step 5: bias correction
		m_hat = m / (1 - beta1 ** t)
		v_hat = v / (1 - beta2 ** t)
		
		# Step 6: update parameters
		x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
	
	return x