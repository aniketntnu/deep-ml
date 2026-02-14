import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    weights = initial_weights.astype(float).copy()
    # copy initial weights so original weights are not modified and ensure float precision for gradient updates
    
    bias = float(initial_bias)
    # initialize bias separately as float so it can be updated using gradient descent
    
    mse_values = []
    # this list will store mean squared error for each epoch so we can track training progress
    
    m = features.shape[0]
    # number of training samples, used to average gradients and compute MSE
    
    for epoch in range(epochs):
        # loop over epochs, each epoch means one full pass over the entire dataset
        
        z = np.dot(features, weights) + bias
        # compute linear combination: z = w·x + b for all samples at once, shape becomes (m,)
        
     