import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization (Batch / Stochastic / Mini-batch)
    """

    m, n = X.shape

    for _ in range(n_epochs):

        # ------------------ BATCH GD ------------------
        if method == 'batch':
            
            # Step 1: predictions for all samples
            y_pred = X @ weights
            
            # Step 2: error
            error = y_pred - y
            
            # Step 3: gradient (average over all samples)
            grad = (2/m) * (X.T @ error)
            
            # Step 4: update weights
            weights = weights - learning_rate * grad

        # ------------------ STOCHASTIC GD ------------------
        elif method == 'stochastic':
            
            for i in range(m):
                
                xi = X[i]                 # single sample (n,)
                yi = y[i]                 # scalar
                
                # Step 1: prediction
                y_pred = xi @ weights
                
                # Step 2: error
                error = y_pred - yi
                
                # Step 3: gradient
                grad = 2 * error * xi     # no averaging
                
                # Step 4: update
                weights = weights - learning_rate * grad

        # ------------------ MINI-BATCH GD ------------------
        elif method == 'mini_batch':
            
            for i in range(0, m, batch_size):
                
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Step 1: predictions
                y_pred = X_batch @ weights
                
                # Step 2: error
                error = y_pred - y_batch
                
                # Step 3: gradient (average over batch)
                grad = (2/len(X_batch)) * (X_batch.T @ error)
                
                # Step 4: update
                weights = weights - learning_rate * grad

    return weights