import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here

    # prediction
    probabilities = []

    m = len(features) # samples row
    n = len(weights) # no of columns 

    # current pred 

    for i in range(m):
        z = 0
        for j in range(n):
            z+= features[i][j]* weights[j]

        z+=bias

        prob = 1 / (1+math.exp(-z))

        probabilities.append(prob)

    mse =0.0

    for i in range(m):
        diff= probabilities[i]-labels[i]
        mse+=diff*diff
    mse/=m

    # rounding
    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)

	return probabilities, mse