import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
    
    points = np.array(points, dtype=float)
    # convert list of points into numpy array of shape (n_points, dimensions)
    
    centroids = np.array(initial_centroids, dtype=float)
    # initialize centroids as numpy array of shape (k, dimensions)
    
    for _ in range(max_iterations):
        # repeat clustering process for maximum given iterations
        
        distances = np.array([np.linalg.norm(points - c, axis=1) for c in centroids]).T
        # compute distance from every point to every centroid
        # points - c subtracts centroid from all points
        # norm(axis=1) gives distance from centroid c to each point
        # doing this for all centroids gives shape (k, n_points)
        # transpose converts to (n_points, k)
        
        labels = np.argmin(distances, axis=1)
        # ass