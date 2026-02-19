import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    """

    # Step 1: get number of samples (2N) and embedding dimension d
    n, d = embeddings.shape
    # n = total number of embeddings, d = embedding dimension

    # Step 2: normalize embeddings to unit length (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # compute L2 norm of each embedding vector, shape (n, 1)

    embeddings = embeddings / norms
    # divide each embedding by its norm so cosine similarity = dot product

    # Step 3: compute cosine similarity matrix between all pairs
    sim_matrix = np.dot(embeddings, embeddings.T)
    # sim_matrix[i][j] = cosine similarity between embedding i and j, shape (n, n)

    # Step 4: scale similarities by temperature
    sim_matrix = sim_matrix / temperature
    # divide similarity values by temperature tau

    # Step 5: initialize tot