import math

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    
    # Step 1: Extract elements
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Step 2: Compute trace and determinant
    trace = a + d
    det = a * d - b * c
    
    # Step 3: Compute discriminant
    # λ^2 - trace*λ + det = 0
    disc = trace**2 - 4 * det
    
    # Step 4: Compute eigenvalues using quadratic formula
    sqrt_disc = math.sqrt(disc)
    
    lambda1 = (trace + sqrt_disc) / 2
    lambda2 = (trace - sqrt_disc) / 2
    
    # Step 5: Sort from highest to lowest
    eigenvalues = sorted([lambda1, lambda2], reverse=True)
    
    return eigenvalues