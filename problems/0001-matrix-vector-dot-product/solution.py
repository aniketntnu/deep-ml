def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	A = a

    n = len(A)                        # number of rows in matrix A
    m = len(A[0])                     # number of columns in matrix A

    if m != len(b):                   # check dimension compatibility
        return -1                     # return -1 if dimensions are incompatible

    result = [0 for _ in range(n)]    # initialize result vector with zeros

    for i in range(n):                # loop over each row of the matrix
        for j in range(m):            # loop over each column / vector element
            result[i] += A[i][j] * b[j]  # multiply and accumulate

    return result                     # return the resulting vector
