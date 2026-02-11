def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    

    r,c = len(a),len(a[0])

    t = [[None for i in range(r)] for j in range(c) ]

    #print(len(t),len(t[0]))

    for i in range(c):
        for j in range(r):
            ele = a[j][i]
            #print("\n i:",i," j:",j," \n")
            #print("\n  ele:",ele," \n")
            
            t[i][j]=ele

    #print(t)

    return t













