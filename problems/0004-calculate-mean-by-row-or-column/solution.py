def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    
    d = dict()
    r, c = len(matrix), len(matrix[0])
    means = []

    if mode == "column":
        
        for i in range(r):
            for j in range(c):
                if j in d:
                    d[j].append(matrix[i][j])
                else:
                    d[j] = [matrix[i][j]]

        for key in d:
            means.append(sum(d[key]) / len(d[key]))

    else:  # row mode
        for i in range(r):
            means.append(sum(matrix[i]) / len(matrix[i]))

    return means
