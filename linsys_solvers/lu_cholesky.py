import numpy as np

def cholesky_decomposition(A):
    if not np.allclose(A, A.T, atol=1e-8):
        raise ValueError("Matrix is not symmetric!")
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix is not positive definite!")

    n = A.shape[0]
    L = np.zeros_like(A)

    for i in range(n):
        for j in range(i + 1):
            if j == 0:
                sum_k = 0
            else:
                sum_k = np.dot(L[i, :j], L[j, :j])
            
            if i == j:
                val = A[i, i] - sum_k
                if val <= 0:
                    raise ValueError("Matrix is not positive definite!")
                L[i, j] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]
    return L


def solve_cholesky(A, b):
    L = cholesky_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x
