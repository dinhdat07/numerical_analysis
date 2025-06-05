import numpy as np

def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            raise ValueError("Matrix is not diagonally dominant!")

    for iteration in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(i)) + sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x

    raise ValueError("Gauss-Seidel did not converge after {} iterations!".format(max_iter))
