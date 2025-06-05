import numpy as np

def jacobi(A, b, x0=None, tol=1e-6, max_iter=1000):
    n = len(A)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    x_new = np.zeros(n)

    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            raise ValueError("Matrix is not diagonally dominant!")

    for iteration in range(max_iter):
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new

        x = x_new.copy()

    raise ValueError("Jacobi did not converge after {} iterations!".format(max_iter))
