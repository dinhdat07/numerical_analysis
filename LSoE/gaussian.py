import numpy as np
import scipy.sparse as sp

epsilon = 1e-4

# Gaussian elimination 1000
def gaussian_elimination(A, b):
    n = len(A)
    Ab = np.hstack((A, b.reshape(-1,1))) 
     
    for i in range(n):
        if Ab[i, i] == 0:
            raise ValueError("Pivot = 0!")
        
        for j in range(i+1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]

    return x

# LU decomposition (Cholesky) 1000
def cholesky_decomposition(A):
    if not np.allclose(A, A.T):
        raise ValueError("Matrix is not symmetric!")
    if np.any(np.linalg.eigvals(A) <= 0):
        raise ValueError("Matrix is not positive definite!")

    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            sum_k = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - sum_k)
            else:
                L[i][j] = (A[i][j] - sum_k) / L[j][j]
    return L

def solve_cholesky(A, b):
    L = cholesky_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)

    return x

# LU decomposition (Doolittle) 1000
def doolittle_lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sum_k = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_k
        for j in range(i, n):
            if i == j:
                L[i][i] = 1
            else:
                sum_k = sum(L[j][k] * U[k][i] for k in range(i))
                if U[i][i] == 0:
                    raise ValueError("Main diagonal element of U is zero, pivoting is required!")
                L[j][i] = (A[j][i] - sum_k) / U[i][i]
    return L, U

def solve_lu(A, b):
    L, U = doolittle_lu_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x

# Jacobi iteration 10000
def jacobi(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)
    x_new = np.zeros(n)

    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            raise ValueError("Matrix is not diagonally dominant, Jacobi may not converge!")

    for iteration in range(max_iter):
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x_new - x, np.inf) < tol:
            return x_new

        x = x_new.copy()

    raise ValueError("Jacobi did not converge after {} iterations!".format(max_iter))

# Gauss-Seidel iteration 10000
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(A)
    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float)

    for i in range(n):
        if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(n) if j != i):
            raise ValueError("Matrix is not diagonally dominant, Gauss-Seidel may not converge!")

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(i)) + sum(A[i, j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sigma) / A[i, i]
        
        progress = (iteration + 1) / max_iter * 100
        print(f"Progress: {progress:.2f}%")

        if np.linalg.norm(x - x_old, np.inf) < tol:
            return x

    raise ValueError("Gauss-Seidel did not converge after {} iterations!".format(max_iter))

# Test
A = np.array([[10, -1, 2], 
              [-1, 11, -1], 
              [2, -1, 10]], dtype=float)

b = np.array([6, 25, -11], dtype=float)

# solution = gaussian_elimination(A, b)
# print("Solution:", solution)
# b_check = np.dot(A, solution)
# print("Check: ", b_check - b < epsilon)

# solution = solve_cholesky(A, b)
# print("Solution:", solution)
# b_check = np.dot(A, solution)
# print("Check: ", b_check - b < epsilon)

# solution = solve_lu(A, b)
# print("Solution:", solution)
# b_check = np.dot(A, solution)
# print("Check: ", b_check - b < epsilon)

# solution = jacobi(A, b)
# print("Solution:", solution)
# b_check = np.dot(A, solution)
# print("Check: ", b_check - b < epsilon)

# solution = gauss_seidel(A, b)
# print("Solution:", solution)
# b_check = np.dot(A, solution)
# print("Check: ", b_check - b < epsilon)

n = 5000
diag = 3 * np.ones(n)  # Increase diagonal dominance
off_diag = -1 * np.ones(n - 1)
A = sp.diags([off_diag, diag, off_diag], [-1, 0, 1], shape=(n, n)).toarray()
b = np.random.rand(n)

# Solve using Gauss-Seidel
solution = gaussian_elimination(A, b)
print("Solution found:")
print(solution)
b_check = A @ solution
print("Check: ", np.allclose(b, b_check, atol=epsilon))
