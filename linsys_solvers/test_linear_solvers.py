import numpy as np
import scipy.sparse as sp
import time

from gaussian_elimination import gaussian_elimination
from jacobi import jacobi
from gauss_seidel import gauss_seidel
from lu_cholesky import solve_cholesky
from lu_dolittle import solve_lu

def test_solver(solver_name, solver_func, A, b, epsilon=1e-4):
    print(f"\n[TEST] {solver_name} - size: {A.shape[0]}")
    try:
        start = time.time()
        x = solver_func(A.copy(), b.copy())
        elapsed = time.time() - start
        residual = np.linalg.norm(A @ x - b, np.inf)
        print(f"> Done in {elapsed:.2f}s, Residual: {residual:.2e}, Accurate: {residual < epsilon}")
    except Exception as e:
        print(f"> Failed: {e}")

def generate_test_matrix(n, kind="spd", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if kind == "spd":
        # Symmetric Positive Definite (Cholesky-friendly)
        A = np.random.randn(n, n)
        A = A @ A.T               # A = A * A^T ⇒ symmetric
        A += n * np.eye(n)        # Make it positive definite (diagonal dominance)
    elif kind == "dd":
        # Strictly Diagonally Dominant (Jacobi / Gauss-Seidel friendly)
        A = np.random.uniform(-1, 1, (n, n))
        for i in range(n):
            A[i, i] = np.sum(np.abs(A[i])) + np.random.uniform(0.5, 1.5)
    else:
        # Random general matrix (Gaussian / LU)
        A = np.random.uniform(-1, 1, (n, n))

    b = np.random.randn(n)
    return A, b

sizes = [1000, 5000, 10000]

for n in sizes:
    print(f"\n======= Testing size {n} =======")
    
    # Gaussian + LU
    A_gauss, b_gauss = generate_test_matrix(n)
    test_solver("Gaussian Elimination", gaussian_elimination, A_gauss, b_gauss)
    # test_solver("LU (Doolittle)", solve_lu, A_gauss, b_gauss)

    # Cholesky
    A_chol, b_chol = generate_test_matrix(n, kind="spd")
    test_solver("Cholesky", solve_cholesky, A_chol, b_chol)

    # Jacobi + Gauss-Seidel
    A_iter, b_iter = generate_test_matrix(n, kind="dd")
    test_solver("Jacobi", jacobi, A_iter, b_iter)
    test_solver("Gauss-Seidel", gauss_seidel, A_iter, b_iter)
