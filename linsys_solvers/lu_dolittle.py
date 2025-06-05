import numpy as np

import numpy as np

def doolittle_lu_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        # calculate i-th row of U
        for j in range(i, n):
            if i == 0:
                sum_k = 0
            else:
                sum_k = np.dot(L[i, :i], U[:i, j])
            U[i, j] = A[i, j] - sum_k

        # calculate i-th column of L
        for j in range(i, n):
            if i == j:
                L[i, i] = 1.0
            else:
                if U[i, i] == 0:
                    raise ValueError("Zero pivot encountered!")
                if i == 0:
                    sum_k = 0
                else:
                    sum_k = np.dot(L[j, :i], U[:i, i])
                L[j, i] = (A[j, i] - sum_k) / U[i, i]

    return L, U


def solve_lu(A, b):
    L, U = doolittle_lu_decomposition(A)
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(U, y)
    return x
