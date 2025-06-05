import numpy as np

def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).flatten()

    n = len(A)
    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        raise ValueError("Invalid input dimensions")

    Ab = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        if abs(Ab[i, i]) < 1e-12:
            # partial pivoting
            for k in range(i + 1, n):
                if abs(Ab[k, i]) > 1e-12:
                    Ab[[i, k]] = Ab[[k, i]]
                    break
            else:
                raise ValueError(f"Matrix is singular at pivot {i}")

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = 0.0
        for j in range(i + 1, n):
            s += Ab[i, j] * x[j]
        x[i] = (Ab[i, -1] - s) / Ab[i, i]

    return x


if __name__ == "__main__":
    A = np.array([
        [2, 1, -1, 2],
        [4, 5, -3, 6],
        [-2, 5, -2, 6],
        [4, 11, -4, 8]
    ], dtype=float)
    b = np.array([5, 9, 4, 2], dtype=float)

    print("Solving system using Gaussian Elimination...")
    try:
        x = gaussian_elimination(A, b)
        print("Solution vector x:")
        print(x)
        print("Check (Ax ≈ b):")
        print(np.dot(A, x))
    except Exception as e:
        print("Failed:", e)

