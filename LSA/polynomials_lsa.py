import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Define target function f(x)
def f(x):
    return np.exp(x)  # Change this as needed

# Construct matrix A and vector B for the system A * a = B
def construct_normal_equation(f, degree, a, b):
    A = np.zeros((degree+1, degree+1))
    B = np.zeros(degree+1)

    for j in range(degree+1):
        for k in range(degree+1):
            A[j, k] = quad(lambda x: x**(j + k), a, b)[0]
        B[j] = quad(lambda x: x**j * f(x), a, b)[0]

    return A, B

# Solve the system to get coefficients a0, a1, ..., an
def continuous_lsq_monomial(f, degree, a, b):
    A, B = construct_normal_equation(f, degree, a, b)
    coeffs = np.linalg.solve(A, B)
    return coeffs

# Evaluate the polynomial at x
def evaluate_poly(coeffs, x):
    return sum(c * x**i for i, c in enumerate(coeffs))

# Plot
def plot_result(f, coeffs, a, b):
    x_vals = np.linspace(a, b, 300)
    y_true = [f(x) for x in x_vals]
    y_approx = [evaluate_poly(coeffs, x) for x in x_vals]

    plt.plot(x_vals, y_true, label="f(x)", color="red")
    plt.plot(x_vals, y_approx, label="P_n(x)", color="blue", linestyle="--")
    plt.title("Continuous Least Squares Approximation by Polynomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
a, b = 0, 1
degree = 3
coeffs = continuous_lsq_monomial(f, degree, a, b)

print("Coefficients of P_n(x):")
for i, c in enumerate(coeffs):
    print(f"a_{i} = {c:.6f}")

plot_result(f, coeffs, a, b)
