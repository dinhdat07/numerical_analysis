import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# target function to approximate
def f(x):
    return np.exp(-x) * np.sin(2 * np.pi * x)


def inner_product(f1, f2, a, b):
    result, _ = quad(lambda x: f1(x) * f2(x), a, b)
    return result

def least_squares_continuous(f, a, b, degree):
    # phi_i(x) = x^i
    phi = [lambda x, i=i: x**i for i in range(degree + 1)]
    A = np.zeros((degree + 1, degree + 1))
    b_vec = np.zeros(degree + 1)

    # construct the system of equations
    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i, j] = inner_product(phi[i], phi[j], a, b)
        b_vec[i] = inner_product(f, phi[i], a, b)

    coef = np.linalg.solve(A, b_vec)
    return coef

def evaluate_polynomial(coef, x):
    return sum(c * x**i for i, c in enumerate(coef))

def plot_approximation(f, a, b, degree):
    coef = least_squares_continuous(f, a, b, degree)
    x_vals = np.linspace(a, b, 500)
    y_true = f(x_vals)
    y_approx = [evaluate_polynomial(coef, x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_true, label="True Function", color="black")
    plt.plot(x_vals, y_approx, '--', label=f"Approx. Polynomial deg {degree}", color="blue")
    plt.title("Continuous Least Squares Polynomial Approximation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # compute L2 error
    error_integral, _ = quad(lambda x: (f(x) - evaluate_polynomial(coef, x))**2, a, b)
    print(f"L2 error over [{a}, {b}] for degree {degree}: {error_integral:.2e}")

if __name__ == "__main__":
    a, b = 0, 1
    for deg in [2, 4, 6, 8]:
        plot_approximation(f, a, b, deg)
