import numpy as np
import matplotlib.pyplot as plt

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'linsys_solvers')))
# from gaussian_elimination import gaussian_elimination # type: ignore


def least_squares_poly_fit(x_data, y_data, degree):
    A = np.vander(x_data, degree + 1, increasing=True)  
    ATA = A.T @ A
    ATy = A.T @ y_data
    coef = np.linalg.solve(ATA, ATy)
    # coef = gaussian_elimination(ATA, ATy)
    return coef 

def evaluate_poly(coef, x):
    return sum(c * x**i for i, c in enumerate(coef))

def plot_least_squares_fit(x_data, y_data, degree):
    coef = least_squares_poly_fit(x_data, y_data, degree)
    x_dense = np.linspace(min(x_data), max(x_data), 500)
    y_dense = evaluate_poly(coef, x_dense)

    y_fit = evaluate_poly(coef, x_data)
    residual = np.linalg.norm(y_data - y_fit)

    plt.figure(figsize=(10, 5))
    plt.scatter(x_data, y_data, color='red', label='Data Points')
    plt.plot(x_dense, y_dense, label=f'Poly Degree {degree}', color='blue')
    plt.title(f"Least Squares Fit (Degree {degree})\nResidual: {residual:.2e}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return coef, residual

if __name__ == "__main__":
    # more complex data
    np.random.seed(0)
    x_data = np.linspace(0, 5, 20)
    y_true = 0.5 * x_data**3 - 2 * x_data**2 + x_data + 3
    noise = np.random.normal(0, 2.0, size=x_data.shape)
    y_data = y_true + noise

    results = {}
    for deg in [1, 2, 3, 5, 7, 9, 10, 12, 15, 20]:
        coef, residual = plot_least_squares_fit(x_data, y_data, degree=deg)
        results[deg] = (coef, residual)
        print(f"Degree {deg} Coefficients: {coef}, Residual: {residual:.2e}")

    results







