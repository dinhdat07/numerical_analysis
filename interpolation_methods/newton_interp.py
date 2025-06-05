import numpy as np
import matplotlib.pyplot as plt

def forward_coefficients(x, y):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def backward_coefficients(x, y):
    n = len(x)
    coef = np.array(y, dtype=float)

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (x[i] - x[i - j])

    return coef


def newton_forward_eval(x_data, coef, x):
    n = len(coef)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term
    return result

def newton_backward_eval(x_data, coef, x):
    n = len(coef)
    result = coef[-1]
    for i in range(n - 2, -1, -1):
        term = coef[i]
        for j in range(n - 1, i, -1):
            term *= (x - x_data[j])
        result += term
    return result

def target_function(x):
    return np.exp(-x) * np.sin(2 * np.pi * x)

def plot_newton(a, b, num_points=8, resolution=1000, mode='forward'):
    x_nodes = np.linspace(a, b, num_points)
    y_nodes = target_function(x_nodes)

    if mode == 'forward':
        coef = forward_coefficients(x_nodes, y_nodes)
        interp_func = lambda x: newton_forward_eval(x_nodes, coef, x)
    elif mode == 'backward':
        coef = backward_coefficients(x_nodes, y_nodes)
        interp_func = lambda x: newton_backward_eval(x_nodes, coef, x)
    else:
        raise ValueError("Mode must be 'forward' or 'backward'")

    x_dense = np.linspace(a, b, resolution)
    y_true = target_function(x_dense)
    y_interp = np.array([interp_func(x) for x in x_dense])
    error = np.abs(y_true - y_interp)

    # plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_dense, y_true, label="True Function", color="black")
    plt.plot(x_dense, y_interp, '--', label=f"Newton {mode.capitalize()}", color="blue")
    plt.scatter(x_nodes, y_nodes, color="red", label="Interpolation Nodes", zorder=5)
    plt.title(f"Newton {mode.capitalize()} Interpolation vs True Function")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_dense, error, label="Absolute Error", color="purple")
    plt.title("Interpolation Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_newton(0, 2, num_points=8, mode='forward')
    plot_newton(0, 2, num_points=8, mode='backward')
