import numpy as np
import matplotlib.pyplot as plt

def lagrange_interpolation(x_points, y_points, x):
    n = len(x_points)
    result = 0.0
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        result += term
    return result

def target_function(x):
    return np.exp(-x) * np.sin(2 * np.pi * x)

def plot_lagrange_with_error(a, b, num_points=8, resolution=1000):
    x_nodes = np.linspace(a, b, num_points)
    y_nodes = target_function(x_nodes)

    x_dense = np.linspace(a, b, resolution)
    y_true = target_function(x_dense)
    y_interp = [lagrange_interpolation(x_nodes, y_nodes, x) for x in x_dense]
    error = np.abs(y_true - y_interp)

    # plot interpolation vs true function
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_dense, y_true, label="True Function", color="black")
    plt.plot(x_dense, y_interp, label="Lagrange Interpolation", color="blue", linestyle="--")
    plt.scatter(x_nodes, y_nodes, color="red", zorder=5, label="Interpolation Nodes")
    plt.title("Lagrange Interpolation vs True Function")
    plt.legend()
    plt.grid(True)

    # plot interpolation error
    plt.subplot(2, 1, 2)
    plt.plot(x_dense, error, label="Interpolation Error", color="purple")
    plt.xlabel("x")
    plt.ylabel("Absolute Error")
    plt.title("Interpolation Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_lagrange_with_error(a=0, b=2, num_points=10)
    # x_points = [1, 2, 3, 4]
    # y_points = [1, 4, 9, 16]  

    # x = 2.5
    # y_interp = lagrange_interpolation(x_points, y_points, x)
    # print(f"Interpolated value at x = {x} is approximately {y_interp:.4f}")

