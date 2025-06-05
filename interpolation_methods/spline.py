import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(x, y):
    n = len(x) - 1
    h = np.diff(x)

    # setup system Ax = b
    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    A[0, 0] = 1
    A[n, n] = 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, b_vec)

    # calculate a, b, c, d coefficients
    a = y[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    # coefficients of splines on each interval
    return a, b, c[:-1], d, x

def eval_spline(a, b, c, d, x_nodes, x_eval):
    x_eval = np.atleast_1d(x_eval)
    result = np.zeros_like(x_eval)

    for i in range(len(x_nodes) - 1):
        idx = (x_eval >= x_nodes[i]) & (x_eval <= x_nodes[i + 1])
        dx = x_eval[idx] - x_nodes[i]
        result[idx] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return result

def target_function(x):
    return np.exp(-x) * np.sin(2 * np.pi * x)

def plot_spline(a, b, c, d, x_nodes):
    x_dense = np.linspace(x_nodes[0], x_nodes[-1], 1000)
    y_true = target_function(x_dense)
    y_spline = eval_spline(a, b, c, d, x_nodes, x_dense)

    error = np.abs(y_true - y_spline)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(x_dense, y_true, label="True Function", color="black")
    plt.plot(x_dense, y_spline, '--', label="Cubic Spline", color="blue")
    plt.scatter(x_nodes, target_function(x_nodes), color="red", label="Interpolation Nodes")
    plt.title("Cubic Spline Interpolation vs True Function")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_dense, error, color="purple", label="Absolute Error")
    plt.title("Interpolation Error")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x_nodes = np.linspace(0, 2, 9)
    y_nodes = target_function(x_nodes)
    a, b, c, d, x_nodes = cubic_spline(x_nodes, y_nodes)
    plot_spline(a, b, c, d, x_nodes)
