import numpy as np
import matplotlib.pyplot as plt

def divided_difference(x, y):
    n = len(x)
    coef = np.array(y, dtype=float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def newton_general_eval(x_data, coef, x):
    n = len(coef)
    result = coef[0]
    for i in range(1, n):
        term = coef[i]
        for j in range(i):
            term *= (x - x_data[j])
        result += term
    return result

def target_function(x):
    return np.exp(-x) * np.sin(2 * np.pi * x)

def plot_newton_general(a, b, num_points=8, resolution=1000, kind='nonuniform'):
    if kind == 'nonuniform':
        x_nodes = np.sort(np.random.uniform(a, b, num_points))
    else:
        x_nodes = np.linspace(a, b, num_points)

    y_nodes = target_function(x_nodes)
    coef = divided_difference(x_nodes, y_nodes)

    interp_func = lambda x: newton_general_eval(x_nodes, coef, x)

    x_dense = np.linspace(a, b, resolution)
    y_true = target_function(x_dense)
    y_interp = np.array([interp_func(x) for x in x_dense])
    error = np.abs(y_true - y_interp)

    # plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_dense, y_true, label="True Function", color="black")
    plt.plot(x_dense, y_interp, '--', label="Newton (General)", color="green")
    plt.scatter(x_nodes, y_nodes, color="red", label="Interpolation Nodes", zorder=5)
    plt.title("Newton General Interpolation vs True Function")
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
    plot_newton_general(0, 2, num_points=10, kind='nonuniform')
    plot_newton_general(0, 2, num_points=10, kind='uniform')
