import numpy as np

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h/2 * (y[0] + 2*np.sum(y[1:-1]) + y[-1])

def midpoint_rule(f, a, b, n):
    h = (b - a) / (n + 2)
    midpoints = np.linspace(a + h, b - h, n + 1)
    return 2 * h * np.sum(f(midpoints[0::2]))

def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("Simpson's rule requires n to be even.")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h/3 * (y[0] + 2 * np.sum(y[2:n:2]) + 4 * np.sum(y[1:n:2]) + y[n])

if __name__ == "__main__":
    f = lambda x: np.exp(-x) * np.sin(2 * np.pi * x)
    a, b = 0, 2
    n = 50

    trap = trapezoidal_rule(f, a, b, n)
    mid = midpoint_rule(f, a, b, n)
    simp = simpson_rule(f, a, b, n if n % 2 == 0 else n+1)

    print("Trapezoidal Rule:", trap)
    print("Midpoint Rule:", mid)
    print("Simpson's Rule:", simp)
