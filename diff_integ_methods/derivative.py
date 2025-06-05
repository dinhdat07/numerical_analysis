import numpy as np


# ==== THREE-POINT =====
def forward_diff_3pt(f, x, h):
    return (-3*f(x) + 4*f(x + h) - f(x + 2*h)) / (2*h)

def backward_diff_3pt(f, x, h):
    return (3*f(x) - 4*f(x - h) + f(x - 2*h)) / (2*h)

def central_diff_3pt(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)

# ==== FIVE-POINT =====
def forward_diff_5pt(f, x, h):
    return (-25*f(x) + 48*f(x + h) - 36*f(x + 2*h) + 16*f(x + 3*h) - 3*f(x + 4*h)) / (12*h)

def backward_diff_5pt(f, x, h):
    return (25*f(x) - 48*f(x - h) + 36*f(x - 2*h) - 16*f(x - 3*h) + 3*f(x - 4*h)) / (12*h)

def central_diff_5pt(f, x, h):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)

# ==== SECOND DERIVATIVE ====
def second_derivative_3pt(f, x, h):
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

# ==== FINITE DIFFERENCE ====
def forward_diff(f, x, h):
    return (f(x + h) - f(x)) / h

def backward_diff(f, x, h):
    return (f(x) - f(x - h)) / h

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2*h)


if __name__ == "__main__":
    f = lambda x: np.exp(-x) * np.sin(2*np.pi*x)
    df_exact = lambda x: np.exp(-x) * (2*np.pi * np.cos(2*np.pi*x) - np.sin(2*np.pi*x))
    d2f_exact = lambda x: np.exp(-x) * ((1 - 4*np.pi**2)*np.sin(2*np.pi*x) - 4*np.pi * np.cos(2*np.pi*x))

    x0 = 0.5
    h = 0.01

    print("=== Derivative at x = 0.5 ===")
    print(f"Exact f'(x): {df_exact(x0):.6f}")
    print("=== Numerical Derivatives ===")
    print(f"3pt Forward: {forward_diff_3pt(f, x0, h):.6f}")
    print(f"3pt Backward: {backward_diff_3pt(f, x0, h):.6f}")
    print(f"3pt Central: {central_diff_3pt(f, x0, h):.6f}")
    print("---")
    print(f"5pt Forward: {forward_diff_5pt(f, x0, h):.6f}")
    print(f"5pt Backward: {backward_diff_5pt(f, x0, h):.6f}")
    print(f"5pt Central: {central_diff_5pt(f, x0, h):.6f}")
    print("---")
    print(f"Finite Forward: {forward_diff(f, x0, h):.6f}")
    print(f"Finite Backward: {backward_diff(f, x0, h):.6f}")
    print(f"Finite Central: {central_diff(f, x0, h):.6f}")

    print("\n=== Second Derivative at x = 0.5 ===")
    print(f"Exact f''(x): {d2f_exact(x0):.6f}")
    print(f"3pt Central: {second_derivative_3pt(f, x0, h):.6f}")