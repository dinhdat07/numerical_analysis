import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import math

from models.ode_system import ODESystem
from solver.rk4 import RK4Solver

# ==== Shooting Method ====

def shooting_method(f_system, a, b, alpha, beta, s_guess1, s_guess2, h=0.1):
    def integrate(s):
        def wrapped_f(x, Y):
            y1, y2 = Y
            dy1 = y2
            dy2 = f_system(x, y1, y2)
            return np.array([dy1, dy2])

        ode_system = ODESystem(f_vec=wrapped_f,
            y0=np.array([alpha, s]),
            x0=a,
            h=h,
            x_end=b,
            label="Shooting Method ODE System"
        )
        rk_solver = RK4Solver(ode_system)
        n_steps = int((b - a) / h)
        result = rk_solver.solve(n_steps)
        xs = [pt[0] for pt in result]
        ys = [pt[1][0] for pt in result]  # y values
        return ys[-1], xs, ys

    def objective(s):
        yb, _, _ = integrate(s)
        return yb - beta

    sol = root_scalar(objective, bracket=[s_guess1, s_guess2], method='bisect')
    s_correct = sol.root
    _, xs, ys = integrate(s_correct)
    return xs, ys, s_correct

# ==== Finite Difference Method ====
def finite_difference_method(p_func, q_func, r_func, a, b, alpha, beta, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)

    A = np.zeros((N-1, N-1))
    b_vec = np.zeros(N-1)

    for i in range(1, N):
        xi = x[i]
        pi = p_func(xi)
        qi = q_func(xi)
        ri = r_func(xi)

        A[i-1][i-1] = -2 / h**2 + qi
        if i > 1:
            A[i-1][i-2] = 1 / h**2 - pi / (2*h)
        if i < N-1:
            A[i-1][i] = 1 / h**2 + pi / (2*h)
        b_vec[i-1] = -ri

    b_vec[0] -= alpha * (1 / h**2 - p_func(x[1]) / (2*h))
    b_vec[-1] -= beta * (1 / h**2 + p_func(x[N-1]) / (2*h))

    y_inner = np.linalg.solve(A, b_vec)
    y_full = np.concatenate(([alpha], y_inner, [beta]))
    return x, y_full

# ==== Main Function ====
def main():
    # BVP: y'' + 2xy' + y = x^2 + 1, y(0) = 0, y(1) = 1
    f = lambda x, y, y_prime: x**2 + 1 - 2*x*y_prime - y
    p_func = lambda x: 2*x
    q_func = lambda x: 1
    r_func = lambda x: -(x**2 + 1)

    a, b = 0, 1
    alpha, beta = 0, 1
    h = 0.05

    print("\n=== SHOOTING METHOD (RK4) ===")
    xs1, ys1, s = shooting_method(f, a, b, alpha, beta, s_guess1=0.0, s_guess2=2.0, h=h)
    print(f"Slope ban đầu đúng (y'(0)) ≈ {s:.6f}")

    print("\n=== FINITE DIFFERENCE METHOD ===")
    xs2, ys2 = finite_difference_method(p_func, q_func, r_func, a, b, alpha, beta, N=len(xs1) - 1)

    plt.plot(xs1, ys1, 'r-', label="Shooting Method (RK4)")
    plt.plot(xs2, ys2, 'b--', label="Finite Difference Method")
    plt.legend()
    plt.title("So sánh các phương pháp giải BVP phức tạp")
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()