import numpy as np
import sympy as sp

def fixed_point(g, x0, tol=1e-10, max_iter=1000, verbose=False, check_convergence=False, interval=None):
    x_sym = sp.Symbol('x')

    # Check convergence
    if isinstance(g, sp.Expr):
        g_expr = g
        g_func = sp.lambdify(x_sym, g_expr, modules='numpy')
        if check_convergence:
            print("Checking convergence conditions...")
            if interval is None:
                raise ValueError("You must provide interval=(a, b) to check convergence.")
            a, b = interval
            dg_expr = sp.diff(g_expr, x_sym)
            dg_func = sp.lambdify(x_sym, dg_expr, modules='numpy')
            xs = np.linspace(a, b, 1000)
            for xi in xs:
                gx = g_func(xi)
                print(f"g({xi:.6f}) = {gx:.6f}, g'({xi:.6f}) = {dg_func(xi):.6f}")
                if gx < a or gx > b:
                    raise ValueError(f"g(x) = {gx:.6f} not in [{a}, {b}] at x = {xi:.6f}")
                if abs(dg_func(xi)) >= 1 - 1e-5:
                    raise ValueError(f"|g'(x)| = {abs(dg_func(xi)):.6f} ≥ 1 at x = {xi:.6f}")
        g = g_func 

    elif not callable(g):
        raise TypeError("g must be a sympy.Expr or a callable function.")

    # Start iteration
    x = x0
    for i in range(1, max_iter + 1):
        x_next = g(x)
        if verbose:
            print(f"Iter {i}: x = {x:.12f}, g(x) = {x_next:.12f}, error = {abs(x_next - x):.2e}")
        if abs(x_next - x) < tol:
            return x_next
        x = x_next

    raise RuntimeError("Fixed-point iteration did not converge within the maximum number of iterations.")


if __name__ == "__main__":
    x = sp.Symbol('x')
    g_expr = sp.cos(x)  

    x0 = -1
    interval = (-1, 1)

    print("Running Fixed-Point Iteration for g(x) = cos(x)")
    try:
        root = fixed_point(
            g=g_expr,
            x0=x0,
            tol=1e-10,
            max_iter=1000,
            verbose=True,
            check_convergence=True,
            interval=interval
        )
        print(f"\Fixed point found: x = {root:.12f}")
    except Exception as e:
        print(f"\nFailed: {e}")
