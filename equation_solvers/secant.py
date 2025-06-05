import sympy as sp
import numpy as np

def secant_method(f, x0, x1, tol=1e-10, max_iter=100, verbose=False):
    x_sym = sp.Symbol('x')

    if isinstance(f, sp.Expr):
        f_func = sp.lambdify(x_sym, f, modules='numpy')
    elif callable(f):
        f_func = f
    else:
        raise TypeError("f must be a sympy.Expr or a callable function.")

    # starting iteration
    for i in range(1, max_iter + 1):
        f_x0 = f_func(x0)
        f_x1 = f_func(x1)

        if abs(f_x1 - f_x0) < 1e-14:
            raise ZeroDivisionError(f"Denominator too small at iteration {i}: f(x1) - f(x0) ≈ 0.")

        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)

        if verbose:
            print(f"Iter {i}: x0 = {x0:.12f}, x1 = {x1:.12f}, x2 = {x2:.12f}, error = {abs(x2 - x1):.2e}")

        if abs(x2 - x1) < tol:
            return x2

        x0, x1 = x1, x2

    raise RuntimeError("Secant method did not converge within the maximum number of iterations.")


if __name__ == "__main__":
    x = sp.Symbol('x')
    f_expr = x * sp.tan(x) - 1  

    x0 = 0.5   
    x1 = 1.0   

    try:
        root = secant_method(
            f=f_expr,
            x0=x0,
            x1=x1,
            tol=1e-12,
            max_iter=100,
            verbose=True
        )
        print(f"\n Root found: x = {root:.12f}")
    except Exception as e:
        print(f"\n Failed: {e}")
