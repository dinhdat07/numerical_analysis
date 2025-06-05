import sympy as sp
import numpy as np

def false_point(f, a, b, tol=1e-10, max_iter=100, verbose=False):
    x_sym = sp.Symbol('x')
    if isinstance(f, sp.Expr):
        f_func = sp.lambdify(x_sym, f, modules='numpy')
    elif callable(f):
        f_func = f
    else:
        raise TypeError("f must be a sympy.Expr or a callable function.")

    fa = f_func(a)
    fb = f_func(b)

    if fa * fb > 0:
        raise ValueError(f"f(a) and f(b) must have opposite signs: f({a}) = {fa}, f({b}) = {fb}")

    # starting iteration
    for i in range(1, max_iter + 1):
        c = b - fb * (b - a) / (fb - fa)
        fc = f_func(c)

        if verbose:
            print(f"Iter {i}: a = {a:.12f}, b = {b:.12f}, c = {c:.12f}, f(c) = {fc:.2e}, error = {abs(fc):.2e}")

        if abs(fc) < tol:
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise RuntimeError("False Position method did not converge within the maximum number of iterations.")


if __name__ == "__main__":
    x = sp.Symbol('x')
    f_expr = x * sp.tan(x) - 1 

    a = 0.5
    b = 1.2

    try:
        root = false_point(
            f=f_expr,
            a=a,
            b=b,
            tol=1e-12,
            max_iter=100,
            verbose=True
        )
        print(f"\n Root found: x = {root:.12f}")
    except Exception as e:
        print(f"\n Failed: {e}")
