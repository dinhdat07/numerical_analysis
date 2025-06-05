import sympy as sp
import numpy as np

def newton_method(f, x0, tol=1e-10, max_iter=100, verbose=False, check_convergence=True):
    x_sym = sp.Symbol('x')

    # define f, f', f'' 
    if isinstance(f, sp.Expr):
        f_expr = f
        f_func = sp.lambdify(x_sym, f_expr, modules='numpy')
        df_expr = sp.diff(f_expr, x_sym)
        ddf_expr = sp.diff(df_expr, x_sym)
        df_func = sp.lambdify(x_sym, df_expr, modules='numpy')
        ddf_func = sp.lambdify(x_sym, ddf_expr, modules='numpy')
    else:
        raise TypeError("f must be a sympy.Expr")

    # check convergence conditions
    if check_convergence:
        f0 = f_func(x0)
        df0 = df_func(x0)
        ddf0 = ddf_func(x0)

        if df0 == 0:
            raise ZeroDivisionError(f"f'(x0) = 0 → Cannot apply Newton method with x0 = {x0}")

        if f0 * ddf0 <= 0:
            raise ValueError(
                f"Not sastify convergence conditions: f(x0) * f''(x0) = {f0 * ddf0:.6f} ≤ 0 at x0 = {x0}"
            )

    # start iteration
    x = x0
    for i in range(1, max_iter + 1):
        fx = f_func(x)
        dfx = df_func(x)

        if dfx == 0:
            raise ZeroDivisionError(f"f'(x) = 0 at x = {x:.12f}, cannot continue.")

        x_next = x - fx / dfx

        if verbose:
            print(f"Iter {i}: x = {x:.12f}, f(x) = {fx:.2e}, f'(x) = {dfx:.2e}, error = {abs(x_next - x):.2e}")

        if abs(x_next - x) < tol:
            return x_next

        x = x_next

    raise RuntimeError("Newton method did not converge within the maximum number of iterations.")


if __name__ == "__main__":
    x = sp.Symbol('x')
    f_expr = x * sp.tan(x) - 1
    x0 = 1.2

    print("Running Newton Method for f(x): ")
    try:
        root = newton_method(
            f=f_expr,
            x0=x0,
            tol=1e-12,
            max_iter=100,
            verbose=True,
            check_convergence=True
        )
        print(f"\n Root found: x = {root:.12f}")
    except Exception as e:
        print(f"\n Failed: {e}")
