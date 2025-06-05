import math

def bisection(f, a, b, tol=1e-10, max_iter=1000, verbose=False):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise ValueError("Bisection method requires f(a) * f(b) < 0.")

    for i in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        if verbose:
            print(f"Iter {i+1}: a = {a:.10f}, b = {b:.10f}, c = {c:.10f}, f(c) = {fc:.2e}")

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    raise RuntimeError("Bisection method did not converge within the maximum number of iterations.")

if __name__ == "__main__":
    def f(x):
        return math.log(x + 1) - math.cos(x)
    try:
        root = bisection(f, a=0, b=1, tol=1e-15, max_iter=1000, verbose=True)
        print(f"Approximate root: {root:.20f}")
    except Exception as e:
        print("Error:", e)


