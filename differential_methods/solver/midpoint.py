import numpy as np
from sympy import symbols

t, y_sym = symbols('t y') 

class MidpointSolver:
    def __init__(self, problem):
        self.problem = problem
        self.solution = [(problem.x0, problem.y0)]

    def step(self, x, y, h):
        is_system = hasattr(self.problem, 'f_vec')

        if is_system:
            f = self.problem.f_vec
            k1 = f(x, y)
            k2 = f(x + h/2, y + h/2 * k1)
        else:
            f = self.problem.f
            k1 = f.subs({t: x, y_sym: y}).evalf()
            k2 = f.subs({t: x + h/2, y_sym: y + h/2 * k1}).evalf()

        y_next = y + h * k2
        return x + h, y_next

    def solve(self, n_steps):
        x, y_val = self.problem.x0, self.problem.y0
        h = self.problem.h
        for _ in range(n_steps):
            x, y_val = self.step(x, y_val, h)
            self.solution.append((x, y_val))
        return self.solution

    def print_results(self):
        print(f"\n ==> Midpoint: {self.problem.label}")
        for x, y in self.solution:
            if hasattr(self.problem, 'f_vec'):
                print(f"x = {float(x):.2f}, y ≈ {np.array2string(y, precision=6)}")
            else:
                y_true = self.problem.y_exact.subs(t, x).evalf()
                error = abs(y_true - y)
                print(
                    f"x = {float(x):.2f}, y ≈ {float(y):.6f}, "
                    f"exact = {float(y_true):.6f}, error = {float(error):.2e}"
                )