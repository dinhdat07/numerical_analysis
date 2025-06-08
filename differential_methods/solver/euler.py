import numpy as np
from sympy import symbols, sympify

t, y_sym = symbols('t y') 


class EulerSolver:
    def __init__(self, problem):
        self.problem = problem
        self.solution = [(problem.x0, problem.y0)]
    
    def step(self, x, y, h):
        if hasattr(self.problem, 'f_vec'):
            f_val = self.problem.f_vec(x, y)
        else:
            f_val = self.problem.f.subs({t: x, y_sym: y}).evalf()
        
        y_next = y + h * f_val
        return x + h, y_next

    
    def solve(self, n_steps):
        x, y_val = self.problem.x0, self.problem.y0
        h = self.problem.h
        for _ in range(n_steps):
            x, y_val = self.step(x, y_val, h)
            self.solution.append((x, y_val))
        return self.solution

    def print_results(self):
        print(f"\n ==> Euler: {self.problem.label}")
        for x, y in self.solution:
            if hasattr(self.problem, 'f_vec'):
                print(f"x = {float(x):.2f}, y ≈ {np.array2string(y, precision=6)}")
            else:
                if self.problem.y_exact is not None:
                    y_true = self.problem.y_exact.subs(t, x).evalf()
                    error = abs(y_true - y)
                    print(
                        f"t = {x:.2f}, i ≈ {float(y):.6f}, "
                        f"exact = {float(y_true):.6f}, error = {float(error):.2e}"
                    )
                else:
                    print(f"t = {x:.2f}, i ≈ {float(y):.6f}")
