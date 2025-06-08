import numpy as np
from sympy import symbols
from solver.rk4 import RK4Solver

t, y_sym = symbols('t y') 

class PredictorCorrectorBase:
    def __init__(self, problem):
        self.problem = problem
        self.solution = []

    def _f(self, x, y):
        if hasattr(self.problem, 'f_vec'):  # ODESystem
            return self.problem.f_vec(x, np.array(y, dtype=float))
        else:  # ODEProblem
            return float(self.problem.f.subs({t: x, y_sym: y}).evalf())

    def rk4_init(self, num_init):
        rk4 = RK4Solver(self.problem)
        rk4.solve(num_init - 1)
        self.solution = rk4.solution.copy()

    def solve(self, n_steps, max_iter=4, tol=1e-8):
        h = self.problem.h
        order = self.order
        self.rk4_init(order)

        for i in range(order - 1, n_steps):
            x_vals = [self.solution[i - j][0] for j in range(order)][::-1]
            y_vals = [self.solution[i - j][1] for j in range(order)][::-1]
            f_vals = [self._f(x, y) for x, y in zip(x_vals, y_vals)]

            x_next = x_vals[-1] + h
            y_pred = self.predict(y_vals, f_vals, h)
            y_corr = self.correct(y_vals, f_vals, h, x_next, y_pred, max_iter, tol)
            print(f"Predictor step: x = {x_next:.6f}, y_pred = {y_pred:.6f}, y_corr = {y_corr:.6f}")

            self.solution.append((x_next, y_corr))

        return self.solution


    def print_results(self):
        print(f"\n ==> Problem: {self.problem.label} (Predictor-Corrector {self.order} steps)")

        if hasattr(self.problem, 'y_exact'):  # Only for ODEProblem
            for x, y in self.solution:
                y_true = self.problem.y_exact.subs(t, x).evalf()
                error = abs(y_true - y)
                print(
                    f"x = {float(x):.2f}, y ≈ {float(y):.6f}, "
                    f"exact = {float(y_true):.6f}, error = {float(error):.2e}"
                )
        else:  # For ODESystem, print without error
            for x, y in self.solution:
                y_str = ", ".join(f"{yi:.6f}" for yi in y)
                print(f"x = {x:.2f}, y ≈ [{y_str}]")

    def predict(self, y_vals, f_vals, h):
        raise NotImplementedError

    def correct(self, y_vals, f_vals, h, x_next, y_pred, max_iter, tol):
        raise NotImplementedError
