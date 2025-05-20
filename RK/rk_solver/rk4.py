from sympy import symbols, sympify

t, y_sym = symbols('t y')  # Thêm y_sym


class RK4Solver:
    def __init__(self, problem):
        self.problem = problem
        self.solution = [(problem.x0, problem.y0)]

    def step(self, x, y, h):
        f = self.problem.f
        k1 = f.subs({t: x, y_sym: y}).evalf()
        k2 = f.subs({t: x + h/2, y_sym: y + k1*h/2}).evalf()
        k3 = f.subs({t: x + h/2, y_sym: y + k2*h/2}).evalf()
        k4 = f.subs({t: x + h, y_sym: y + k3*h}).evalf()
        y_next = y + (k1 + 2*k2 + 2*k3 + k4) * h / 6
        x_next = x + h
        return x_next, y_next

    def solve(self, n_steps):
        x, y_val = self.problem.x0, self.problem.y0
        h = self.problem.h
        for _ in range(n_steps):
            x, y_val = self.step(x, y_val, h)
            self.solution.append((x, y_val))
        return self.solution

    def print_results(self):
        print(f"\n ==> Problem: {self.problem.label}")
        for x, y_approx in self.solution:
            y_true = self.problem.y_exact.subs(t, x).evalf()
            error = abs(y_true - y_approx)
            print(
                f"x = {float(x):.2f}, y ≈ {float(y_approx):.6f}, "
                f"exact = {float(y_true):.6f}, error = {float(error):.2e}"
            )

