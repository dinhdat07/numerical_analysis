from sympy import symbols

t, y = symbols('t y')

class RK6Solver:
    def __init__(self, problem):
        self.problem = problem
        self.solution = [(problem.x0, problem.y0)]

    def step(self, x, y_val, h):
        f = self.problem.f

        k1 = f.subs({t: x, y: y_val}).evalf()
        k2 = f.subs({t: x + h/4, y: y_val + h/4 * k1}).evalf()
        k3 = f.subs({t: x + h/4, y: y_val + h/8 * k1 + h/8 * k2}).evalf()
        k4 = f.subs({t: x + h/2, y: y_val - h/2 * k2 + h * k3}).evalf()
        k5 = f.subs({t: x + 3*h/4, y: y_val + 3*h/16 * k1 + 9*h/16 * k4}).evalf()
        k6 = f.subs({t: x + h, y: y_val - 3*h/7 * k1 + 2*h/7 * k2 + 12*h/7 * k3 - 12*h/7 * k4 + 8*h/7 * k5}).evalf()

        y_next = y_val + h * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6) / 90
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
        print(f"\n ==> Problem: {self.problem.label} (RK6)")
        for x, y_approx in self.solution:
            y_true = self.problem.y_exact.subs(t, x).evalf()
            error = abs(y_true - y_approx)
            print(
                f"x = {float(x):.2f}, y ≈ {float(y_approx):.6f}, "
                f"exact = {float(y_true):.6f}, error = {float(error):.2e}"
            )
