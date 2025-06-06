import numpy as np
from sympy import parse_expr, symbols
from models.ode_system import ODESystem
from models.ode_problem import ODEProblem
from problems.problem_bank import problems_info
from solver.rk6 import RK6Solver
from solver.rk4 import RK4Solver
from solver.midpoint import MidpointSolver
from solver.euler import EulerSolver
from solver.heun import HeunSolver
from predictor_corrector.precor_2 import PredictorCorrector2
from predictor_corrector.precor_3 import PredictorCorrector3
from predictor_corrector.precor_4 import PredictorCorrector4
t = symbols('t')

def main():
    # parse the problem definitions
    problems = [
        ODEProblem(
            f=parse_expr(f_expr, evaluate=False),
            x0=x0,
            y0=y0,
            x_end=x_end,
            h=h,
            y_exact=parse_expr(y_exact_expr, evaluate=False),
            label=label
        )
        for label, f_expr, x0, y0, h, x_end, y_exact_expr in problems_info
    ]

    for prob in problems:
        print(f"\n==> Problem: {prob.label}")
        n_steps = int((prob.x_end - prob.x0) / prob.h)
        print(f"Number of steps: {n_steps}, h: {prob.h}, x0: {prob.x0}, y0: {prob.y0}")

        solvers = [
            ("Euler", EulerSolver(prob)),
            ("Heun", HeunSolver(prob)),
            ("Midpoint", MidpointSolver(prob)),
            ("RK4", RK4Solver(prob)),
            ("RK6", RK6Solver(prob)),
        ]

        # run solvers
        results = {}
        for name, solver in solvers:
            results[name] = solver.solve(n_steps)

        # header row
        header = f"\n{'x':>5} | " + " | ".join([f"{name:^12}" for name in results.keys()]) + f" | {'Exact y':^12}" + " | " + " | ".join([f"{name} Error".ljust(12) for name in results.keys()])
        print(header)
        print("-" * len(header))

        for i in range(len(results["RK4"])): 
            x = results["RK4"][i][0]
            y_exact = prob.y_exact.subs(t, x).evalf()

            row = f"{float(x):5.2f} | "
            row += " | ".join([f"{float(results[name][i][1]):12.6f}" for name in results.keys()])
            row += f" | {float(y_exact):12.6f} | "
            row += " | ".join([f"{abs(float(results[name][i][1]) - float(y_exact)):12.2e}" for name in results.keys()])
            print(row)


def test_ODEsystem():
    def system_f(t, y):
        return np.array([y[1], 2*y[1] - y[0] + t*np.exp(t) - t])  # dao động điều hòa

    p2 = ODESystem(
        f_vec=system_f,
        x0=0,
        y0=[0, 0],
        h=0.1,
        x_end=1,
        label="Ode System Example"
    )

    n_steps = int((p2.x_end - p2.x0) / p2.h)

    solver2 = RK6Solver(p2)
    solver2.solve(n_steps)
    solver2.print_results()

def test_PredictorCorrector():
    problems = [
        ODEProblem(
            f=parse_expr(f_expr, evaluate=False),
            x0=x0,
            y0=y0,
            x_end=x_end,
            h=h,
            y_exact=parse_expr(y_exact_expr, evaluate=False),
            label=label
        )
        for label, f_expr, x0, y0, h, x_end, y_exact_expr in problems_info
    ]

    for prob in problems:
        print(f"\n==> Problem: {prob.label}")
        n_steps = int((prob.x_end - prob.x0) / prob.h)
        print(f"Number of steps: {n_steps}, h: {prob.h}, x0: {prob.x0}, y0: {prob.y0}")

        solvers = [
            ("RK4", RK4Solver(prob)),
            ("RK6", RK6Solver(prob)),
            ("PC 2", PredictorCorrector2(prob)),
            ("PC 3", PredictorCorrector3(prob)),
            ("PC 4", PredictorCorrector4(prob)),
        ]

        # run solvers
        results = {}
        for name, solver in solvers:
            results[name] = solver.solve(n_steps)

        # header row
        header = f"\n{'x':>5} | " + " | ".join([f"{name:^12}" for name in results.keys()]) + f" | {'Exact y':^12}" + " | " + " | ".join([f"{name} Error".ljust(12) for name in results.keys()])
        print(header)
        print("-" * len(header))

        for i in range(len(results["RK4"])): 
            x = results["RK4"][i][0]
            y_exact = prob.y_exact.subs(t, x).evalf()

            row = f"{float(x):5.2f} | "
            row += " | ".join([f"{float(results[name][i][1]):12.6f}" for name in results.keys()])
            row += f" | {float(y_exact):12.6f} | "
            row += " | ".join([f"{abs(float(results[name][i][1]) - float(y_exact)):12.2e}" for name in results.keys()])
            print(row)

if __name__ == "__main__":
    # test_ODEsystem()
    # main()
    test_PredictorCorrector()
