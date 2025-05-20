from sympy import parse_expr, symbols
from models.ode_problem import ODEProblem
from problems.problem_bank import problems_info
from rk_solver.rk6 import RK6Solver
from rk_solver.rk4 import RK4Solver

t = symbols('t')

def main():
    # Parse from problems_info
    problems = [
        ODEProblem(
            f=parse_expr(f_expr, evaluate=False),
            x0=x0,
            y0=y0,
            x_end = x_end,
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

        # solve with RK4
        rk4_solver = RK4Solver(prob)
        rk4_solution = rk4_solver.solve(n_steps)

        # solve with RK6
        rk6_solver = RK6Solver(prob)
        rk6_solution = rk6_solver.solve(n_steps)

        # results for comparison
        print(f"\n{'x':>5} | {'RK4 y':>12} | {'RK6 y':>12} | {'Exact y':>12} | {'RK4 Error':>12} | {'RK6 Error':>12}")
        print("-" * 75)
        for i in range(len(rk4_solution)):
            x = rk4_solution[i][0]
            y_rk4 = rk4_solution[i][1]
            y_rk6 = rk6_solution[i][1]
            y_exact = prob.y_exact.subs(t, x).evalf()
            err_rk4 = abs(y_rk4 - y_exact)
            err_rk6 = abs(y_rk6 - y_exact)

            print(f"{float(x):5.2f} | {float(y_rk4):12.6f} | {float(y_rk6):12.6f} | {float(y_exact):12.6f} | {float(err_rk4):12.2e} | {float(err_rk6):12.2e}")

if __name__ == "__main__":
    main()
