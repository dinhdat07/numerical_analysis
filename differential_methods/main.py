from matplotlib import pyplot as plt
import numpy as np
from sympy import parse_expr, symbols
from models.ode_system import ODESystem
from models.ode_problem import ODEProblem
from problems.problem_bank import problems_info, problems_info_low_h, problems_info_precor
from solver.rk6 import RK6Solver
from solver.rk4 import RK4Solver
from solver.midpoint import MidpointSolver
from solver.euler import EulerSolver
from solver.heun import HeunSolver
from predictor_corrector.precor_2 import PredictorCorrector2
from predictor_corrector.precor_3 import PredictorCorrector3
from predictor_corrector.precor_4 import PredictorCorrector4
t, y_sym = symbols('t y')

interpolation_points_17 = [1.25, 1.93, 1.3, 2.1, 2.75, 0.54, 0.94]
interpolation_points_18 = [0.54, 0.94, 1.3, 2.93, 1.25, 1.93]

def linear_interpolation(results, x_target):
    for i in range(len(results) - 1):
        x0, y0 = results[i]
        x1, y1 = results[i + 1]
        if x0 <= x_target <= x1:
            y_target = y0 + (y1 - y0) * (x_target - x0) / (x1 - x0)
            return y_target
    return None 

def cubic_hermite_interpolation(results, f, x_target):
    for i in range(len(results) - 1):
        x0, y0 = results[i]
        x1, y1 = results[i + 1]
        if x0 <= x_target <= x1:
            # đạo hàm tại hai đầu
            dy0 = f.subs({t: x0, y_sym: y0}).evalf()
            dy1 = f.subs({t: x1, y_sym: y1}).evalf()
            
            h = x1 - x0
            s = (x_target - x0) / h

            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2

            y_target = (h00 * y0 + 
                        h10 * h * dy0 + 
                        h01 * y1 + 
                        h11 * h * dy1)
            return y_target
    return None

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
        for label, f_expr, x0, y0, h, x_end, y_exact_expr in problems_info_precor
    ]

    for prob in problems:
        print(f"\n==> Problem: {prob.label}")
        n_steps = int((prob.x_end - prob.x0) / prob.h)
        print(f"Number of steps: {n_steps}, h: {prob.h}, x0: {prob.x0}, y0: {prob.y0}")

        solvers = [
            # ("Euler", EulerSolver(prob)),
            ("Heun", HeunSolver(prob)),
            ("Midpoint", MidpointSolver(prob)),
            ("RK4", RK4Solver(prob)),
            # ("RK6", RK6Solver(prob)),
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

        if prob.label.startswith("3.") :
            print("\n==> Interpolation Results (E17)")
            for interp_x in interpolation_points_17:
                print(f"\nx = {interp_x:.2f}")

                y_exact_interp = prob.y_exact.subs(t, interp_x).evalf()
                print(f"Exact y({interp_x}) = {float(y_exact_interp):.6f}")

                for name in results.keys():
                    y_interp_linear = linear_interpolation(results[name], interp_x)
                    y_interp_hermite = cubic_hermite_interpolation(results[name], prob.f, interp_x)

                    # if y_interp_linear is not None:
                    #     err_lin = abs(float(y_interp_linear) - float(y_exact_interp))
                    #     print(f"{name} Linear: {float(y_interp_linear):.6f} | Error: {err_lin:.2e}")
                    # else:
                    #     print(f"{name} Linear: out of section!")

                    if y_interp_hermite is not None:
                        err_herm = abs(float(y_interp_hermite) - float(y_exact_interp))
                        print(f"{name} Cubic Hermite: {float(y_interp_hermite):.6f} | Error: {err_herm:.2e}")
                    else:
                        print(f"{name} Cubic Hermite: out of section!")


        elif prob.label.startswith("4."):
            # interpolation cho bài 19
            print("\n==> Interpolation Results (E18)")
            for interp_x in interpolation_points_18:
                print(f"\nx = {interp_x:.2f}")

                y_exact_interp = prob.y_exact.subs(t, interp_x).evalf()
                print(f"Exact y({interp_x}) = {float(y_exact_interp):.6f}")

                for name in results.keys():
                    y_interp_linear = linear_interpolation(results[name], interp_x)
                    y_interp_hermite = cubic_hermite_interpolation(results[name], prob.f, interp_x)

                    # if y_interp_linear is not None:
                    #     err_lin = abs(float(y_interp_linear) - float(y_exact_interp))
                    #     print(f"{name} Linear: {float(y_interp_linear):.6f} | Error: {err_lin:.2e}")
                    # else:
                    #     print(f"{name} Linear: out of section!")

                    if y_interp_hermite is not None:
                        err_herm = abs(float(y_interp_hermite) - float(y_exact_interp))
                        print(f"{name} Cubic Hermite: {float(y_interp_hermite):.6f} | Error: {err_herm:.2e}")
                    else:
                        print(f"{name} Cubic Hermite: out of section!")

def solve_twenty_ninth():
    t, y = symbols('t y')
    k = 6.22e-19
    n1 = 2e3
    n2 = 2e3
    n3 = 3e3
    f_expr = f"{k} * ( {n1} - y/2 )**2 * ( {n2} - y/2 )**2 * ( {n3} - 3*y/4 )**3"
    f = parse_expr(f_expr, evaluate=False)

    p = ODEProblem(
        f=f,
        x0=0,
        y0=0,
        h=0.01,
        x_end=0.2,
        y_exact=None,
        label="Chemical Reaction Rate",
    )
    n_steps = int((p.x_end - p.x0) / p.h)
    solver = RK4Solver(p)
    solver.solve(n_steps)
    solver.print_results()

def solve_twenty_eight():
    t, y = symbols('t y')
    f_expr = f"-0.048075 * y**(-1.5)"
    f = parse_expr(f_expr, evaluate=False)

    p = ODEProblem(
        f=f,
        x0=1500,
        y0=0.884531,
        h=1,
        x_end=3000,
        y_exact=None,
        label="Conical Tank Problem",
    )
    n_steps = int((p.x_end - p.x0) / p.h)
    solver = RK4Solver(p)
    solver.solve(n_steps)
    solver.print_results()

def solve_ODEsystem():
    # Define all systems (from problems 1a-d and 2a-d)
    systems = []

    # Problem 1a
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            3*y[0] + 2*y[1] - (2*t**2 + 1)*np.exp(2*t),
            4*y[0] + y[1] + (t**2 + 2*t - 4)*np.exp(2*t)
        ]),
        x0=0, y0=[1, 1], h=0.2, x_end=1, label="1a"
    ))

    # Problem 1b
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            -4*y[0] - 2*y[1] + np.cos(t) + 4*np.sin(t),
            3*y[0] + y[1] - 3*np.sin(t)
        ]),
        x0=0, y0=[0, -1], h=0.2, x_end=2, label="1b"
    ))

    # Problem 1c
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            -y[0] - 2*np.exp(t) + 1,
            -y[0] - np.exp(t) + 1
        ]),
        x0=0, y0=[1, 0, 1], h=0.5, x_end=2, label="1c"
    ))

    # Problem 1d
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1] - y[2] + t,
            3*t**2,
            y[1] + np.exp(-t)
        ]),
        x0=0, y0=[1, 1, -1], h=0.1, x_end=1, label="1d"
    ))

    # Problem 2a
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[0] - y[1] + 2,
            -y[0] + y[1] +  4*t
        ]),
        x0=0, y0=[-1, 0], h=0.1, x_end=1, label="2a"
    ))

    # Problem 2b
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[0]/9 - 2*y[1]/3 - (t**2)/9 + 2/3,
            y[1] + 3*t - 4
        ]),
        x0=0, y0=[-3, 5], h=0.2, x_end=2, label="2b"
    ))

    # Problem 2c
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[0] + 2*y[1] - 2*y[2] + np.exp(-t),
            y[1] + y[2] - 2*np.exp(-t),
            y[0] + 2*y[1] + np.exp(-t)
        ]),
        x0=0, y0=[3, -1, 1], h=0.1, x_end=1, label="2c"
    ))

    # Problem 2d
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            3*y[0] + 2*y[1] - y[2] - 1 - 3*t - 2*np.sin(t),
            y[0] - 2*y[1] + 3*y[2] + 6 - t + 2*np.sin(t) + np.cos(t),
            2*y[0] + 4*y[2] + 8 - 2*t
        ]),
        x0=0, y0=[5, -9, -5], h=0.2, x_end=2, label="2d"
    ))

        # Bài 3
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            2*y[1] - y[0] + t*np.exp(t) - t
        ]),
        x0=0, y0=[0, 0], h=0.1, x_end=1, label="3a"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            (t**3 * np.log(t) + 2*t*y[1] - 2*y[0]) / (t**2)
        ]),
        x0=1, y0=[1, 0], h=0.1, x_end=2, label="3b"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            y[2],
            np.exp(t) - 2*y[2] + y[1] + 2*y[0]
        ]),
        x0=0, y0=[1, 2, 0], h=0.2, x_end=3, label="3c"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            y[2],
            (5*t**3 * np.log(t) + 9*t**3 + t**2*y[2] - 3*t*y[1] + 4*y[0]) / (t**3)
        ]),
        x0=1, y0=[0, 1, 3], h=0.1, x_end=2, label="3d"
    ))

    # Bài 4
    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            3*y[1] - 2*y[0] + 6*np.exp(-t)
        ]),
        x0=0, y0=[2, 2], h=0.1, x_end=1, label="4a"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            (-3*t - t*y[1] + 4*y[0]) / (t**2)
        ]),
        x0=1, y0=[4, 3], h=0.2, x_end=3, label="4b"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            y[2],
            -y[2] + 4*y[1] + 4*y[0]
        ]),
        x0=0, y0=[3, -1, 9], h=0.2, x_end=2, label="4c"
    ))

    systems.append(ODESystem(
        f_vec=lambda t, y: np.array([
            y[1],
            y[2],
            (8*t**3 - 2 - t**2*y[2] + 2*t*y[1] - 2*y[0]) / (t**3)
        ]),
        x0=1, y0=[2, 8, 6], h=0.1, x_end=2, label="4d"
    ))


    # Solve and store results for each
    results_all = []

    for sys in systems:
        solver = PredictorCorrector4(sys)
        n_steps = int((sys.x_end - sys.x0) / sys.h)
        solver.solve(n_steps)
        results_all.append((sys.label, solver.solution))
        print(f"\n==> System: {sys.label}")
        solver.print_results()


def solve_PredictorCorrector():
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
        for label, f_expr, x0, y0, h, x_end, y_exact_expr in problems_info_precor
    ]

    for prob in problems:
        print(f"\n==> Problem: {prob.label}")
        n_steps = int((prob.x_end - prob.x0) / prob.h)
        print(f"Number of steps: {n_steps}, h: {prob.h}, x0: {prob.x0}, y0: {prob.y0}")

        solvers = [
            ("PC 2", PredictorCorrector2(prob)),
            ("PC 3", PredictorCorrector3(prob)),
            ("PC 4", PredictorCorrector4(prob)),
        ]

        # run solvers
        results = {}
        for name, solver in solvers:
            print(f"Solving with {name}...")
            results[name] = solver.solve(n_steps)

        # header row
        header = f"\n{'x':>5} | " + " | ".join([f"{name:^12}" for name in results.keys()]) + f" | {'Exact y':^12}" + " | " + " | ".join([f"{name} Error".ljust(12) for name in results.keys()])
        # header = f"\n{'x':>8} | {'Exact y':>12} | " + " | ".join([f"{name} Error".rjust(12) for name in results.keys()])
        print(header)
        print("-" * len(header))

        for i in range(len(results["PC 2"])): 
            x = results["PC 4"][i][0]
            y_exact = prob.y_exact.subs(t, x).evalf()

            row = f"{float(x):5.2f} | "
            row += " | ".join([f"{float(results[name][i][1]):12.6f}" for name in results.keys()])
            row += f" | {float(y_exact):12.6f} | "
            row += " | ".join([f"{abs(float(results[name][i][1]) - float(y_exact)):12.2e}" for name in results.keys()])
            print(row)

        # for i in range(len(results["PC 2"])): 
        #     x = results["PC 4"][i][0]
        #     y_exact = prob.y_exact.subs(t, x).evalf()
        #     row = f"{float(x):8.2f} | {float(y_exact):12.6f} | "
        #     row += " | ".join([f"{abs(float(results[name][i][1]) - float(y_exact)):12.2e}".rjust(12) for name in results.keys()])
        #     print(row)

def solve_nine_precor():
    t, y = symbols('t y')

    # Problem definition for Q9a: y' = exp(y), y(0)=1, exact solution y(t) = 1 - ln(1 - e*t)
    problems_info_precor = [
        (
            "Q9a: y' = e^y, y(0)=1",
            "exp(y)",    # f_expr
            0.0,         # x0
            1.0,         # y0
            0.01,        # h
            0.20,        # x_end
            "1 - ln(1 - exp(1)*t)"  # y_exact_expr
        )
    ]

    # Setup problems
    problems = [
        ODEProblem(
            f=parse_expr(f_expr, evaluate=False),
            x0=0.0,
            y0=1.0,
            x_end=0.2,
            h=0.01,
            y_exact=parse_expr(y_exact_expr, evaluate=False),
            label="Q9a: y' = e^y, y(0)=1"
        )
        for label, f_expr, x0, y0, h, x_end, y_exact_expr in problems_info_precor
    ]

    # Solve and prepare results
    for prob in problems:
        n_steps = int((prob.x_end - prob.x0) / prob.h)
        solver = PredictorCorrector4(prob)
        results = {}
        print(f"Solving with PC4...")
        results["PC4"] = solver.solve(n_steps)

        # header row
        header = f"\n{'x':>5} | " + " | ".join([f"{name:^12}" for name in results.keys()]) + f" | {'Exact y':^12}" + " | " + " | ".join([f"{name} Error".ljust(12) for name in results.keys()])
        print(header)
        print("-" * len(header))

        for i in range(len(results["PC4"])): 
            x = results["PC4"][i][0]
            y_exact = prob.y_exact.subs(t, x).evalf()

            row = f"{float(x):5.2f} | "
            row += " | ".join([f"{float(results[name][i][1]):12.6f}" for name in results.keys()])
            row += f" | {float(y_exact):12.6f} | "
            row += " | ".join([f"{abs(float(results[name][i][1]) - float(y_exact)):12.2e}" for name in results.keys()])
            print(row)


def solve_eight():
    g = 32.17  # ft/s²
    L = 2.0    # ft
    h = 0.1    # s
    x0 = 0.0
    x_end = 2.0
    theta0 = np.pi / 6
    omega0 = 0.0\
    # a
    def f_nonlinear(x, Y):
        theta, omega = Y
        return np.array([omega, - (g / L) * np.sin(theta)])
    # b
    def f_linear(x, Y):
        theta, omega = Y
        return np.array([omega, - (g / L) * theta])

    # create the ODESystem instances
    nonlinear_system = ODESystem(f_vec=f_nonlinear, y0=[theta0, omega0], x0=x0, h=h, x_end=x_end, label="Nonlinear Pendulum")
    linear_system = ODESystem(f_vec=f_linear, y0=[theta0, omega0], x0=x0, h=h, x_end=x_end, label="Linearized Pendulum")
    # solve the systems
    n_steps = int((x_end - x0) / h)
    nonlinear_solver = RK4Solver(nonlinear_system)
    linear_solver = RK4Solver(linear_system)
    nonlinear_solver.solve(n_steps)
    linear_solver.solve(n_steps)
    # print results
    print("\n==> Nonlinear Pendulum Results")
    nonlinear_solver.print_results()
    print("\n==> Linearized Pendulum Results")
    linear_solver.print_results()

def solve_nine_ten():

    # (9) Lotka-Volterra predator-prey model
    def lotka_volterra(t, Y):
        x1, x2 = Y
        k1, k2, k3, k4 = 3, 0.002, 0.0006, 0.5
        dx1 = k1 * x1 - k2 * x1 * x2
        dx2 = k3 * x1 * x2 - k4 * x2
        return np.array([dx1, dx2])

    # (10) Competitive species model
    def competition_model(t, Y):
        x1, x2 = Y
        dx1 = x1 * (4 - 0.0003 * x1 - 0.0004 * x2)
        dx2 = x2 * (2 - 0.0002 * x1 - 0.0001 * x2)
        return np.array([dx1, dx2])

    # Create and solve both problems
    system_9 = ODESystem(f_vec=lotka_volterra, y0=[1000, 500], x0=0, h=0.1, x_end=4, label="Lotka-Volterra")
    system_10 = ODESystem(f_vec=competition_model, y0=[10000, 10000], x0=0, h=0.1, x_end=4, label="Competition Model")

    solver_9 = RK4Solver(system_9)
    solver_10 = RK4Solver(system_10)
    n_steps_9 = int((system_9.x_end - system_9.x0) / system_9.h)
    n_steps_10 = int((system_10.x_end - system_10.x0) / system_10.h)
    sol_9 = solver_9.solve(n_steps_9)
    sol_10 = solver_10.solve(n_steps_10)
    print("\n==> Lotka-Volterra Results")
    solver_9.print_results()
    print("\n==> Competition Model Results")
    solver_10.print_results()


    # Extract values for plotting
    times_9, values_9 = zip(*sol_9)
    x1_vals_9, x2_vals_9 = zip(*values_9)

    times_10, values_10 = zip(*sol_10)
    x1_vals_10, x2_vals_10 = zip(*values_10)

    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot for Lotka-Volterra (Bài 9)
    axs[0].plot(times_9, x1_vals_9, label='Prey (x1)', color='blue')
    axs[0].plot(times_9, x2_vals_9, label='Predator (x2)', color='red')
    axs[0].set_title('Bài 9 - Mô hình săn mồi-con mồi (Lotka–Volterra)')
    axs[0].set_ylabel('Population')
    axs[0].legend()
    axs[0].grid(True)

    # Plot for Competition Model (Bài 10)
    axs[1].plot(times_10, x1_vals_10, label='Species 1 (x1)', color='green')
    axs[1].plot(times_10, x2_vals_10, label='Species 2 (x2)', color='orange')
    axs[1].set_title('Bài 10 - Mô hình cạnh tranh loài')
    axs[1].set_xlabel('Time (t)')
    axs[1].set_ylabel('Population')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # solve_ODEsystem()
    # solve_eight()
    # solve_nine_ten()
    # main()
    # solve_PredictorCorrector()
    solve_nine_precor()
    # solve_twenty_eight()
