import numpy as np
import sympy as sp

from solver.euler import EulerSolver
from models.ode_problem import ODEProblem

t = sp.Symbol('t')
C = 0.3
R = 1.4
L = 1.7

# E(t)
E_t = sp.exp(-0.06 * sp.pi * t) * sp.sin(2*t - sp.pi)
E_t_prime = sp.diff(E_t, t)
E_t_double_prime = sp.diff(E_t_prime, t)

# f(t)
f = C * E_t_double_prime + (1/R) * E_t_prime + (1/L) * E_t

# Tạo bài toán
problem = ODEProblem(
    f=f,
    x0=0,
    y0=0,
    h=0.1,
    y_exact=None,  # Không có nghiệm chính xác
    x_end=10,
    label="Circuit current",  
)

# Giải bằng Euler
solver = EulerSolver(problem)
n_steps = int((problem.x_end - problem.x0) / problem.h)
solver.solve(n_steps)
solver.print_results()
