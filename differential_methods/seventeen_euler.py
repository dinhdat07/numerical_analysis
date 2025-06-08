import sympy as sp

from solver.euler import EulerSolver
from models.ode_problem import ODEProblem

t, y_sym = sp.symbols('t y')

p0 = 0.01
b = 0.02
d = 0.015
r = 0.1
rb = r * b
h = 1
t_end = 50
n_steps = int(t_end / h)

# b) dp/dt = rb*(1-p)
f_expr = rb * (1 - y_sym)

# c) p(t) = 1 - (1 - p0) * e^(-rb*t)
p_exact = 1 - (1 - p0) * sp.exp(-rb * t)

problem = ODEProblem(
    f=f_expr,
    x0=0,
    y0=p0,
    h=h,
    y_exact=p_exact,
    x_end=t_end,
    label="Nonconformist proportion"
)


solver = EulerSolver(problem)
solution = solver.solve(n_steps)
solver.print_results()


p_euler_50 = solution[-1][1]
p_exact_50 = p_exact.subs(t, t_end).evalf()
error = abs(p_euler_50 - p_exact_50)

print(f"\n==> Sau {t_end} năm:")
print(f"Giá trị bằng Euler: p({t_end}) ≈ {float(p_euler_50):.6f}")
print(f"Giá trị nghiệm giải tích: p({t_end}) = {float(p_exact_50):.6f}")
print(f"Sai số tuyệt đối: {float(error):.2e}")
