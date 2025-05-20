import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === Define target function and weight ===
def f(x):
    return np.exp(x)  # You can change this function

def w(x):
    return 1  # Change to any weight function

# === Inner product with weight ===
def inner_product(f1, f2, a, b):
    return quad(lambda x: w(x) * f1(x) * f2(x), a, b)[0]

# === Generate orthogonal basis using recurrence ===
def generate_orthogonal_polynomials(n, a, b):
    phi = [lambda x: 1]  # phi_0
    B = []
    C = []

    if n >= 1:
        B1 = inner_product(lambda x: x * phi[0](x), phi[0], a, b) / inner_product(phi[0], phi[0], a, b)
        phi.append(lambda x, B1=B1: x - B1)
        B.append(B1)

    for k in range(2, n + 1):
        phi_k_minus_1 = phi[k-1]
        phi_k_minus_2 = phi[k-2]

        Bk = inner_product(lambda x: x * phi_k_minus_1(x), phi_k_minus_1, a, b) / inner_product(phi_k_minus_1, phi_k_minus_1, a, b)
        Ck = inner_product(lambda x: x * phi_k_minus_1(x) * phi_k_minus_2(x), lambda x: 1, a, b) / inner_product(phi_k_minus_2, phi_k_minus_2, a, b)

        B.append(Bk)
        C.append(Ck)

        # define phi_k(x)
        phi_k = lambda x, Bk=Bk, Ck=Ck, phi_k_minus_1=phi_k_minus_1, phi_k_minus_2=phi_k_minus_2: \
            (x - Bk) * phi_k_minus_1(x) - Ck * phi_k_minus_2(x)

        phi.append(phi_k)

    return phi

# === Compute coefficients a_j ===
def compute_coefficients(phi, f, a, b):
    a_coeffs = []
    for j in range(len(phi)):
        alpha_j = inner_product(phi[j], phi[j], a, b)
        a_j = inner_product(lambda x: phi[j](x) * f(x), lambda x: 1, a, b) / alpha_j
        a_coeffs.append(a_j)
    return a_coeffs

# === Construct approximation function ===
def approximation_function(phi, a_coeffs):
    return lambda x: sum(a * p(x) for a, p in zip(a_coeffs, phi))

# === MAIN EXECUTION ===
a, b = 0, 1       # Interval [a, b]
n = 3             # Degree of approximation

phi = generate_orthogonal_polynomials(n, a, b)
a_coeffs = compute_coefficients(phi, f, a, b)
P = approximation_function(phi, a_coeffs)

# === Plot original and approximation ===
x_vals = np.linspace(a, b, 400)
f_vals = [f(x) for x in x_vals]
P_vals = [P(x) for x in x_vals]

plt.plot(x_vals, f_vals, label='f(x)', color='blue')
plt.plot(x_vals, P_vals, label=f'Least Squares Approx (deg {n})', color='red', linestyle='--')
plt.legend()
plt.title('Continuous Least Squares Approximation')
plt.grid(True)
plt.show()

# === Print coefficients ===
print("Coefficients a_j:")
for i, a_j in enumerate(a_coeffs):
    print(f"a_{i} = {a_j:.6f}")
