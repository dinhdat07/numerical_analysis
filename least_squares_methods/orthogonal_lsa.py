import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def weight_function(x):
    return 1  # w(x) = 1

def generate_orthogonal_polynomials(n, a, b, w):
    phi_list = [lambda x: 1]
    B = []
    C = []

    for k in range(1, n+1):
        phi_km1 = phi_list[k-1]

        def phi_km1_sq(x):
            return phi_km1(x) ** 2 * w(x)

        def x_phi_km1_sq(x):
            return x * phi_km1(x) ** 2 * w(x)

        Bk = quad(x_phi_km1_sq, a, b)[0] / quad(phi_km1_sq, a, b)[0]
        B.append(Bk)

        if k == 1:
            def phi_k(x, Bk=Bk, phi_km1=phi_km1):
                return (x - Bk) * phi_km1(x)
        else:
            phi_km2 = phi_list[k-2]

            def x_phi_km1_phi_km2(x):
                return x * phi_km1(x) * phi_km2(x) * w(x)

            def phi_km2_sq(x):
                return phi_km2(x) ** 2 * w(x)

            Ck = quad(x_phi_km1_phi_km2, a, b)[0] / quad(phi_km2_sq, a, b)[0]
            C.append(Ck)

            def phi_k(x, Bk=Bk, Ck=Ck, phi_km1=phi_km1, phi_km2=phi_km2):
                return (x - Bk) * phi_km1(x) - Ck * phi_km2(x)

        phi_list.append(phi_k)

    return phi_list, B, C

def compute_approx_coefficients(f, phi_list, a, b, w):
    a_list = []
    for phi in phi_list:
        num = quad(lambda x: f(x) * phi(x) * w(x), a, b)[0]
        den = quad(lambda x: phi(x)**2 * w(x), a, b)[0]
        a_list.append(num / den)
    return a_list

def evaluate_poly(phi_list, a_list, x):
    return sum(a * phi(x) for a, phi in zip(a_list, phi_list))

def plot_continuous_lsa(f, a, b, degree, resolution=500):
    w = weight_function
    phi_list, _, _ = generate_orthogonal_polynomials(degree, a, b, w)
    a_list = compute_approx_coefficients(f, phi_list, a, b, w)

    x_vals = np.linspace(a, b, resolution)
    y_true = np.array([f(x) for x in x_vals])
    y_approx = np.array([evaluate_poly(phi_list, a_list, x) for x in x_vals])
    error = np.abs(y_true - y_approx)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x_vals, y_true, label='True Function', color='black')
    plt.plot(x_vals, y_approx, label=f'Least Squares Approx. (deg={degree})', color='blue')
    plt.title("Continuous Least Squares Approximation with Orthogonal Polynomials")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_vals, error, label='Absolute Error', color='purple')
    plt.title("Absolute Error")
    plt.xlabel("x")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return a_list

if __name__ == "__main__":
    f = lambda x: np.exp(x) * np.sin(2 * np.pi * x)
    a_list = plot_continuous_lsa(f, a=0, b=1, degree=10)
    print("Approximation Coefficients:", a_list)
