import numpy as np
import matplotlib.pyplot as plt


def lsq(x, y, degree):
    A, B = construct_matrix(x, y, degree)
    # coeffs = gaussian_elimination(A, B)
    coeffs = np.linalg.solve(A, B)
    return coeffs

def evaluate(coeffs, x):
    return sum(c * (x**i) for i, c in enumerate(coeffs))

def construct_matrix(x, y, degree):
    n = len(x)
    A = [[np.sum(x**(j+k)) for k in range(degree+1)] for j in range(degree+1)]
    B = [np.sum(y * x**j) for j in range(degree+1)]
    return A, B

def gaussian_elimination(A, B):
    n = len(B)
    # Forward elimination
    for i in range(n):
        # Make the diagonal element 1
        factor = A[i][i]
        for j in range(n):
            A[i][j] /= factor
        B[i] /= factor
        
        # Make the column below diagonal zero
        for k in range(i+1, n):
            factor = A[k][i]
            for j in range(n):
                A[k][j] -= factor * A[i][j]
            B[k] -= factor * B[i]
    
    # Back substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        x[i] = B[i] - sum(A[i][j] * x[j] for j in range(i+1, n))
    
    return x
    

#Example
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1.3, 3.5, 4.2, 5.0, 7.0, 8.8, 10.1, 12.5, 13.0, 15.6])

degree = input("Enter the degree of the polynomial: ")
degree = int(degree)

manual_coeffs = lsq(x, y, degree)

numpy_coeffs = np.polyfit(x, y, degree)
numpy_coeffs = numpy_coeffs[::-1]
print(f"Manual Polynomial Coefficients (Lowest to Highest Degree): {manual_coeffs}")
print(f"Numpy Polynomial Coefficients (Lowest to Highest Degree): {numpy_coeffs}")



# Generate fitted curve
x_fit = np.linspace(min(x), max(x), 100)
y_fit_manual = np.vectorize(lambda xi: evaluate(manual_coeffs, xi))(x_fit)
y_fit_numpy = np.vectorize(lambda xi: evaluate(numpy_coeffs, xi))(x_fit)

plt.scatter(x, y, color="red", label="Data points")
plt.plot(x_fit, y_fit_manual, color="blue", linestyle="--", label=f"Manual LSA (Degree {degree})")
plt.plot(x_fit, y_fit_numpy, color="green", linestyle="-", label=f"numpy.polyfit() (Degree {degree})")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Manual LSA vs numpy.polyfit()")
plt.legend()
plt.grid()
plt.show()
