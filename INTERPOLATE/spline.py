import numpy as np
import matplotlib.pyplot as plt

def cubic_spline(x, y, k0, kn):
    n = len(x) - 1  # Số khoảng nội suy
    h = np.diff(x)  
    alpha = np.zeros(n + 1)

    # Bước 1: Tính alpha
    for i in range(1, n):
        alpha[i] = (3/h[i]) * (y[i+1] - y[i]) - (3/h[i-1]) * (y[i] - y[i-1])

    # Bước 2: Giải hệ phương trình tuyến tính
    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)

    l[0] = 1
    z[0] = k0  
    mu[0] = 0  

    for i in range(1, n):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]

    l[n] = 1  
    z[n] = kn  

    # Tìm hệ số b, c, d
    b, c, d = np.zeros(n), np.zeros(n + 1), np.zeros(n)
    c[n] = z[n]  

    for j in range(n-1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - (h[j] * (c[j+1] + 2 * c[j])) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])

    return b, c, d

# Dữ liệu từ bài toán
x_points = np.array([0, 1, 2, 3])
y_points = np.array([1, 0, -1, 0])
k0, kn = 0, -6  # Điều kiện biên

# Tính spline
b, c, d = cubic_spline(x_points, y_points, k0, kn)

# Hiển thị phương trình Spline
for i in range(len(x_points) - 1):
    print(f"S_{i}(x) = {y_points[i]:.6f} + {b[i]:.6f}*(x - {x_points[i]}) + {c[i]:.6f}*(x - {x_points[i]})^2 + {d[i]:.6f}*(x - {x_points[i]})^3")
