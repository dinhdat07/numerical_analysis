import numpy as np
import matplotlib.pyplot as plt

# 1. Sinh dữ liệu nhiễu: 1000 điểm
n = 1000
x = np.linspace(0, 10, n)
y_true = np.sin(x)                   # Dữ liệu gốc
noise = np.random.normal(0, 0.1, n)  # Nhiễu Gaussian
y_noisy = y_true + noise             # Dữ liệu có nhiễu

# 2. Tính h = x_{i+1} - x_i
h = x[1:] - x[:-1]

# 3. Lập hệ phương trình tridiagonal để tìm M (đạo hàm bậc 2)
alpha = np.zeros(n - 2)
for i in range(1, n - 1):
    alpha[i - 1] = (3 / h[i]) * (y_noisy[i + 1] - y_noisy[i]) - (3 / h[i - 1]) * (y_noisy[i] - y_noisy[i - 1])

# 4. Giải hệ tridiagonal (dùng Thomas algorithm thủ công)
a = np.ones(n - 2) * 2 * (h[:-1] + h[1:])
l = h[1:-1]
u = h[1:-1]

# Forward elimination
for i in range(1, n - 2):
    w = l[i - 1] / a[i - 1]
    a[i] -= w * u[i - 1]
    alpha[i] -= w * alpha[i - 1]

# Back substitution
M = np.zeros(n)
M[n - 2] = alpha[-1] / a[-1]
for i in range(n - 3, 0, -1):
    M[i] = (alpha[i - 1] - u[i - 1] * M[i + 1]) / a[i - 1]

# 5. Tính hệ số spline từng đoạn
spline_coefs = []
for i in range(n - 1):
    hi = h[i]
    Ai = y_noisy[i]
    Bi = (y_noisy[i + 1] - y_noisy[i]) / hi - (hi / 3) * (2 * M[i] + M[i + 1])
    Ci = M[i]
    Di = (M[i + 1] - M[i]) / (3 * hi)
    spline_coefs.append((Ai, Bi, Ci, Di))

# 6. Hàm nội suy spline tại điểm bất kỳ
def spline_eval(x0):
    if x0 <= x[0]:
        i = 0
    elif x0 >= x[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x, x0) - 1
    dx = x0 - x[i]
    Ai, Bi, Ci, Di = spline_coefs[i]
    return Ai + Bi * dx + Ci * dx**2 + Di * dx**3

# 7. Nội suy để làm mịn
x_smooth = np.linspace(0, 10, 5000)
y_smooth = np.array([spline_eval(xi) for xi in x_smooth])

# 8. Vẽ kết quả
plt.figure(figsize=(12, 5))
plt.plot(x, y_noisy, '.', alpha=0.3, label='Dữ liệu nhiễu', markersize=3)
plt.plot(x_smooth, y_smooth, '-', label='Spline làm mịn', linewidth=2)
plt.plot(x, y_true, '--', label='Hàm gốc', linewidth=1.5)
plt.title('Làm mịn dữ liệu bằng spline nội suy bậc 3 (1000 điểm)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
