import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Sinh dữ liệu 1000 điểm
n = 1000
x = np.linspace(0, 10, n)
y = np.sin(x)  # hoặc y = np.random.rand(n) nếu muốn dữ liệu ngẫu nhiên

# 2. Tính khoảng cách h_i = x_{i+1} - x_i
h = x[1:] - x[:-1]

# 3. Lập hệ số vế phải cho hệ phương trình tridiagonal
alpha = np.zeros(n - 2)
for i in range(1, n - 1):
    alpha[i - 1] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])

# 4. Lập hệ số của ma trận tam trục (tridiagonal system)
l = np.ones(n)
mu = np.zeros(n - 1)
z = np.zeros(n)

# Hệ số tridiagonal: d_i * M_{i-1} + a_i * M_i + c_i * M_{i+1} = RHS
a = np.ones(n - 2) * 2 * (h[:-1] + h[1:])
b = h[1:-1]
c = h[1:-1]

# 5. Giải hệ phương trình tam trục: dùng thuật toán Thomas
# Forward elimination
for i in range(1, n - 2):
    w = h[i] / a[i - 1]
    a[i] -= w * h[i]
    alpha[i] -= w * alpha[i - 1]

# Back substitution
M = np.zeros(n)
M[n - 2] = alpha[-1] / a[-1]
for i in range(n - 3, 0, -1):
    M[i] = (alpha[i - 1] - h[i] * M[i + 1]) / a[i - 1]

# 6. Tính hệ số spline a_i, b_i, c_i, d_i cho từng đoạn
spline_coefs = []
for i in range(n - 1):
    hi = h[i]
    Ai = y[i]
    Bi = (y[i + 1] - y[i]) / hi - (hi / 3) * (2 * M[i] + M[i + 1])
    Ci = M[i]
    Di = (M[i + 1] - M[i]) / (3 * hi)
    spline_coefs.append((Ai, Bi, Ci, Di))

# 7. Hàm nội suy tại một điểm x0
def spline_eval(x0):
    # Tìm đoạn chứa x0
    if x0 <= x[0]:
        i = 0
    elif x0 >= x[-1]:
        i = n - 2
    else:
        i = np.searchsorted(x, x0) - 1
    dx = x0 - x[i]
    Ai, Bi, Ci, Di = spline_coefs[i]
    return Ai + Bi * dx + Ci * dx**2 + Di * dx**3

# 8. Vẽ kết quả nội suy
x_eval = np.linspace(x[0], x[-1], 5000)
y_eval = np.array([spline_eval(xi) for xi in x_eval])

plt.figure(figsize=(10, 4))
plt.plot(x, y, 'o', markersize=2, label='Dữ liệu')
plt.plot(x_eval, y_eval, '-', label='Spline nội suy')
plt.legend()
plt.title('Natural Cubic Spline - 1000 điểm (Không dùng thư viện)')
plt.show()

# # 9. Nội suy với scipy để so sánh
# cs = CubicSpline(x, y, bc_type='natural')  # bc_type='natural' để đúng với spline tự cài
# y_eval_scipy = cs(x_eval)

# # 10. So sánh độ sai lệch
# diff = np.abs(y_eval - y_eval_scipy)
# max_error = np.max(diff)
# mean_error = np.mean(diff)

# print(f"Max error giữa spline tự cài và scipy: {max_error:.2e}")
# print(f"Mean error: {mean_error:.2e}")

# # 11. Vẽ chênh lệch
# plt.figure(figsize=(10, 2))
# plt.plot(x_eval, diff, color='red', label='Sai lệch |ours - scipy|')
# plt.title('Sai lệch giữa spline tự cài và spline scipy')
# plt.ylabel('Sai lệch')
# plt.xlabel('x')
# plt.legend()
# plt.tight_layout()
# plt.show()