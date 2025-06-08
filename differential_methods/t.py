# def f(t, y):
#     return 1 / t**2 - y / t - y**2

# def y_exact(t):
#     return -1 / t

# def euler_method(f, t0, y0, h, t_end):
#     t_values = [t0]
#     w_values = [y0]
#     t = t0
#     w = y0
#     while t < t_end:
#         w += h * f(t, w)
#         t += h
#         t_values.append(round(t, 5))
#         w_values.append(w)
#     return t_values, w_values

# # Thông số ban đầu
# t0 = 1.0
# y0 = -1.0
# h = 0.05
# t_end = 2.0

# # Chạy thuật toán Euler
# t_vals, w_vals = euler_method(f, t0, y0, h, t_end)

# # In bảng kết quả
# print(f"{'t_i':<6}{'w_i':<15}{'y(t_i)':<15}{'|w_i - y(t_i)|':<15}")
# for t, w in zip(t_vals, w_vals):
#     y = y_exact(t)
#     error = abs(w - y)
#     print(f"{t:<6.3f}{w:<15.8f}{y:<15.8f}{error:<15.8f}")


def f(t, x):
    if x <= 0:
        return 0 
    return -0.048075 * x**(-1.5)

def rk4_solver(f, x0, t0, tf, h):
    xs = [x0]
    ts = [t0]
    t, x = t0, x0
    while t < tf:
        if x <= 0:  # tránh chia cho 0
            break
        if t + h > tf:
            h = tf - t
        k1 = f(t, x)
        k2 = f(t + h/2, x + h/2 * k1)
        k3 = f(t + h/2, x + h/2 * k2)
        k4 = f(t + h, x + h * k3)
        x += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
        ts.append(t)
        xs.append(x)
    return ts, xs

# Dữ kiện ban đầu
t0 = 0         # thời gian ban đầu
x0 = 8.0       # chiều cao nước ban đầu (ft)
tf = 600       # 10 phút = 600 giây
h = 20         # bước thời gian 20 giây

ts, xs = rk4_solver(f, x0, t0, tf, h)
print(f"Chiều cao nước sau 10 phút: x = {xs[-1]:.4f} ft")


def find_empty_time(f, x0, t0, h, eps=0.001, t_max=3600):
    t, x = t0, x0
    while t < t_max:
        if x <= eps:
            break
        k1 = f(t, x)
        k2 = f(t + h/2, x + h/2 * k1)
        k3 = f(t + h/2, x + h/2 * k2)
        k4 = f(t + h, x + h * k3)
        x += h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += h
    return t

time_to_empty = find_empty_time(f, 8.0, 0, 20)
print(f"Bể cạn sau khoảng {time_to_empty/60:.2f} phút")
