import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Đọc ảnh xám và chuyển sang numpy array
img = Image.open('INTERPOLATE\sample.jpg').convert('L')  # 'L' = grayscale
img_np = np.array(img)
H, W = img_np.shape

# Tỉ lệ phóng to (2x)
scale = 2
new_H = int(H * scale)
new_W = int(W * scale)

# --- Hàm spline 1 chiều (cho 1 hàng/cột) ---
def cubic_spline_interpolate_1d(x, y, x_new):
    n = len(x)
    h = x[1:] - x[:-1]
    
    # Bước 1: Tính alpha
    alpha = np.zeros(n - 2)
    for i in range(1, n - 1):
        alpha[i - 1] = (3/h[i])*(y[i + 1] - y[i]) - (3/h[i - 1])*(y[i] - y[i - 1])

    # Bước 2: Giải hệ tridiagonal
    a = np.ones(n - 2) * 2 * (h[:-1] + h[1:])
    l = h[1:-1]
    u = h[1:-1]

    for i in range(1, n - 2):
        w = l[i - 1] / a[i - 1]
        a[i] -= w * u[i - 1]
        alpha[i] -= w * alpha[i - 1]

    M = np.zeros(n)
    M[n - 2] = alpha[-1] / a[-1]
    for i in range(n - 3, 0, -1):
        M[i] = (alpha[i - 1] - u[i - 1] * M[i + 1]) / a[i - 1]

    # Bước 3: Tính hệ số spline
    coefs = []
    for i in range(n - 1):
        hi = h[i]
        Ai = y[i]
        Bi = (y[i + 1] - y[i])/hi - hi*(2*M[i] + M[i+1])/3
        Ci = M[i]
        Di = (M[i+1] - M[i])/(3*hi)
        coefs.append((Ai, Bi, Ci, Di))

    # Bước 4: Nội suy tại x_new
    y_new = []
    for xi in x_new:
        if xi <= x[0]:
            i = 0
        elif xi >= x[-1]:
            i = n - 2
        else:
            i = np.searchsorted(x, xi) - 1
        dx = xi - x[i]
        Ai, Bi, Ci, Di = coefs[i]
        yi = Ai + Bi*dx + Ci*dx**2 + Di*dx**3
        y_new.append(yi)
    return np.clip(y_new, 0, 255)  # clamp 0-255

# --- Bước 1: nội suy theo chiều ngang ---
x_orig = np.arange(W)
x_resized = np.linspace(0, W - 1, new_W)
img_h_scaled = np.zeros((H, new_W))
for i in range(H):
    img_h_scaled[i, :] = cubic_spline_interpolate_1d(x_orig, img_np[i, :], x_resized)

# --- Bước 2: nội suy theo chiều dọc ---
x_orig_v = np.arange(H)
x_resized_v = np.linspace(0, H - 1, new_H)
img_final = np.zeros((new_H, new_W))
for j in range(new_W):
    img_final[:, j] = cubic_spline_interpolate_1d(x_orig_v, img_h_scaled[:, j], x_resized_v)

# Hiển thị ảnh gốc và ảnh đã resize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_np, cmap='gray')
plt.title("Ảnh gốc")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img_final, cmap='gray')
plt.title(f"Ảnh phóng to {scale}x bằng spline")
plt.axis('off')
plt.tight_layout()
plt.show()
