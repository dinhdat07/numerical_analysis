import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hằng số hấp dẫn
G = 1.0
m1 = m2 = m3 = 1.0

# Thời gian
dt = 0.001
num_steps = 3000  # Đủ dài để xem quỹ đạo


def generate_random_state():
    rng = np.random.default_rng()
    
    # Random vị trí trong phạm vi [-1, 1]
    positions = rng.uniform(-1, 1, size=(3, 2))
    
    # Random vận tốc trong phạm vi [-0.5, 0.5]
    velocities = rng.uniform(-0.5, 0.5, size=(3, 2))

    # Giảm drift bằng cách trừ trung bình vận tốc (cho gần tổng động lượng = 0)
    velocities -= velocities.mean(axis=0)

    # Ghép lại thành state (x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...)
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

# Dùng
state = generate_random_state()

# Trạng thái ban đầu (hình số 8 nổi tiếng)
# state = np.array([
#      0.97000436, -0.24308753, -0.46620368, -0.43236573,  # Body 1
#     -0.97000436,  0.24308753, -0.46620368, -0.43236573,  # Body 2
#      0.0,          0.0,         0.93240737,  0.86473146   # Body 3
# ], dtype=np.float64)

# Hàm tính đạo hàm
def derivatives(state):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state
    r12 = np.hypot(x2 - x1, y2 - y1)
    r13 = np.hypot(x3 - x1, y3 - y1)
    r23 = np.hypot(x3 - x2, y3 - y2)

    a1x = G * m2 * (x2 - x1) / r12**3 + G * m3 * (x3 - x1) / r13**3
    a1y = G * m2 * (y2 - y1) / r12**3 + G * m3 * (y3 - y1) / r13**3

    a2x = G * m1 * (x1 - x2) / r12**3 + G * m3 * (x3 - x2) / r23**3
    a2y = G * m1 * (y1 - y2) / r12**3 + G * m3 * (y3 - y2) / r23**3

    a3x = G * m1 * (x1 - x3) / r13**3 + G * m2 * (x2 - x3) / r23**3
    a3y = G * m1 * (y1 - y3) / r13**3 + G * m2 * (y2 - y3) / r23**3

    return np.array([
        vx1, vy1, a1x, a1y,
        vx2, vy2, a2x, a2y,
        vx3, vy3, a3x, a3y
    ])

# Hàm Runge-Kutta bậc 4
def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

# Lưu trạng thái
positions = np.zeros((num_steps, 3, 2))
for i in range(num_steps):
    positions[i, 0] = state[0:2]
    positions[i, 1] = state[4:6]
    positions[i, 2] = state[8:10]
    state = rk4_step(state, dt)

# ======== ANIMATION ========
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['r', 'g', 'b']
lines = [ax.plot([], [], color)[0] for color in colors]
dots = [ax.plot([], [], color + 'o')[0] for color in colors]


ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.set_title("Three-Body Problem - Animation")

def init():
    for line, dot in zip(lines, dots):
        line.set_data([], [])
        dot.set_data([], [])
    return lines + dots

collision_detected = False
paused_frame = None

def detect_collision(pos, threshold=0.05):
    for i in range(3):
        for j in range(i + 1, 3):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < threshold:
                return True
    return False

def update_lines_and_dots(frame):
    for i in range(3):
        x = positions[:frame, i, 0]
        y = positions[:frame, i, 1]
        lines[i].set_data(x, y)
        if len(x) > 0:
            dots[i].set_data([x[-1]], [y[-1]])
    return lines + dots

def update(frame):
    global collision_detected, paused_frame

    if collision_detected:
        return update_lines_and_dots(paused_frame)

    current_positions = positions[frame]
    if detect_collision(current_positions):
        collision_detected = True
        paused_frame = frame
        ax.set_title("Collision detected!", color='red')
        fig.canvas.draw_idle()
        return update_lines_and_dots(frame)

    return update_lines_and_dots(frame)

ani = FuncAnimation(
    fig, update,
    frames=range(1, num_steps, 10),  # ← nhảy từng 3 frame (nhanh gấp ~3 lần)
    init_func=init,
    blit=True,
    interval=10,  # có thể giảm thêm nếu muốn nhanh nữa
    repeat=True
)

plt.show()
