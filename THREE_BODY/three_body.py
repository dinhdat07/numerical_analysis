import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Hằng số hấp dẫn
G = 1.0
m1 = m2 = m3 = 1.0

# Thời gian
dt = 0.0001
num_steps = 300000  # Đủ dài để xem quỹ đạo


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

def generate_random_state_sym():
    rng = np.random.default_rng()
    
    # Đối xứng qua trục y
    a = rng.uniform(0.5, 1.5)
    positions = np.array([
        [-a, 0],
        [ a, 0],
        [ 0, 0]
    ])

    # Vận tốc đối xứng
    v1 = rng.uniform(-0.8, 0.8)
    v2 = rng.uniform(-0.8, 0.8)
    velocities = np.array([
        [v1, v2],
        [v1, v2],
        [-2*v1, -2*v2]
    ])

    # Ghép lại
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

# Use
# state = generate_random_state_sym()
# print("Initial state:", state)
#Run okay:
# Initial state: [-1.44332343  0.         -0.24852747 -0.36869087  1.44332343  0.
#  -0.24852747 -0.36869087  0.          0.          0.49705494  0.73738173]

# 8-figure
state = np.array([
     0.97000436, -0.24308753, -0.46620368, -0.43236573,  
    -0.97000436,  0.24308753, -0.46620368, -0.43236573,  
     0.0,          0.0,         0.93240737,  0.86473146  
], dtype=np.float64)


# lagrange
# r = 1.0  # Bán kính quỹ đạo
# v = 0.5  # Tốc độ góc
# state = np.array([
#     r, 0.0,                     0.0, v,                         # Vật thể 1
#     -r/2,  r*np.sqrt(3)/2,     -v*np.sqrt(3)/2, -v/2,           # Vật thể 2
#     -r/2, -r*np.sqrt(3)/2,      v*np.sqrt(3)/2, -v/2            # Vật thể 3
# ])

# butterfly
# state = np.array([
#     -1.0, 0.0,       0.306893,  0.125507,    # Body 1
#      1.0, 0.0,       0.306893,  0.125507,    # Body 2
#      0.0, 0.0,      -0.613786, -0.251014     # Body 3
# ])


# yin-yang
# state = np.array([
#     -1.0, 0.0,       0.513938,  0.304736,   # Body 1
#      1.0, 0.0,       0.513938,  0.304736,   # Body 2
#      0.0, 0.0,      -1.027876, -0.609472    # Body 3
# ])


#dragon-fly
state = np.array([
    -1.0, 0.0,        0.080584,  0.588836,     # Body 1
     1.0, 0.0,        0.080584,  0.588836,     # Body 2
     0.0, 0.0,       -0.161168, -1.177672      # Body 3
])




# Hàm tính đạo hàm
def derivatives(state):
    x1, y1, vx1, vy1, x2, y2, vx2, vy2, x3, y3, vx3, vy3 = state
    eps = 0

    r12 = np.hypot(x2 - x1, y2 - y1) + eps
    r13 = np.hypot(x3 - x1, y3 - y1) + eps
    r23 = np.hypot(x3 - x2, y3 - y2) + eps

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

def compute_energy(pos, vel):
    x1, y1 = pos[0]
    x2, y2 = pos[1]
    x3, y3 = pos[2]

    vx1, vy1 = vel[0]
    vx2, vy2 = vel[1]
    vx3, vy3 = vel[2]

    # Động năng
    T = 0.5 * m1 * (vx1**2 + vy1**2) + \
        0.5 * m2 * (vx2**2 + vy2**2) + \
        0.5 * m3 * (vx3**2 + vy3**2)

    # Thế năng hấp dẫn
    r12 = np.hypot(x2 - x1, y2 - y1)
    r13 = np.hypot(x3 - x1, y3 - y1)
    r23 = np.hypot(x3 - x2, y3 - y2)

    U = -G * m1 * m2 / r12 - G * m1 * m3 / r13 - G * m2 * m3 / r23

    return T + U


# Hàm Runge-Kutta bậc 4
def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)
    return state + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

# Hàm Runge-Kutta bậc 6
def rk6_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + dt * k1 / 4)
    k3 = derivatives(state + dt * (k1 + k2) / 8)
    k4 = derivatives(state + dt * (-0.5*k2 + k3))
    k5 = derivatives(state + dt * (3*k1 + 9*k4) / 16)
    k6 = derivatives(state + dt * (-3*k1 + 2*k2 + 12*k3 - 12*k4 + 8*k5) / 7)

    return state + dt / 90 * (7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)

# Lưu trạng thái
positions = np.zeros((num_steps, 3, 2))
for i in range(num_steps):
    positions[i, 0] = state[0:2]
    positions[i, 1] = state[4:6]
    positions[i, 2] = state[8:10]
    state = rk6_step(state, dt)

# ======== ANIMATION ========
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['r', 'g', 'b']
lines = [ax.plot([], [], color)[0] for color in colors]
dots = [ax.plot([], [], color + 'o')[0] for color in colors]
energy_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))


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

def detect_collision(pos, threshold=0.01):
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
        return update_lines_and_dots(paused_frame) + [energy_text]

    current_positions = positions[frame]

    if detect_collision(current_positions):
        collision_detected = True
        paused_frame = frame
        ax.set_title("Collision detected!", color='red')
        fig.canvas.draw_idle()
        return update_lines_and_dots(frame) + [energy_text]

    if frame % 300 == 0:
        velocities = (positions[frame] - positions[frame - 1]) / dt
        energy = compute_energy(current_positions, velocities)
        energy_text.set_text(f"Energy: {energy:.6f}")

    return update_lines_and_dots(frame) + [energy_text]


ani = FuncAnimation(
    fig, update,
    frames=range(0, num_steps, 300),  # ← nhảy từng 3 frame (nhanh gấp ~3 lần)
    init_func=init,
    blit=True,
    interval=10,  # có thể giảm thêm nếu muốn nhanh nữa
    repeat=True
)

plt.show()
