import numpy as np


G = 1.0
m1 = m2 = m3 = 1.0

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
