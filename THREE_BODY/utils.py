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

def derivatives_3D(state):
    x1, y1, z1, vx1, vy1, vz1 = state[0:6]
    x2, y2, z2, vx2, vy2, vz2 = state[6:12]
    x3, y3, z3, vx3, vy3, vz3 = state[12:18]

    r12 = np.linalg.norm([x2 - x1, y2 - y1, z2 - z1])
    r13 = np.linalg.norm([x3 - x1, y3 - y1, z3 - z1])
    r23 = np.linalg.norm([x3 - x2, y3 - y2, z3 - z2])

    a1 = G * m2 * (np.array([x2, y2, z2]) - np.array([x1, y1, z1])) / r12**3 \
       + G * m3 * (np.array([x3, y3, z3]) - np.array([x1, y1, z1])) / r13**3
    a2 = G * m1 * (np.array([x1, y1, z1]) - np.array([x2, y2, z2])) / r12**3 \
       + G * m3 * (np.array([x3, y3, z3]) - np.array([x2, y2, z2])) / r23**3
    a3 = G * m1 * (np.array([x1, y1, z1]) - np.array([x3, y3, z3])) / r13**3 \
       + G * m2 * (np.array([x2, y2, z2]) - np.array([x3, y3, z3])) / r23**3

    return np.concatenate([
        [vx1, vy1, vz1], a1,
        [vx2, vy2, vz2], a2,
        [vx3, vy3, vz3], a3
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

def rk45_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + dt * (1/4) * k1)
    k3 = derivatives(state + dt * ((3/32)*k1 + (9/32)*k2))
    k4 = derivatives(state + dt * ((1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3))
    k5 = derivatives(state + dt * ((439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4))
    k6 = derivatives(state + dt * ((-8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5))

    y5 = state + dt * ((16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6)

    y4 = state + dt * ((25/216)*k1 + (1408/2565)*k3 + (2197/4104)*k4 - (1/5)*k5)

    return y4, y5

def rk54_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + dt * (1/5) * k1)
    k3 = derivatives(state + dt * ((3/40)*k1 + (9/40)*k2))
    k4 = derivatives(state + dt * ((44/45)*k1 - (56/15)*k2 + (32/9)*k3))
    k5 = derivatives(state + dt * ((19372/6561)*k1 - (25360/2187)*k2 + (64448/6561)*k3 - (212/729)*k4))
    k6 = derivatives(state + dt * ((9017/3168)*k1 - (355/33)*k2 + (46732/5247)*k3 + (49/176)*k4 - (5103/18656)*k5))
    k7 = derivatives(state + dt * ((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6))

    y5 = state + dt * ((35/384)*k1 + (500/1113)*k3 + (125/192)*k4 - (2187/6784)*k5 + (11/84)*k6)

    y4 = state + dt * ((5179/57600)*k1 + (7571/16695)*k3 + (393/640)*k4 - (92097/339200)*k5 + (187/2100)*k6 + (1/40)*k7)

    return y5, y4



def rk6_step_3D(state, dt):
    k1 = derivatives_3D(state)
    k2 = derivatives_3D(state + dt * k1 / 4)
    k3 = derivatives_3D(state + dt * (k1 + k2) / 8)
    k4 = derivatives_3D(state + dt * (-0.5 * k2 + k3))
    k5 = derivatives_3D(state + dt * (3 * k1 + 9 * k4) / 16)
    k6 = derivatives_3D(state + dt * (-3 * k1 + 2 * k2 + 12 * k3 - 12 * k4 + 8 * k5) / 7)
    return state + dt / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5 + 7 * k6)