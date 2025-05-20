import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11           # gravitational constant (m^3 kg^-1 s^-2)
M = 5.972e24              # mass of Earth (kg)
R = 6.371e6               # radius of Earth (m)
rho0 = 1.225              # air density at sea level (kg/m^3)
H = 8000                  # scale height of atmosphere (m)
Cd = 0.5                  # drag coefficient
A = 0.3                   # cross-sectional area of the rocket (m^2)
m = 1000                  # mass of the rocket (kg)
T = 20000                 # constant thrust (N)

# Functions
def air_density(y):
    return rho0 * np.exp(-y / H)

def acceleration(y, v):
    drag = 0.5 * air_density(y) * Cd * A * v * abs(v)
    gravity = G * M / (R + y)**2
    return (T - drag) / m - gravity

# RK4 solver
def rk4_step(y, v, dt):
    def f(y, v): return v
    def g(y, v): return acceleration(y, v)

    k1y = dt * f(y, v)
    k1v = dt * g(y, v)

    k2y = dt * f(y + 0.5 * k1y, v + 0.5 * k1v)
    k2v = dt * g(y + 0.5 * k1y, v + 0.5 * k1v)

    k3y = dt * f(y + 0.5 * k2y, v + 0.5 * k2v)
    k3v = dt * g(y + 0.5 * k2y, v + 0.5 * k2v)

    k4y = dt * f(y + k3y, v + k3v)
    k4v = dt * g(y + k3y, v + k3v)

    y_next = y + (k1y + 2*k2y + 2*k3y + k4y) / 6
    v_next = v + (k1v + 2*k2v + 2*k3v + k4v) / 6

    return y_next, v_next

# Time integration
t_max = 300  # seconds
dt = 0.1
steps = int(t_max / dt)

time = np.linspace(0, t_max, steps)
ys = np.zeros(steps)
vs = np.zeros(steps)

y, v = 0.0, 0.0
for i in range(steps):
    ys[i] = y
    vs[i] = v
    y, v = rk4_step(y, v, dt)

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time, ys / 1000)
plt.xlabel("Thời gian (s)")
plt.ylabel("Độ cao (km)")
plt.title("Độ cao theo thời gian")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time, vs)
plt.xlabel("Thời gian (s)")
plt.ylabel("Vận tốc (m/s)")
plt.title("Vận tốc theo thời gian")
plt.grid(True)

plt.tight_layout()
plt.show()
