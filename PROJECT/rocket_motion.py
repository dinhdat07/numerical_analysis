import numpy as np 
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11       # gravitational constant
M = 5.972e24          # mass of Earth (kg)
R = 6.371e6           # radius of Earth (m)
rho0 = 1.225          # air density at sea level (kg/m^3)
H = 8000              # scale height of atmosphere (m)
Cd = 1.5              # drag coefficient
A = 1.0               # cross-sectional area (m^2)

# Stage 1 rocket parameters
m0 = 10000            # initial mass including fuel (kg)
m_dry = 3000          # dry mass after fuel burns out (kg)
burn_time = 120       # duration thrust is active (s)
T = 2e5               # thrust during burn (N)
mdot = (m0 - m_dry) / burn_time  # mass flow rate (kg/s)

# Functions
def air_density(y):
    return rho0 * np.exp(-y / H)

def acceleration(y, v, t, m, drag=True):
    thrust = T if t <= burn_time else 0
    if drag:
        drag_force = 0.5 * air_density(y) * Cd * A * v * abs(v)
    else:
        drag_force = 0.0

    gravity = G * M / (R + y)**2
    return (thrust - drag_force) / m - gravity

def total_energy(y, v, m):
    kinetic = 0.5 * m * v**2
    potential = -G * M * m / (R + y)
    return kinetic + potential

def rk4_step(y, v, t, m, dt, drag=True):
    def f(y, v): return v
    def g(y, v): return acceleration(y, v, t, m, drag)

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

# Time setup
t_max = 600           # total simulation time (s)
dt = 0.1
steps = int(t_max / dt)
time = np.linspace(0, t_max, steps)

# Init arrays
y = 0.0
v = 0.0
m = m0
ys = np.zeros(steps)
vs = np.zeros(steps)
accs = np.zeros(steps)
energies = np.zeros(steps)

# Simulation loop
for i in range(steps):
    t = i * dt
    ys[i] = y
    vs[i] = v
    accs[i] = acceleration(y, v, t, m, drag=True)
    energies[i] = total_energy(y, v, m)
    y, v = rk4_step(y, v, t, m, dt, drag=True)
    
    if t < burn_time:
        m -= mdot * dt
    else:
        m = max(m, m_dry)

# --- Plotting ---
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(time, ys / 1000)
plt.ylabel("Độ cao (km)")
plt.title("Độ cao theo thời gian")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(time, vs)
plt.ylabel("Vận tốc (m/s)")
plt.title("Vận tốc theo thời gian (có ngừng đẩy sau 120s)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(time, accs)
plt.xlabel("Thời gian (s)")
plt.ylabel("Gia tốc (m/s²)")
plt.title("Gia tốc theo thời gian")
plt.grid()

plt.tight_layout()
plt.show()

# Plot energy
plt.figure(figsize=(10, 5))
plt.plot(time, energies / 1e9)
plt.axhline(0, color='gray', linestyle='--', label="Mốc năng lượng thoát")
plt.xlabel("Thời gian (s)")
plt.ylabel("Năng lượng toàn phần (GJ)")
plt.title("Năng lượng toàn phần của tên lửa")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
