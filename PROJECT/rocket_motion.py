import numpy as np 
import matplotlib.pyplot as plt

# --- Constants ---
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
burn_time = 100       # duration thrust is active (s)
T = 2e5               # thrust during burn (N)
mdot = (m0 - m_dry) / burn_time  # mass flow rate (kg/s)

# --- Functions ---
def air_density(y):
    return rho0 * np.exp(-y / H)

def acceleration(y, v, t, m, drag=True):
    thrust = T if t <= burn_time else 0
    drag_force = 0.5 * air_density(y) * Cd * A * v * abs(v) if drag else 0.0
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

def simulate_rocket(t_max=800, dt=0.1, drag=True):
    steps = int(t_max / dt)
    time = np.linspace(0, t_max, steps)
    y, v, m = 0.0, 0.0, m0

    ys, vs, accs, energies = np.zeros(steps), np.zeros(steps), np.zeros(steps), np.zeros(steps)
    
    for i in range(steps):
        t = i * dt
        ys[i], vs[i] = y, v
        accs[i] = acceleration(y, v, t, m, drag)
        energies[i] = total_energy(y, v, m)

        y, v = rk4_step(y, v, t, m, dt, drag)
        m = m - mdot * dt if t < burn_time else max(m, m_dry)

    return time, ys, vs, accs, energies


def plot_trajectory(time, ys_drag, ys_nodrag, vs_drag, vs_nodrag, accs_drag, accs_nodrag):
    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, ys_drag / 1000, label="Có lực cản")
    plt.plot(time, ys_nodrag / 1000, label="Không lực cản", linestyle='--')
    plt.ylabel("Độ cao (km)")
    plt.title("Độ cao theo thời gian")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(time, vs_drag, label="Có lực cản")
    plt.plot(time, vs_nodrag, label="Không lực cản", linestyle='--')
    plt.ylabel("Vận tốc (m/s)")
    plt.title("Vận tốc theo thời gian")
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(time, accs_drag, label="Có lực cản")
    plt.plot(time, accs_nodrag, label="Không lực cản", linestyle='--')
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Gia tốc (m/s²)")
    plt.title("Gia tốc theo thời gian")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_energy(time, energies_drag, energies_nodrag):
    plt.figure(figsize=(10, 5))
    plt.plot(time, energies_drag / 1e9, label="Có lực cản")
    plt.plot(time, energies_nodrag / 1e9, linestyle='--', label="Không lực cản")
    plt.axhline(0, color='gray', linestyle='--', label="Mốc năng lượng thoát")
    plt.xlabel("Thời gian (s)")
    plt.ylabel("Năng lượng toàn phần (GJ)")
    plt.title("Năng lượng toàn phần của tên lửa")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    time, ys_drag, vs_drag, accs_drag, energies_drag = simulate_rocket(drag=True)
    _, ys_nodrag, vs_nodrag, accs_nodrag, energies_nodrag = simulate_rocket(drag=False)

    plot_trajectory(time, ys_drag, ys_nodrag, vs_drag, vs_nodrag, accs_drag, accs_nodrag)
    plot_energy(time, energies_drag, energies_nodrag)

main()
