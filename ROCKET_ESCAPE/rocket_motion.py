from matplotlib import animation
import numpy as np 
import matplotlib.pyplot as plt


# --- Constants ---
G = 6.67430e-11       # gravitational constant
M = 5.972e24          # mass of Earth (kg)
R = 6.371e6           # radius of Earth (m)
rho0 = 1.225          # air density at sea level (kg/m^3)
H = 8000              # scale height of atmosphere (m)
Cd = 0.5              # drag coefficient
A = 10.5              # cross-sectional area (m^2)

# Stage 1 rocket parameters
m0 = 300000            # initial mass including fuel (kg)
m_dry = 150000          # dry mass after fuel burns out (kg)
burn_time = 100       # duration thrust is active (s)
T = 2.55e7               # thrust during burn (N)
mdot = (m0 - m_dry) / burn_time  # mass flow rate (kg/s)

# --- Functions ---

def gravity(y):
    return -G * M / (R + y) ** 2

def air_density(y):
    return rho0 * np.exp(-y / H)

def drag_force(vx, vy, y):
    v = np.sqrt(vx**2 + vy**2)
    rho = air_density(y)
    return 0.5 * rho * Cd * A * v**2 if v != 0 else 0


def acceleration(x, y, vx, vy, t, m, drag=True, theta=0):
    v = np.sqrt(vx**2 + vy**2)
    if v == 0: v = 1e-6  # avoid division by zero
    unit_vx, unit_vy = vx / v, vy / v
    rho = air_density(y)
    F_drag_x = 0.5 * rho * Cd * A * v * vx if drag else 0
    F_drag_y = 0.5 * rho * Cd * A * v * vy if drag else 0

    thrust = T if t <= burn_time else 0
    thrust_x = thrust * np.cos(theta)
    thrust_y = thrust * np.sin(theta)

    ax = (thrust_x - F_drag_x) / m
    ay = (thrust_y - F_drag_y) / m - G * M / (R + y)**2
    return ax, ay

def total_energy(x, y, vx, vy, m):
    v = np.sqrt(vx**2 + vy**2)
    r = np.sqrt(x**2 + (R + y)**2)
    kinetic = 0.5 * m * v**2
    potential = -G * M * m / r
    return kinetic + potential

def rk4_step(x, y, vx, vy, t, m, dt, drag=True, theta=0):
    def f_vx(x, y, vx, vy): return vx
    def f_vy(x, y, vx, vy): return vy
    def g_ax(x, y, vx, vy): return acceleration(x, y, vx, vy, t, m, drag, theta)[0]
    def g_ay(x, y, vx, vy): return acceleration(x, y, vx, vy, t, m, drag, theta)[1]

    k1x = dt * f_vx(x, y, vx, vy)
    k1y = dt * f_vy(x, y, vx, vy)
    k1vx = dt * g_ax(x, y, vx, vy)
    k1vy = dt * g_ay(x, y, vx, vy)

    k2x = dt * f_vx(x + 0.5 * k1x, y + 0.5 * k1y, vx + 0.5 * k1vx, vy + 0.5 * k1vy)
    k2y = dt * f_vy(x + 0.5 * k1x, y + 0.5 * k1y, vx + 0.5 * k1vx, vy + 0.5 * k1vy)
    k2vx = dt * g_ax(x + 0.5 * k1x, y + 0.5 * k1y, vx + 0.5 * k1vx, vy + 0.5 * k1vy)
    k2vy = dt * g_ay(x + 0.5 * k1x, y + 0.5 * k1y, vx + 0.5 * k1vx, vy + 0.5 * k1vy)

    k3x = dt * f_vx(x + 0.5 * k2x, y + 0.5 * k2y, vx + 0.5 * k2vx, vy + 0.5 * k2vy)
    k3y = dt * f_vy(x + 0.5 * k2x, y + 0.5 * k2y, vx + 0.5 * k2vx, vy + 0.5 * k2vy)
    k3vx = dt * g_ax(x + 0.5 * k2x, y + 0.5 * k2y, vx + 0.5 * k2vx, vy + 0.5 * k2vy)
    k3vy = dt * g_ay(x + 0.5 * k2x, y + 0.5 * k2y, vx + 0.5 * k2vx, vy + 0.5 * k2vy)

    k4x = dt * f_vx(x + k3x, y + k3y, vx + k3vx, vy + k3vy)
    k4y = dt * f_vy(x + k3x, y + k3y, vx + k3vx, vy + k3vy)
    k4vx = dt * g_ax(x + k3x, y + k3y, vx + k3vx, vy + k3vy)
    k4vy = dt * g_ay(x + k3x, y + k3y, vx + k3vx, vy + k3vy)

    x_next = x + (k1x + 2*k2x + 2*k3x + k4x) / 6
    y_next = y + (k1y + 2*k2y + 2*k3y + k4y) / 6
    vx_next = vx + (k1vx + 2*k2vx + 2*k3vx + k4vx) / 6
    vy_next = vy + (k1vy + 2*k2vy + 2*k3vy + k4vy) / 6

    return x_next, y_next, vx_next, vy_next

def simulate_rocket(t_max=500, dt=0.5, drag=True, theta_deg=75):
    theta = np.radians(theta_deg)
    steps = int(t_max / dt)
    x, y, vx, vy, m = 0.0, 0.0, np.cos(theta) * 100, np.sin(theta) * 100, m0

    xs, ys, vxs, vys, time, masses = [], [], [], [], [], []

    for i in range(steps):
        t = i * dt
        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        masses.append(m)
        time.append(t)

        x, y, vx, vy = rk4_step(x, y, vx, vy, t, m, dt, drag, theta)
        m = m - mdot * dt if t < burn_time else max(m, m_dry)

        if y < 0:
            break

    return np.array(xs), np.array(ys), np.array(vxs), np.array(vys), np.array(time), np.array(masses)




def plot_motion(time_drag, ys_drag, time_nodrag, ys_nodrag,
                    vs_drag, vs_nodrag, accs_drag, accs_nodrag):

    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_drag, ys_drag / 1000, label="Có lực cản")
    plt.plot(time_nodrag, ys_nodrag / 1000, label="Không lực cản", linestyle='--')
    plt.ylabel("Độ cao (km)")
    plt.title("Độ cao theo thời gian")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(3, 1, 2)
    plt.plot(time_drag, vs_drag, label="Có lực cản")
    plt.plot(time_nodrag, vs_nodrag, label="Không lực cản", linestyle='--')
    plt.ylabel("Vận tốc (m/s)")
    plt.title("Vận tốc theo thời gian")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(3, 1, 3)
    plt.plot(time_drag, accs_drag, label="Có lực cản")
    plt.plot(time_nodrag, accs_nodrag, label="Không lực cản", linestyle='--')
    plt.ylabel("Gia tốc (m/s²)")
    plt.xlabel("Thời gian (s)")
    plt.title("Gia tốc theo thời gian")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_trajectory_2D(xs, ys, label):
    plt.figure(figsize=(8, 6))
    plt.plot(xs / 1000, ys / 1000, label=label)
    plt.xlabel("Quãng đường ngang (km)")
    plt.ylabel("Độ cao (km)")
    plt.title("Quỹ đạo tên lửa")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

def plot_energy(time_drag, energies_drag, time_nodrag, energies_nodrag):
    energies_drag_GJ = energies_drag / 1e9
    energies_nodrag_GJ = energies_nodrag / 1e9

    min_len = min(len(time_drag), len(time_nodrag))
    time = time_drag[:min_len]
    diff = energies_nodrag_GJ[:min_len] - energies_drag_GJ[:min_len]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time_drag, energies_drag_GJ, label="Có lực cản")
    axs[0].plot(time_nodrag, energies_nodrag_GJ, linestyle='--', label="Không lực cản")
    axs[0].axhline(0, color='gray', linestyle='--', label="Mốc năng lượng thoát")
    axs[0].set_ylabel("Năng lượng toàn phần (GJ)")
    axs[0].set_title("Năng lượng toàn phần của tên lửa")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    status_drag = "KHÔNG ĐẠT" if np.max(energies_drag_GJ) < 0 else "ĐÃ ĐẠT"
    status_nodrag = "KHÔNG ĐẠT" if np.max(energies_nodrag_GJ) < 0 else "ĐÃ ĐẠT"

    axs[0].text(0.6 * time[-1], 0.1 * max(np.max(energies_drag_GJ), np.max(energies_nodrag_GJ)), 
                f"Có lực cản: {status_drag}", color='blue')

    axs[0].text(0.6 * time[-1], -0.2 * max(abs(np.min(energies_drag_GJ)), abs(np.min(energies_nodrag_GJ))),
                f"Không lực cản: {status_nodrag}", color='orange')
    
    axs[1].plot(time, diff, color='red')
    axs[1].set_xlabel("Thời gian (s)")
    axs[1].set_ylabel("Chênh lệch năng lượng (GJ)")
    axs[1].set_title("Chênh lệch năng lượng: Không lực cản - Có lực cản")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def animate_trajectory(xs, ys):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, max(xs)/1000 + 10)
    ax.set_ylim(0, max(ys)/1000 + 10)
    ax.set_xlabel("Quãng đường (km)")
    ax.set_ylabel("Độ cao (km)")
    ax.set_title("Mô phỏng quỹ đạo tên lửa")
    ax.grid(True, linestyle='--', alpha=0.5)

    point, = ax.plot([], [], 'ro')  # điểm chuyển động
    path, = ax.plot([], [], 'b-', linewidth=1)  # quỹ đạo đã đi qua

    def update(frame):
        point.set_data([xs[frame]/1000], [ys[frame]/1000])
        path.set_data(xs[:frame+1]/1000, ys[:frame+1]/1000)
        return point, path

    ani = animation.FuncAnimation(fig, update, frames=len(xs), interval=30, blit=True)
    plt.show()

def main():
    xs_d, ys_d, vxs_d, vys_d, t_d, ms_d = simulate_rocket(drag=True)
    xs_n, ys_n, vxs_n, vys_n, t_n, ms_n = simulate_rocket(drag=False)
    energies_drag = total_energy(xs_d, ys_d, vxs_d, vys_d, ms_d)
    energies_nodrag = total_energy(xs_n, ys_n, vxs_n, vys_n, ms_n)
    energies_drag_GJ = energies_drag / 1e9
    energies_nodrag_GJ = energies_nodrag / 1e9

    # Vẽ quỹ đạo
    plot_trajectory_2D(xs_d, ys_d, label="Có lực cản")
    plot_trajectory_2D(xs_n, ys_n, label="Không lực cản")
    plot_energy(t_d, energies_drag_GJ, t_n, energies_nodrag_GJ)

    # Tạo animation
    animate_trajectory(xs_d, ys_d)

main()
