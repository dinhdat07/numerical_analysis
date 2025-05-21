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
def air_density(y):
    return rho0 * np.exp(-y / H)

def acceleration(y, v, t, m, drag=True):
    thrust = T if t <= burn_time else 0
    drag_force = 0.5 * air_density(y) * Cd * A * v**2 * np.sign(v) if drag else 0.0
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

def simulate_rocket(t_max=1600, dt=0.1, drag=True):
    steps = int(t_max / dt)
    y, v, m = 0.0, 0.0, m0
    y_prev = y 
    energy_loss = 0  

    ys, vs, accs, energies, time = [], [], [], [], []

    for i in range(steps):
        t = i * dt
        ys.append(y)
        vs.append(v)
        accs.append(acceleration(y, v, t, m, drag))

        total_mechanical = total_energy(y, v, m)

        if drag:
            F_drag = 0.5 * air_density(y) * v**2 * Cd * A * np.sign(v)
            dy = y - y_prev
            energy_loss += F_drag * dy 
            total_mechanical -= energy_loss  
            y_prev = y

        energies.append(total_mechanical)
        time.append(t)

        y, v = rk4_step(y, v, t, m, dt, drag)
        m = m - mdot * dt if t < burn_time else max(m, m_dry)

        if y < 0:
            break

    return np.array(time), np.array(ys), np.array(vs), np.array(accs), np.array(energies)




def plot_trajectory(time_drag, ys_drag, time_nodrag, ys_nodrag,
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


def plot_energy(time_drag, energies_drag, time_nodrag, energies_nodrag):
    # Chuyển sang GJ
    energies_drag_GJ = energies_drag / 1e9
    energies_nodrag_GJ = energies_nodrag / 1e9

    # Đảm bảo time_drag và time_nodrag có cùng độ dài
    min_len = min(len(time_drag), len(time_nodrag))
    time = time_drag[:min_len]
    diff = energies_nodrag_GJ[:min_len] - energies_drag_GJ[:min_len]

    # Tạo 2 subplot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Biểu đồ chính: năng lượng toàn phần
    axs[0].plot(time_drag, energies_drag_GJ, label="Có lực cản")
    axs[0].plot(time_nodrag, energies_nodrag_GJ, linestyle='--', label="Không lực cản")
    axs[0].axhline(0, color='gray', linestyle='--', label="Mốc năng lượng thoát")
    axs[0].set_ylabel("Năng lượng toàn phần (GJ)")
    axs[0].set_title("Năng lượng toàn phần của tên lửa")
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.5)

    # ✅ THÊM ghi chú đạt mốc năng lượng
    status_drag = "KHÔNG ĐẠT" if np.max(energies_drag_GJ) < 0 else "ĐÃ ĐẠT"
    status_nodrag = "KHÔNG ĐẠT" if np.max(energies_nodrag_GJ) < 0 else "ĐÃ ĐẠT"

    axs[0].text(0.6 * time[-1], 0.1 * max(np.max(energies_drag_GJ), np.max(energies_nodrag_GJ)), 
                f"Có lực cản: {status_drag}", color='blue')

    axs[0].text(0.6 * time[-1], -0.2 * max(abs(np.min(energies_drag_GJ)), abs(np.min(energies_nodrag_GJ))),
                f"Không lực cản: {status_nodrag}", color='orange')
    

    # Biểu đồ phụ: chênh lệch năng lượng
    axs[1].plot(time, diff, color='red')
    axs[1].set_xlabel("Thời gian (s)")
    axs[1].set_ylabel("Chênh lệch năng lượng (GJ)")
    axs[1].set_title("Chênh lệch năng lượng: Không lực cản - Có lực cản")
    axs[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()



def main():
    time_drag, ys_drag, vs_drag, accs_drag, energies_drag = simulate_rocket(drag=True)
    time_nodrag, ys_nodrag, vs_nodrag, accs_nodrag, energies_nodrag = simulate_rocket(drag=False)

    plot_trajectory(time_drag, ys_drag, time_nodrag, ys_nodrag,
                    vs_drag, vs_nodrag, accs_drag, accs_nodrag)

    plot_energy(time_drag, energies_drag, time_nodrag, energies_nodrag)

main()
