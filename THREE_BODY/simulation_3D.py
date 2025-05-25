import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import rk6_step_3D, rk45_step_3D, rk54_step_3D, rk4_step_3D
from utils import compute_energy_3D
import tkinter as tk
from tkinter import  ttk
from threading import Thread
from matplotlib.widgets import Button
from ui.loading import LoadingUI

# Constants
G = 1.0
dt = 0.0001
num_steps = 300000

rk_methods = {
    "rk4": rk4_step_3D,
    "rk6": rk6_step_3D,
    "rk45": rk45_step_3D,
    "rk54": rk54_step_3D
}

# Global simulation result
positions = np.zeros((num_steps, 3, 3))  # [time][body][x,y,z]
is_adaptive = False
# Initial center/zoom values for animation
prev_center = np.zeros(3)
prev_half_range = 10.0
prev_half_range = 10.0
sphere_surfaces = []

def run_simulation_3D(state, method="rk45"):
    loading = LoadingUI()
    result = {"next": "exit"}
    state = np.array(state, dtype=np.float64).flatten()

    def compute_simulation(method="rk6", dt=0.0001, num_steps=300000):
        global positions
        positions = np.zeros((num_steps, 3, 3))
        rk_step = rk_methods.get(method)

        if method in ["rk45", "rk54"]:
            is_adaptive = True
            adaptive_integrator(rk_step, t0=0, t_end=dt * num_steps, dt_init=dt)
        else:
            for i in range(num_steps):
                for b in range(3):
                    positions[i, b] = state[b * 6: b * 6 + 3] 
                state[:] = rk6_step_3D(state, dt)

                if i % 3000 == 0:
                    update_progress(i / num_steps * 100)

        loading.get_root().after(100, on_finish_simulation)

    def adaptive_integrator(rk_step, t0, t_end, dt_init, tol=1e-6, dt_min=1e-5, dt_max=0.1, record_dt=0.0001):
        global positions, times
        t, dt = t0, dt_init
        next_record_time = t0
        adaptive_positions, adaptive_times = [], []

        while t < t_end:
            dt = min(dt, t_end - t)
            y_high, y_low = rk_step(state, dt)
            R = np.linalg.norm(y_high - y_low) / dt

            if R <= tol or dt <= dt_min:
                t += dt
                state[:] = y_high
                while next_record_time <= t:
                    interp = (next_record_time - (t - dt)) / dt
                    y_interp = (1 - interp) * state + interp * y_high

                    # Lấy x, y, z của mỗi vật
                    pos1 = y_interp[0:3]
                    pos2 = y_interp[6:9]
                    pos3 = y_interp[12:15]

                    adaptive_positions.append([pos1, pos2, pos3])
                    adaptive_times.append(next_record_time)
                    next_record_time += record_dt

            delta = 0.84 * (tol / R) ** 0.25 if R > 0 else 4
            dt *= min(max(delta, 0.1), 4)
            dt = max(dt_min, min(dt, dt_max))

            if dt < dt_min and R > tol:
                print("Minimum dt exceeded, stopping.")
                break

            percent = t / t_end * 100
            update_progress(percent)

        positions = np.array(adaptive_positions)  # shape: (steps, 3, 3)
        times = np.array(adaptive_times)

    def on_finish_simulation():
        loading.close_after_delay(100)
        show_animation()

    def show_animation():
        # Vẽ 3D
        fig = plt.figure(figsize=(10, 8))
       
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-5, 10)
        ax.set_ylim(-5, 10)
        ax.set_zlim(-5, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Three-Body Problem in 3D")
        step = max(1, len(positions) // 1000)
        energy_text = fig.text(0.02, 0.95, "", fontsize=10,
                      verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))
        colors = ['orange', 'blue', 'gray']
        sizes = [0.1, 0.05, 0.05]

        lines = [ax.plot([], [], [], color=c)[0] for c in colors]

        # Sphere mesh (dùng chung)
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_x = np.cos(u) * np.sin(v)
        sphere_y = np.sin(u) * np.sin(v)
        sphere_z = np.cos(v)


        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            global sphere_surfaces
            # Xoá hết các surface nếu có
            for surf in sphere_surfaces:
                surf.remove()
            sphere_surfaces = []
            return lines

        def update(frame):
            global sphere_surfaces, prev_half_range, prev_center

            # Xóa các mặt cầu cũ
            for surf in sphere_surfaces:
                surf.remove()
            sphere_surfaces = []

            # Vẽ đường đi và mặt cầu cho từng vật thể
            for i in range(3):
                x = positions[:frame, i, 0]
                y = positions[:frame, i, 1]
                z = positions[:frame, i, 2]
                lines[i].set_data(x, y)
                lines[i].set_3d_properties(z)

                cx, cy, cz = positions[frame, i]
                r = sizes[i]
                surf = ax.plot_surface(
                    cx + r * sphere_x,
                    cy + r * sphere_y,
                    cz + r * sphere_z,
                    color=colors[i],
                    shade=True,
                    linewidth=0,
                    alpha=1.0
                )
                sphere_surfaces.append(surf)

            # === Tính zoom động (auto-scale view) ===
            all_pos = positions[frame]
            target_center = np.mean(all_pos, axis=0)
            max_dist = max(np.linalg.norm(all_pos[i] - all_pos[j]) for i in range(3) for j in range(i+1, 3))
            target_half_range = max_dist / 2 + 1.0  # thêm khoảng đệm

            # Làm mượt (chậm) quá trình zoom
            alpha = 0.02  # nhỏ hơn = zoom mượt hơn
            prev_center = (1 - alpha) * prev_center + alpha * target_center
            prev_half_range = (1 - alpha) * prev_half_range + alpha * target_half_range

            # Cập nhật không gian hiển thị theo center và zoom hiện tại
            ax.set_xlim(prev_center[0] - prev_half_range, prev_center[0] + prev_half_range)
            ax.set_ylim(prev_center[1] - prev_half_range, prev_center[1] + prev_half_range)
            ax.set_zlim(prev_center[2] - prev_half_range, prev_center[2] + prev_half_range)
            
            current_positions = positions[frame]

            if frame > 0 and frame % step == 0:
                if is_adaptive:
                    dt_real = times[frame] - times[frame - 1]
                else:
                    dt_real = dt
                velocities = (positions[frame] - positions[frame - 1]) / dt_real
                energy = compute_energy_3D(current_positions, velocities)
                energy_text.set_text(f"Energy: {energy:.6f}")

            return lines + sphere_surfaces

        ani = FuncAnimation(
            fig, update,
            frames=range(0, num_steps, 500),
            init_func=init,
            blit=False,
            interval=10,
            repeat=True
        )

        def on_back(event=None):
            result["next"] = "replay"
            plt.close(fig)

        fig.canvas.manager.set_window_title("Three-Body Simulation")
        back_button_ax = fig.add_axes([0.8, 0.01, 0.18, 0.05])
        back_button = Button(back_button_ax, '← Back to Choose')
        back_button.on_clicked(on_back)

        plt.show()




    def update_progress(percent):
        loading.update_progress(percent)

    # Chạy giả lập trong thread
    Thread(target=compute_simulation, kwargs={"method": method, "dt": 0.0001, "num_steps": 300000}, daemon=True).start()
    # Thread(target= update_progress, daemon=True).start()
    # Hiển thị cửa sổ
    loading.run()
    return result["next"]


if __name__ == "__main__":
    # Tạo state ban đầu cho 3 vật thể
    initial_state = np.array([
        -0.85504536,  0.50204268,  0.00267094, -0.94863563,  0.7238842,   0.82469848,
        0.47365597, -0.62878416,  0.64240703, -0.36761325, -0.13233019,  0.6942114,
        -0.5652762,   0.62295359, -0.51591877,  0.5598127,  -0.69730335, -0.35694745
    ])
    
    print("Bắt đầu simulation...")
    result = run_simulation_3D(initial_state)
    print(f"Kết quả: {result}")