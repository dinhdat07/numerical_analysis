import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import rk6_step, rk4_step, rk54_step, rk45_step, compute_energy
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
from matplotlib.widgets import Button

G = 1.0
dt = 0.0001
num_steps = 300000

rk_methods = {
    "rk4": rk4_step,
    "rk6": rk6_step,
    "rk45": rk45_step,
    "rk54": rk54_step
}

positions = None
times = None
result = {"next": "exit"}
is_adaptive = False

def run_simulation(state):
    def compute_simulation(method="rk6", dt=0.0001, num_steps=300000):
        global positions, times, is_adaptive
        rk_step = rk_methods.get(method)
        if rk_step is None:
            raise ValueError(f"Method '{method}' is invalid. Please choose: {list(rk_methods.keys())}")

        if method in ["rk45", "rk54"]:
            is_adaptive = True
            t = 0
            t_end = dt * num_steps
            dt_adapt = dt
            tol = 1e-8
            dt_min = 1e-8
            dt_max = 0.001
            record_dt = 0.0001
            next_record_time = t
            adaptive_positions, adaptive_times = [], []

            percent_prev = 0

            while t < t_end:
                dt_adapt = min(dt_adapt, t_end - t)

                prev_state = state.copy()  # Lưu trạng thái trước khi cập nhật
                y_high, y_low = rk_step(state, dt_adapt)
                R = np.linalg.norm(y_high - y_low) / dt_adapt

                if R <= tol or dt_adapt <= dt_min:
                    t += dt_adapt
                    state[:] = y_high
                    while next_record_time <= t:
                        interp = (next_record_time - (t - dt_adapt)) / dt_adapt
                        y_interp = (1 - interp) * prev_state + interp * y_high
                        adaptive_positions.append([y_interp[0:2], y_interp[4:6], y_interp[8:10]])
                        adaptive_times.append(next_record_time)
                        next_record_time += record_dt

                delta = 0.84 * (tol / R) ** 0.25 if R > 0 else 4
                dt_adapt *= min(max(delta, 0.1), 4)
                dt_adapt = max(dt_min, min(dt_adapt, dt_max))

                if dt_adapt < dt_min and R > tol:
                    print("Minimum dt exceeded, stopping.")
                    break

                percent = int(t / t_end * 100)
                if percent != percent_prev:
                    update_progress(percent)
                    percent_prev = percent

            positions = np.array(adaptive_positions)
            times = np.array(adaptive_times)

        else:
            positions = np.zeros((num_steps, 3, 2))
            percent_prev = 0
            for i in range(num_steps):
                positions[i, 0] = state[0:2]
                positions[i, 1] = state[4:6]
                positions[i, 2] = state[8:10]
                state[:] = rk_step(state, dt)

                percent = int(i / num_steps * 100)
                if percent != percent_prev and i % 1000 == 0:
                    update_progress(percent)
                    percent_prev = percent

        loading.after(100, on_finish_simulation)

    def on_finish_simulation():
        loading.destroy()
        show_animation()

    def update_progress(percent):
        loading.after(0, lambda: (
            progress_var.set(percent),
            loading.update_idletasks()
        ))

    def show_animation():
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['r', 'g', 'b']
        lines = [ax.plot([], [], color)[0] for color in colors]
        dots = [ax.plot([], [], color + 'o')[0] for color in colors]
        energy_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
                              verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_title("Three-Body Problem")
        step = max(1, len(positions) // 1000)
        print(f"Total frames: {len(positions)}, Step: {step}")

        def init():
            for line, dot in zip(lines, dots):
                line.set_data([], [])
                dot.set_data([], [])
            return lines + dots

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

        collision_state = {"detected": False, "frame": None}

        def update(frame):
            if collision_state["detected"]:
                return update_lines_and_dots(collision_state["frame"]) + [energy_text]

            current_positions = positions[frame]

            if detect_collision(current_positions):
                collision_state["detected"] = True
                collision_state["frame"] = frame
                ax.set_title("Collision detected!", color='red')
                fig.canvas.draw_idle()
                return update_lines_and_dots(frame) + [energy_text]
            

            if frame > 0 and frame % step == 0:
                if is_adaptive:
                    dt_real = times[frame] - times[frame - 1]
                else:
                    dt_real = dt
                velocities = (positions[frame] - positions[frame - 1]) / dt_real
                energy = compute_energy(current_positions, velocities)
                energy_text.set_text(f"Energy: {energy:.6f}")

            return update_lines_and_dots(frame) + [energy_text]
        
        
        ani = FuncAnimation(fig, update, frames=range(0, len(positions), step),
                            init_func=init, blit=True, interval=10, repeat=True)
        
        
        # Add a back button to return to the choose state screen
        def on_back(event=None):
            result["next"] = "replay"
            plt.close(fig)

        fig.canvas.manager.set_window_title("Three-Body Simulation")
        back_button_ax = fig.add_axes([0.8, 0.01, 0.18, 0.05])
        back_button = Button(back_button_ax, '← Back to Choose')
        back_button.on_clicked(on_back)

        plt.show()

    # loading UI
    loading = tk.Tk()
    loading.title("Đang tải...")
    loading.geometry("400x180")
    loading.configure(bg="#2c3e50")  # màu nền tối
    loading.resizable(False, False)
    # Style
    style = ttk.Style()
    style.theme_use('clam')  # dùng theme nhẹ
    style.configure("TProgressbar",
                    troughcolor='#34495e',
                    background='#1abc9c',
                    thickness=20,
                    bordercolor='#2c3e50',
                    relief='flat')
    # Title
    title_label = tk.Label(loading, text="⏳ Đang xử lý, vui lòng chờ...", 
                        font=("Segoe UI", 12, "bold"), 
                        bg="#2c3e50", fg="white")
    title_label.pack(pady=(30, 15))

    # Progessbar
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(loading, length=300, mode='determinate',
                                variable=progress_var, style="TProgressbar")
    progress_bar.pack(pady=10)

    # Percent
    percent_label = tk.Label(loading, text="0%", font=("Segoe UI", 10),
                            bg="#2c3e50", fg="white")
    percent_label.pack()


    def update_percent():
        while True:
            value = progress_var.get()
            percent_label.config(text=f"{int(value)}%")
            if value >= 100:
                break
            time.sleep(0.05)


    # Chạy giả lập trong thread
    Thread(target=compute_simulation, kwargs={"method": "rk6", "dt": 0.0001, "num_steps": 300000}, daemon=True).start()
    Thread(target=update_percent, daemon=True).start()

    # Hiển thị cửa sổ
    loading.mainloop() # Chờ cửa sổ loading đóng, không cần mainloop()

