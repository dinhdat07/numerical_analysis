import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import rk6_step, compute_energy
import tkinter as tk
from tkinter import ttk
from threading import Thread
import time
from matplotlib.widgets import Button

G = 1.0
dt = 0.0001
num_steps = 300000

positions = np.zeros((num_steps, 3, 2))

def run_simulation(state):
    result = {"next": "exit"}

    def compute_simulation():
        for i in range(num_steps):
            positions[i, 0] = state[0:2]
            positions[i, 1] = state[4:6]
            positions[i, 2] = state[8:10]
            state[:] = rk6_step(state, dt)
            if i % 3000 == 0:
                progress_var.set(i / num_steps * 100)
                loading.update_idletasks()
        loading.after(100, on_finish_simulation)  

    def on_finish_simulation():
        loading.destroy()
        show_animation()  

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

            if frame % 300 == 0:
                velocities = (positions[frame] - positions[frame - 1]) / dt
                energy = compute_energy(current_positions, velocities)
                energy_text.set_text(f"Energy: {energy:.6f}")

            return update_lines_and_dots(frame) + [energy_text]

        ani = FuncAnimation(fig, update, frames=range(0, num_steps, 300),
                            init_func=init, blit=True, interval=10, repeat=True)
        
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

    # Tùy chỉnh style cho Progressbar
    style = ttk.Style()
    style.theme_use('clam')  # dùng theme nhẹ
    style.configure("TProgressbar",
                    troughcolor='#34495e',
                    background='#1abc9c',
                    thickness=20,
                    bordercolor='#2c3e50',
                    relief='flat')

    # Tiêu đề
    title_label = tk.Label(loading, text="⏳ Đang xử lý, vui lòng chờ...", 
                        font=("Segoe UI", 12, "bold"), 
                        bg="#2c3e50", fg="white")
    title_label.pack(pady=(30, 15))

    # Thanh tiến trình
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(loading, length=300, mode='determinate',
                                variable=progress_var, style="TProgressbar")
    progress_bar.pack(pady=10)

    # Phần trăm tiến trình (tuỳ chọn)
    percent_label = tk.Label(loading, text="0%", font=("Segoe UI", 10),
                            bg="#2c3e50", fg="white")
    percent_label.pack()

    # Cập nhật phần trăm hiển thị
    def update_percent():
        while True:
            value = progress_var.get()
            percent_label.config(text=f"{int(value)}%")
            if value >= 100:
                break
            time.sleep(0.05)

    # Chạy giả lập trong thread
    Thread(target=compute_simulation, daemon=True).start()
    Thread(target=update_percent, daemon=True).start()

    # Hiển thị cửa sổ
    loading.mainloop()
    return result["next"]

