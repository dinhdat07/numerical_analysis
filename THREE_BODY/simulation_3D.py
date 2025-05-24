import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import rk6_step_3D
import tkinter as tk
from tkinter import  ttk
from threading import Thread
from matplotlib.widgets import Button

# Constants
G = 1.0
dt = 0.0001
num_steps = 300000

# Global simulation result
positions = np.zeros((num_steps, 3, 3))  # [time][body][x,y,z]

# Initial center/zoom values for animation
prev_center = np.zeros(3)
prev_half_range = 10.0
prev_half_range = 10.0
sphere_surfaces = []

def run_simulation_3D(state):
    result = {"next": "exit"}
    state = np.array(state, dtype=np.float64).flatten()

    def compute_simulation():
        global positions
        scale = 5
        positions = np.zeros((num_steps, 3, 3))

        for i in range(num_steps):
            for b in range(3):
                positions[i, b] = state[b * 6: b * 6 + 3] * scale
            state[:] = rk6_step_3D(state, dt)

            if i % 3000 == 0:
                progress_var.set(i / num_steps * 100)
                loading.update_idletasks()

        loading.after(100, on_finish_simulation)

    def on_finish_simulation():
        loading.destroy()
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

        colors = ['orange', 'blue', 'gray']
        sizes = [0.2, 0.1, 0.1]

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
