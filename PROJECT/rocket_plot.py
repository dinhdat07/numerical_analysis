# rocket_plot.py

import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(time_drag, ys_drag, time_nodrag, ys_nodrag,
                    vs_drag, vs_nodrag, accs_drag, accs_nodrag):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time_drag, ys_drag / 1000, label="Có lực cản")
    plt.plot(time_nodrag, ys_nodrag / 1000, linestyle='--', label="Không lực cản")
    plt.ylabel("Độ cao (km)")
    plt.title("Độ cao theo thời gian")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(time_drag, vs_drag, label="Có lực cản")
    plt.plot(time_nodrag, vs_nodrag, linestyle='--', label="Không lực cản")
    plt.ylabel("Vận tốc (m/s)")
    plt.title("Vận tốc theo thời gian")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(time_drag, accs_drag, label="Có lực cản")
    plt.plot(time_nodrag, accs_nodrag, linestyle='--', label="Không lực cản")
    plt.ylabel("Gia tốc (m/s²)")
    plt.xlabel("Thời gian (s)")
    plt.title("Gia tốc theo thời gian")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_energy(time_drag, e_drag, time_nodrag, e_nodrag):
    e_drag /= 1e9
    e_nodrag /= 1e9
    min_len = min(len(time_drag), len(time_nodrag))
    time = time_drag[:min_len]
    diff = e_nodrag[:min_len] - e_drag[:min_len]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time_drag, e_drag, label="Có lực cản")
    axs[0].plot(time_nodrag, e_nodrag, linestyle='--', label="Không lực cản")
    axs[0].axhline(0, color='gray', linestyle='--')
    axs[0].set_ylabel("Năng lượng (GJ)")
    axs[0].legend()
    axs[0].grid(True)

    status_drag = "ĐẠT" if np.max(e_drag) > 0 else "KHÔNG ĐẠT"
    status_nodrag = "ĐẠT" if np.max(e_nodrag) > 0 else "KHÔNG ĐẠT"

    axs[0].text(0.7 * time[-1], max(np.max(e_drag), np.max(e_nodrag)) * 0.8,
                f"Có lực cản: {status_drag}", color="blue")
    axs[0].text(0.7 * time[-1], -max(abs(np.min(e_drag)), abs(np.min(e_nodrag))) * 0.8,
                f"Không lực cản: {status_nodrag}", color="orange")

    axs[1].plot(time, diff, color='red')
    axs[1].set_xlabel("Thời gian (s)")
    axs[1].set_ylabel("Chênh lệch năng lượng (GJ)")
    axs[1].set_title("Không lực cản - Có lực cản")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
