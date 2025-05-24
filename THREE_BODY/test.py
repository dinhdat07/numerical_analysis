import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time
from multiprocessing import Process
from utils import rk4_step, rk54_step, rk45_step, rk6_step

def compute_error(y_test, y_ref):
    return np.linalg.norm(y_test - y_ref)

def init_states():
    return [
        np.array([
            0.97000436, -0.24308753, -0.46620368, -0.43236573,
           -0.97000436,  0.24308753, -0.46620368, -0.43236573,
            0.0,          0.0,         0.93240737,  0.86473146
        ]),
        np.array([
            1.0, 0.0, 0.0, 0.5,
            -0.5, np.sqrt(3)/2, -0.5*np.sqrt(3), -0.25,
            -0.5, -np.sqrt(3)/2, 0.5*np.sqrt(3), -0.25
        ]),
        np.array([
            -1.0, 0.0, 0.306893, 0.125507,
             1.0, 0.0, 0.306893, 0.125507,
             0.0, 0.0, -0.613786, -0.251014
        ]),
    ]

def solve(method, state0, dt, T):
    steps = int(T / dt)
    states = [state0]
    state = state0.copy()
    for _ in range(steps):
        state = method(state, dt)
        states.append(state)
    return np.array(states)

# Hàm chạy độc lập từng state
def run_for_state(idx, state0, T, dt_ref, dt_test, methods):
    print(f"\n==== Running for Initial State #{idx+1} ====")

    print("Computing reference solution (RK54)...")
    start_ref = time.time()
    ref_states = solve(rk54_step, state0, dt_ref, T)
    end_ref = time.time()
    print(f"Time for reference RK54 solution: {end_ref - start_ref:.4f} seconds")

    time_ref = np.linspace(0, T, int(T/dt_ref) + 1)
    plt.figure(figsize=(10, 6))

    for name, method in methods.items():
        print(f"\nComputing solution with {name}...")
        start_time = time.time()
        test_states = solve(method, state0, dt_test, T)
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Time for {name}: {elapsed_time:.4f} seconds")

        time_test = np.linspace(0, T, int(T/dt_test) + 1)
        interp_ref = interp1d(time_ref, ref_states, axis=0)
        ref_at_test_times = interp_ref(time_test)

        errors = [compute_error(y_t, y_r) for y_t, y_r in zip(test_states, ref_at_test_times)]

        error_L2 = np.sqrt(np.mean(np.array(errors) ** 2))
        error_Linf = np.max(errors)

        print(f"{name} L2 error: {error_L2:.4e}")
        print(f"{name} L_inf error: {error_Linf:.4e}")

        plt.plot(time_test, errors, label=f'{name}')

    plt.xlabel('Time')
    plt.ylabel('Error (Euclidean norm)')
    plt.yscale('log')
    plt.title(f'Sai số cho Initial State #{idx+1}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'error_state_{idx+1}.png')  # Lưu file ảnh thay vì show nếu chạy song song
    print(f"Saved plot for Initial State #{idx+1}.")

if __name__ == "__main__":
    T = 100
    dt_ref = 0.001
    dt_test = 0.01

    state_list = init_states()
    methods = {
        "RK4": rk4_step,
        "RK45": rk45_step,
        "RK6": rk6_step,
        "RK54": rk54_step
    }

    processes = []
    for idx, state0 in enumerate(state_list):
        p = Process(target=run_for_state, args=(idx, state0, T, dt_ref, dt_test, methods))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n=== All simulations finished ===")
    print("\n=== All simulations finished ===")

    # Hiển thị ảnh sau khi tất cả tiến trình đã hoàn tất
    import matplotlib.image as mpimg

    for idx in range(len(state_list)):
        img = mpimg.imread(f'error_state_{idx+1}.png')
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')  # Ẩn trục
        plt.title(f'Error Plot for Initial State #{idx+1}')
        plt.show()