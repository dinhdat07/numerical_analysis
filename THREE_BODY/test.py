import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import rk4_step, rk54_step, rk45_step, rk6_step

# Hàm tính sai số Euclid
def compute_error(y_test, y_ref):
    return np.linalg.norm(y_test - y_ref)

def init_state():
    return np.array([
         0.97000436, -0.24308753, -0.46620368, -0.43236573,  
        -0.97000436,  0.24308753, -0.46620368, -0.43236573,  
         0.0,          0.0,         0.93240737,  0.86473146
    ])

# Giải bài toán với phương pháp cho trước
def solve(method, state0, dt, T):
    steps = int(T / dt)
    states = [state0]
    state = state0.copy()
    for _ in range(steps):
        state = method(state, dt)
        states.append(state)
    return np.array(states)

if __name__ == "__main__":
    T = 100
    dt_ref = 0.001  # Bước rất nhỏ cho nghiệm chuẩn (RK54)
    dt_test = 0.01   # Bước lớn cho các phương pháp test

    state0 = init_state()

    # Nghiệm chuẩn
    print("Computing reference solution (RK54)...")
    ref_states = solve(rk54_step, state0, dt_ref, T)
    time_ref = np.linspace(0, T, int(T/dt_ref) + 1)

    # Chọn phương pháp cần test ở đây:
    methods = {
        "RK4": rk4_step,
        "RK45": rk45_step,
        "RK6": rk6_step
    }

    for name, method in methods.items():
        print(f"\nComputing solution with {name}...")
        test_states = solve(method, state0, dt_test, T)
        time_test = np.linspace(0, T, int(T/dt_test) + 1)

        # Nội suy nghiệm chuẩn tại thời điểm nghiệm test
        interp_ref = interp1d(time_ref, ref_states, axis=0)
        ref_at_test_times = interp_ref(time_test)

        # Tính sai số từng bước
        errors = [compute_error(y_t, y_r) for y_t, y_r in zip(test_states, ref_at_test_times)]

        # Tính chuẩn L2 và L_inf
        error_L2 = np.sqrt(np.mean(np.array(errors) ** 2))
        error_Linf = np.max(errors)

        print(f"{name} L2 error: {error_L2:.4e}")
        print(f"{name} L_inf error: {error_Linf:.4e}")

        # Vẽ biểu đồ sai số theo thời gian
        plt.plot(time_test, errors, label=f'Error {name}')

    plt.xlabel('Time')
    plt.ylabel('Error (Euclidean norm)')
    plt.yscale('log')
    plt.title('Sai số so với nghiệm chuẩn RK54')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
