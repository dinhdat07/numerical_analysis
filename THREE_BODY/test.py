import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from utils import rk4_step, rk54_step, rk45_step, rk6_step

# Euclidean error
def compute_error(y_test, y_ref):
    return np.linalg.norm(y_test - y_ref)

def init_state():
    return np.array([
         0.97000436, -0.24308753, -0.46620368, -0.43236573,  
        -0.97000436,  0.24308753, -0.46620368, -0.43236573,  
         0.0,          0.0,         0.93240737,  0.86473146
    ])

# Solve using fixed step size
def solve_fixed(method, state0, dt, T):
    steps = int(T / dt)
    states = [state0.copy()]
    state = state0.copy()
    for _ in range(steps):
        result = method(state, dt)
        state = result[0] if isinstance(result, tuple) else result
        states.append(state.copy())
    times = np.linspace(0, T, steps + 1)
    
    return np.array(times), np.array(states)


# Solve using adaptive step size
def solve_adaptive(rk_step, state0, T, dt_init=0.001, tol=1e-3):
    t = 0
    dt = dt_init
    state = state0.copy()
    times = [0]
    states = [state.copy()]

    while t < T:
        if t + dt > T:
            dt = T - t

        y_high, y_low = rk_step(state, dt)
        error = np.linalg.norm(y_high - y_low)

        if error < tol:
            t += dt
            state = y_high
            times.append(t)
            states.append(state.copy())

        s = 2 if error == 0 else 0.9 * (tol / error) ** (1/5)
        dt *= min(max(0.1, s), 5)

    return np.array(times), np.array(states)

if __name__ == "__main__":
    T = 100
    dt_ref = 0.001
    state0 = init_state()

    print("Computing reference solution with RK54 (fine step)...")
    time_ref, ref_states = solve_adaptive(rk54_step, state0, T, dt_init=dt_ref, tol=1e-6)
    solve_fixed(rk54_step, state0, dt_ref, T)
    methods = {
        "RK4": (rk4_step, False),
        "RK45": (rk45_step, True),
        "RK6": (rk6_step, False)
    }

    dt_test = 0.01

    for name, (method, is_adaptive) in methods.items():
        print(f"\nComputing solution with {name}...")
        if is_adaptive:
            time_test, test_states = solve_adaptive(method, state0, T, dt_init=dt_test, tol=1e-4)
        else:
            time_test, test_states = solve_fixed(method, state0, dt_test, T)

        # Interpolate reference at test time points
        interp_ref = interp1d(time_ref, ref_states, axis=0, bounds_error=False, fill_value="extrapolate")
        ref_at_test_times = interp_ref(time_test)

        # Compute errors
        errors = [compute_error(y_t, y_r) for y_t, y_r in zip(test_states, ref_at_test_times)]
        error_L2 = np.sqrt(np.mean(np.array(errors) ** 2))
        error_Linf = np.max(errors)

        print(f"{name} L2 error: {error_L2:.4e}")
        print(f"{name} L_inf error: {error_Linf:.4e}")

        plt.plot(time_test, errors, label=f'{name} (dt={dt_test})')

    plt.xlabel('Time')
    plt.ylabel('Error (Euclidean norm)')
    plt.yscale('log')
    plt.title('Sai số so với nghiệm chuẩn RK54')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
