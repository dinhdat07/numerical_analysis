import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

from utils import rk4_step, rk54_step, rk45_step, rk6_step

def compute_error(y_test, y_ref):
    return np.linalg.norm(y_test - y_ref)

def init_state():
    return np.array([
        -1.0, 0.0, 0.080584, 0.588836,
         1.0, 0.0, 0.080584, 0.588836,
         0.0, 0.0, -0.161168, -1.177672
    ])

masses = np.array([1.0, 1.0, 1.0])

def compute_energy(states, masses):
    n = len(masses)
    E = []
    for state in states:
        kinetic = 0
        potential = 0
        for i in range(n):
            v = state[6 + 2*i:6 + 2*(i+1)]
            kinetic += 0.5 * masses[i] * np.dot(v, v)
        for i in range(n):
            for j in range(i+1, n):
                ri = state[2*i:2*(i+1)]
                rj = state[2*j:2*(j+1)]
                potential -= masses[i] * masses[j] / np.linalg.norm(ri - rj)
        E.append(kinetic + potential)
    return np.array(E)

def compute_momentum(states, masses):
    Px, Py = [], []
    for state in states:
        px = np.sum([masses[i] * state[6 + 2*i] for i in range(3)])
        py = np.sum([masses[i] * state[6 + 2*i + 1] for i in range(3)])
        Px.append(px)
        Py.append(py)
    return np.array(Px), np.array(Py)

def compute_angular_momentum(states, masses):
    Lz = []
    for state in states:
        total_L = 0
        for i in range(3):
            x, y = state[2*i:2*(i+1)]
            vx, vy = state[6 + 2*i:6 + 2*(i+1)]
            total_L += masses[i] * (x * vy - y * vx)
        Lz.append(total_L)
    return np.array(Lz)

def plot_physical_errors(times, E, P, L, label=None):
    E0, P0x, P0y, L0 = E[0], P[0][0], P[1][0], L[0]

    rel_E = np.abs((E - E0) / E0)
    rel_P = np.sqrt((P[0] - P0x)**2 + (P[1] - P0y)**2) / np.sqrt(P0x**2 + P0y**2 + 1e-20)
    rel_L = np.abs((L - L0) / (L0 + 1e-20))

    plt.figure(figsize=(10, 6))
    plt.plot(times, rel_E, label=f"Energy {label}")
    plt.plot(times, rel_P, label=f"Momentum {label}")
    plt.plot(times, rel_L, label=f"Angular Momentum {label}")
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Relative Error')
    plt.title('Conservation Quantity Errors')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'L2_Energy': np.sqrt(np.mean(rel_E**2)),
        'Linf_Energy': np.max(rel_E),
        'L2_Momentum': np.sqrt(np.mean(rel_P**2)),
        'Linf_Momentum': np.max(rel_P),
        'L2_AngularMomentum': np.sqrt(np.mean(rel_L**2)),
        'Linf_AngularMomentum': np.max(rel_L),
    }

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

def solve_adaptive(rk_step, state0, T, dt_init=0.0001, tol=1e-8, dt_min=1e-8, dt_max=0.01, record_dt=0.0001):
    t = 0
    dt = dt_init
    state = state0.copy()

    states = []
    times = []

    record_times = np.arange(0, T + record_dt/2, record_dt)
    record_index = 0

    while t < T:
        dt = min(dt, T - t)

        prev_state = state.copy()
        y_high, y_low = rk_step(state, dt)
        R = np.linalg.norm(y_high - y_low) / dt

        if R <= tol or dt <= dt_min:
            t += dt
            state[:] = y_high

            while record_index < len(record_times) and record_times[record_index] <= t:
                interp_time = record_times[record_index]
                interp = (interp_time - (t - dt)) / dt
                y_interp = (1 - interp) * prev_state + interp * y_high
                times.append(interp_time)
                states.append(y_interp.copy())
                record_index += 1

        delta = 0.84 * (tol / R) ** 0.25 if R > 0 else 4
        dt *= min(max(delta, 0.1), 4)
        dt = max(dt_min, min(dt, dt_max))

        if dt < dt_min and R > tol:
            print("Minimum dt exceeded, stopping.")
            break

    return np.array(times), np.array(states)

def compute_physical_errors(time, E, P, L, normalize='step'):
    n = len(E)
    T = time[-1] - time[0]
    num_steps = n - 1 if n > 1 else 1

    def l2(x):
        return np.linalg.norm(x - x[0]) / n

    def linf(x):
        return np.max(np.abs(x - x[0]))

    def rel_l2(x):
        return np.linalg.norm((x - x[0]) / x[0]) / n

    def rel_linf(x):
        return np.max(np.abs((x - x[0]) / x[0]))

    # Sai số tuyệt đối
    err = {
        'L2_Energy': l2(E),
        'Linf_Energy': linf(E),
        'L2_Momentum': l2(P[0]) + l2(P[1]),
        'Linf_Momentum': linf(P[0]) + linf(P[1]),
        'L2_AngularMomentum': l2(L),
        'Linf_AngularMomentum': linf(L),
    }

    if normalize == 'time':
        for k in err: err[k] /= T
    elif normalize == 'step':
        for k in err: err[k] /= num_steps
    elif normalize == 'relative':
        err = {
            'L2_Energy': rel_l2(E),
            'Linf_Energy': rel_linf(E),
            'L2_Momentum': rel_l2(P[0]) + rel_l2(P[1]),
            'Linf_Momentum': rel_linf(P[0]) + rel_linf(P[1]),
            'L2_AngularMomentum': rel_l2(L),
            'Linf_AngularMomentum': rel_linf(L),
        }

    return err

def plot_physical_quantities(time, E, P, L, label=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(time, E, label=label)
    axs[0].set_ylabel("Energy")
    axs[0].grid(True)

    axs[1].plot(time, P[0], label='Px' if label is None else f'{label} - Px')
    axs[1].plot(time, P[1], label='Py' if label is None else f'{label} - Py')
    axs[1].set_ylabel("Linear Momentum")
    axs[1].grid(True)

    axs[2].plot(time, L, label=label)
    axs[2].set_ylabel("Angular Momentum")
    axs[2].set_xlabel("Time")
    axs[2].grid(True)

    if label is not None:
        for ax in axs:
            ax.legend()

    plt.tight_layout()
    plt.show()


import time

if __name__ == "__main__":
    T = 100
    dt_test = 0.001

    initial_states = {
        "8-Figure": np.array([ 
            0.97000436, -0.24308753, -0.46620368, -0.43236573,
        -0.97000436,  0.24308753, -0.46620368, -0.43236573,
            0.0,          0.0,         0.93240737,  0.86473146
        ]),
        "Lagrange": np.array([
            1.0, 0.0,               0.0,        0.5,
        -0.5,  np.sqrt(3)/2, -np.sqrt(3)/2 * 0.5, -0.5 * 0.5,
        -0.5, -np.sqrt(3)/2,  np.sqrt(3)/2 * 0.5, -0.5 * 0.5
        ]),
        "Butterfly": np.array([
            -1.0, 0.0, 0.306893, 0.125507,
            1.0, 0.0, 0.306893, 0.125507,
            0.0, 0.0, -0.613786, -0.251014
        ]),
        "Yin-Yang": np.array([
            -1.0, 0.0, 0.513938, 0.304736,
            1.0, 0.0, 0.513938, 0.304736,
            0.0, 0.0, -1.027876, -0.609472
        ]),
        "Dragonfly": np.array([
            -1.0, 0.0, 0.080584, 0.588836,
            1.0, 0.0, 0.080584, 0.588836,
            0.0, 0.0, -0.161168, -1.177672
        ]),
        "Custom1": np.array([
            -1.44332343, 0.0, -0.24852747, -0.36869087,
            1.44332343, 0.0, -0.24852747, -0.36869087,
            0.0, 0.0, 0.49705494, 0.73738173
        ]),
        "Custom2": np.array([
            -1.14591565, 0.0, -0.22081826, -0.15568209,
            1.14591565, 0.0, -0.22081826, -0.15568209,
            0.0, 0.0, 0.44163652, 0.31136418
        ]),
    }

    methods = {
        "RK4": (rk4_step, False),
        "RK45": (rk45_step, True),
        "RK6": (rk6_step, False),
        "RK54": (rk54_step, True),
    }

    for name, (method, is_adaptive) in methods.items():
        print(f"\nEvaluating {name} across all states...")
        total_errors = {
            'L2_Energy': 0, 'Linf_Energy': 0,
            'L2_Momentum': 0, 'Linf_Momentum': 0,
            'L2_AngularMomentum': 0, 'Linf_AngularMomentum': 0,
        }
        total_time = 0.0

        for state_name, state0 in initial_states.items():
            start = time.perf_counter()

            if is_adaptive:
                time_test, test_states = solve_adaptive(
                    method, state0.copy(), T,
                    dt_init=dt_test, tol=1e-7,
                    dt_min=1e-7, dt_max=0.001, record_dt=dt_test
                )
            else:
                time_test, test_states = solve_fixed(method, state0.copy(), dt_test, T)

            elapsed = time.perf_counter() - start
            total_time += elapsed

            E = compute_energy(test_states, masses)
            P = compute_momentum(test_states, masses)
            L = compute_angular_momentum(test_states, masses)

            errors = compute_physical_errors(time_test, E, P, L, normalize='step')
            for k in total_errors:
                total_errors[k] += errors[k]

        # Tính trung bình sai số và thời gian
        n = len(initial_states)
        for k in total_errors:
            total_errors[k] /= n
        avg_time = total_time / n

        print(f"{name} - Mean Energy L2/step: {total_errors['L2_Energy']:.2e}, L_inf: {total_errors['Linf_Energy']:.2e}")
        print(f"{name} - Mean Momentum L2/step: {total_errors['L2_Momentum']:.2e}, L_inf: {total_errors['Linf_Momentum']:.2e}")
        print(f"{name} - Mean Angular Momentum L2/step: {total_errors['L2_AngularMomentum']:.2e}, L_inf: {total_errors['Linf_AngularMomentum']:.2e}")
        print(f"{name} - Avg. runtime per state: {avg_time:.3f} seconds")
