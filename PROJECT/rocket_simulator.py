# rocket_simulator.py

import numpy as np
from rocket_config import G, M, R, rho0, H, Cd, A

def air_density(y):
    return rho0 * np.exp(-y / H)

def acceleration(y, v, t, m, stage, drag=True):
    thrust = stage.T if t <= stage.burn_time else 0
    F_drag = 0.5 * air_density(y) * Cd * A * v**2 * np.sign(v) if drag else 0
    gravity = G * M / (R + y)**2
    return (thrust - F_drag) / m - gravity

def rk4_step(y, v, t, m, dt, stage, drag=True):
    def f(y, v): return v
    def g(y, v): return acceleration(y, v, t, m, stage, drag)

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

def total_energy(y, v, m):
    kinetic = 0.5 * m * v**2
    potential = -G * M * m / (R + y)
    return kinetic + potential

def simulate(stage, t_max=1600, dt=0.1, drag=True):
    steps = int(t_max / dt)
    y, v, m = 0.0, 0.0, stage.m0
    y_prev = y
    energy_loss = 0

    ys, vs, accs, energies, time = [], [], [], [], []

    for i in range(steps):
        t = i * dt
        ys.append(y)
        vs.append(v)
        accs.append(acceleration(y, v, t, m, stage, drag))
        total_mech = total_energy(y, v, m)

        if drag:
            F_drag = 0.5 * air_density(y) * Cd * A * v**2 * np.sign(v)
            dy = y - y_prev
            energy_loss += F_drag * dy
            total_mech -= energy_loss
            y_prev = y

        energies.append(total_mech)
        time.append(t)

        y, v = rk4_step(y, v, t, m, dt, stage, drag)
        m = m - stage.mdot * dt if t < stage.burn_time else max(m, stage.m_dry)

        if y < 0:
            break

    return np.array(time), np.array(ys), np.array(vs), np.array(accs), np.array(energies)
