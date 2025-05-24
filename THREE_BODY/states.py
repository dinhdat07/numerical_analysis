import numpy as np

def generate_random_state(dim="2D"):
    rng = np.random.default_rng()
    dim_val = 2 if dim == "2D" else 3
    positions = rng.uniform(-1, 1, size=(3, dim_val))
    velocities = rng.uniform(-0.5, 0.5, size=(3, dim_val))
    velocities -= velocities.mean(axis=0)
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

def generate_random_state_sym(dim="2D"):
    rng = np.random.default_rng()
    dim_val = 2 if dim == "2D" else 3
    a = rng.uniform(0.5, 1.5)
    # 3 bodies positions - 2D or 3D (z=0 if 2D)
    if dim == "2D":
        positions = np.array([
            [-a, 0], [a, 0], [0, 0]
        ])
        velocities = np.array([
            [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)],
            [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)],
            [rng.uniform(-1.6, 1.6), rng.uniform(-1.6, 1.6)]
        ])
    else:  # 3D
        positions = np.array([
            [-a, 0, 0], [a, 0, 0], [0, 0, 0]
        ])
        velocities = np.array([
            [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)],
            [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8)],
            [-2*rng.uniform(-0.8, 0.8), -2*rng.uniform(-0.8, 0.8), -2*rng.uniform(-0.8, 0.8)]
        ])
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

named_states = {
    "8-Figure": np.array([
         0.97000436, -0.24308753, -0.46620368, -0.43236573,  
        -0.97000436,  0.24308753, -0.46620368, -0.43236573,  
         0.0,          0.0,         0.93240737,  0.86473146
    ]),
    "Lagrange": np.array([
        1.0, 0.0, 0.0, 0.5,
        -0.5, np.sqrt(3)/2, -0.5*np.sqrt(3), -0.25,
        -0.5, -np.sqrt(3)/2, 0.5*np.sqrt(3), -0.25
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
    "3D": np.array([
        -0.85504536,  0.50204268,  0.00267094, -0.94863563,  0.7238842,   0.82469848,
        0.47365597, -0.62878416,  0.64240703, -0.36761325, -0.13233019,  0.6942114,
        -0.5652762,   0.62295359, -0.51591877,  0.5598127,  -0.69730335, -0.35694745
    ]),
}

# Example usage
# Initial state: [-1.44332343  0.         -0.24852747 -0.36869087  1.44332343  0.
#  -0.24852747 -0.36869087  0.          0.          0.49705494  0.73738173]
# Initial state: [-1.14591565  0.         -0.22081826 -0.15568209  1.14591565  0.
#  -0.22081826 -0.15568209  0.          0.          0.44163652  0.31136418]