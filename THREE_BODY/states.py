import numpy as np

def generate_random_state():
    rng = np.random.default_rng()
    positions = rng.uniform(-1, 1, size=(3, 2))
    velocities = rng.uniform(-0.5, 0.5, size=(3, 2))
    velocities -= velocities.mean(axis=0)
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

def generate_random_state_sym():
    rng = np.random.default_rng()
    a = rng.uniform(0.5, 1.5)
    positions = np.array([
        [-a, 0], [a, 0], [0, 0]
    ])
    v1 = rng.uniform(-0.8, 0.8)
    v2 = rng.uniform(-0.8, 0.8)
    velocities = np.array([
        [v1, v2], [v1, v2], [-2*v1, -2*v2]
    ])
    state = np.hstack([np.hstack([positions[i], velocities[i]]) for i in range(3)])
    return state

named_states = {
    "8-Figure": np.array([
         0.97000436, -0.24308753, -0.46620368, -0.43236573,  
        -0.97000436,  0.24308753, -0.46620368, -0.43236573,  
         0.0,          0.0,         0.93240737,  0.86473146
    ]),

    # r=1.0, v=0.5
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
    ])
}

# Example usage
# Initial state: [-1.44332343  0.         -0.24852747 -0.36869087  1.44332343  0.
#  -0.24852747 -0.36869087  0.          0.          0.49705494  0.73738173]
# Initial state: [-1.14591565  0.         -0.22081826 -0.15568209  1.14591565  0.
#  -0.22081826 -0.15568209  0.          0.          0.44163652  0.31136418]