from gui import choose_state
from simulation import run_simulation
from simulation_3D import run_simulation_3D


if __name__ == "__main__":
    while True:
        state,dim, method = choose_state()
        print("State:", state)

        if state is None:
            break
        if dim == "2D":
            next_action = run_simulation(state,method=method)
        else: 
            next_action = run_simulation_3D(state)

        if next_action == "exit":
            break
