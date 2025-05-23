from gui import choose_state
from simulation import run_simulation

if __name__ == "__main__":
    while True:
        state = choose_state()
        if state is None:  
            break
        next_action = run_simulation(state)
        if next_action == "exit":
            break

