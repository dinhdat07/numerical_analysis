# main.py

from rocket_config import STAGE1
from rocket_simulator import simulate
from rocket_plot import plot_trajectory, plot_energy

def main():
    t_d, y_d, v_d, a_d, e_d = simulate(STAGE1, drag=True)
    t_n, y_n, v_n, a_n, e_n = simulate(STAGE1, drag=False)

    plot_trajectory(t_d, y_d, t_n, y_n, v_d, v_n, a_d, a_n)
    plot_energy(t_d, e_d, t_n, e_n)

if __name__ == "__main__":
    main()
