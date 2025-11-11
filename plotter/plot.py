import matplotlib.pyplot as plt
import numpy as np

def plot_xz_trajectory(traj, N, steps_to_plot=None):
    """
    Plot x-z positions of the string nodes at selected time steps.

    Args:
        traj : array (num_steps, (N+1)*3)
        N    : number of elements (so N+1 nodes)
        steps_to_plot : list of indices of time steps to plot
    """
    num_steps = traj.shape[0]
    num_nodes = N + 1

    if steps_to_plot is None:
        # pick first, middle, and last
        steps_to_plot = [0, num_steps // 2, num_steps - 1]

    plt.figure(figsize=(6, 6))
    for step in steps_to_plot:
        coords = traj[step].reshape(num_nodes, 3)
        x = coords[:, 0]
        z = coords[:, 2]
        plt.plot(x, z, label=f"t =  {int(np.round(step / 10000))} s")
        plt.plot(x[-1], z[-1] , marker = 'o')


    plt.xlabel("x")
    plt.ylabel("z")
    plt.title("String configuration (x–z plane)")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()
