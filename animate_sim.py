from plotter.animate import animate_traj
import numpy as np

data = np.load("results_5x_1.1Wz/sim_rank6_omega1.700_tf500.0s_L20.0_N50.npz")
traj_nodes = data["trajectories"]   # shape (num_steps, flat_dim)

# === Reshape ===
# num_steps, flat_dim = traj_nodes.shape
# num_nodes = flat_dim // 3
# traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)
h = 0.0001

animate_traj(traj_nodes, duration_sec=500, fps=60, stl_file="models/Assembly.STL")