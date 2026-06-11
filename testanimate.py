from plotter.animate import animate_trajc,snapshot_multiple_times,animate_traj
import numpy as np
import matplotlib.pyplot as plt
# === Test file ===
test_file = "test.npz"

# === Simulation parameters ===



h = 1e-4




# # === Load test data ===
data = np.load(test_file)
traj_nodes = data["trajectories"]  # (T, N, 3)
traj_R = data["rot_mat"]  # (T, 3, 3)
print(traj_nodes.shape)
num_steps, num_nodes = traj_nodes.shape

time = np.arange(num_steps) * h
spk_val = traj_nodes[:,0]
spk_dot = -(spk_val[1:] - spk_val[:-1])/h
traj_nodes = traj_nodes[:,1:]

num_steps = traj_nodes.shape[0]
num_nodes = traj_nodes.shape[1] // 3  # 6 // 3 = 2

traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)
# ---- Plot ----
# fig, ax = plt.subplots(figsize=(8, 4.5))

# ax.plot(time[:-1], spk_dot, color="tab:purple", lw=1.5, label="Unstretched Length")

# # ---- Styling ----
# ax.set_xlabel("Time [s]", fontsize=12)
# ax.set_ylabel("L_dot", fontsize=12)
# # ax.set_title("Cable Unstretched Length Over Time", fontsize=14)

# ax.grid(True, which="both", linestyle="--", alpha=0.7)
# # ax.legend(loc="best", fontsize=11)
# fig.tight_layout()

# plt.show()
animate_trajc(np.array(traj_nodes), duration_sec=20, fps= 30, stl_file="models/Assembly.STL")
# animate_traj(np.array(traj_nodes),traj_R, duration_sec=30, fps= 10, stl_file="models/Assembly.STL")
# time_indices = [0, 20, 50, 100, 150]

# snapshot_multiple_times(np.array(traj_nodes), time_indices)
# snapshot_multiple_times(np.array(traj_nodes), frame_indices=[0, 10, 20], stl_file="models/Assembly.STL", out_names=["tether_snapshot_0.png", "tether_snapshot_10.png", "tether_snapshot_20.png"])