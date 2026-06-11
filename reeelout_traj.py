import numpy as np
import matplotlib.pyplot as plt

files = {
    "1 link": "reelout_1.npz",
    "20 links": "reelout_20.npz",
}

h = 0.01   # your time step
ds = 1     # downsampling if needed

plt.figure(figsize=(7, 6))

for label, filepath in files.items():
    data = np.load(filepath)
    traj_nodes = data["trajectories"]   # shape: (T, flat_dim)

    traj_nodes = traj_nodes[:, 1:]      # keep your original slicing

    num_steps = traj_nodes.shape[0]
    num_nodes = traj_nodes.shape[1] // 3
    traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)

    # drone position: use the last node as the tip/drone attachment point
    tip_pos = traj_nodes[:, -1, :]

    # plot x-z trajectory
    plt.plot(tip_pos[::ds, 0], tip_pos[::ds, 2], label=label, linewidth=2)

plt.xlabel("x")
plt.ylabel("z")
plt.title("Drone trajectory during reel-out")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()