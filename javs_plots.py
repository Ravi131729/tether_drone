import numpy as np
import matplotlib.pyplot as plt
import os
import re

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 30,            # base font size
    "axes.titlesize": 16,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 30
})
# =========================
# Select one npz file here
# =========================
fname = "results/sim_rank90_omega0.900.npz"   # <-- change this if needed
# fname = "vel2_circle.npz"   # <-- change this if needed
filepath = os.path.join("", fname)

# =========================
# Simulation parameters
# =========================
L = 10.0
N = 20
EA = 1e5
l_k = L / N
h = 1e-4
ds = 1000          # downsample factor
window = 1.0       # integration window [s]
window_steps = int(window / h)
window_ds = max(1, window_steps // ds)

# =========================
# Helper functions
# =========================
def compute_drone_tension(q):
    """
    Drone-side tension magnitude using last segment.
    q: (num_nodes, 3) positions for one time step
    """
    diff = q[-1] - q[-2]
    length = np.linalg.norm(diff)
    if length == 0.0:
        return 0.0
    stretch = (length - l_k) / length
    T_vec = (EA / l_k) * stretch * diff
    return float(np.linalg.norm(T_vec))

def moving_average(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x.copy()
    return np.convolve(x, np.ones(w) / w, mode="valid")

def extract_omega(fname):
    m = re.search(r"omega(-?\d+(?:\.\d+)?)", fname)
    return float(m.group(1)) if m else None

# # =========================
# # Load data
# # =========================
data = np.load(filepath)
traj_nodes = data["trajectories"]   # shape: (T, flat_dim) or similar
traj_nodes = traj_nodes[:, 1:]      # keep your original slicing

num_steps = traj_nodes.shape[0]
num_nodes = traj_nodes.shape[1] // 3
traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)

# =========================
# Time vector
# =========================
time = np.arange(num_steps) * h
time_ds = time[::ds]

# =========================
# Radial position + elevation angle
# =========================
tip_pos = traj_nodes[:, -1, :]
base_pos = traj_nodes[:, 0, :]

tip_mag = np.linalg.norm(tip_pos - base_pos, axis=1)
tip_angle = np.degrees(np.arctan2(tip_pos[:, 2], tip_pos[:, 0]))

# =========================
# Drone tension
# =========================
tensions_ds = np.array(
    [compute_drone_tension(traj_nodes[j]) for j in range(0, num_steps, ds)],
    dtype=float
)

smoothed_tension = moving_average(tensions_ds, w=50)
time_smoothed = time_ds[:len(smoothed_tension)]

# =========================
# Plot 3x1 subplot
# =========================
omega_val = extract_omega(fname)
title_main = (
    f"Drone Response (ω = {omega_val:.3f})"
    if omega_val is not None
    else "Drone Response"
)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 1) Radial position magnitude
axes[0].plot(time[::ds], tip_mag[::ds], label="radial distance")
# axes[0].axhline(tip_mag[0], color="gray", linestyle="--", linewidth=1, label="Initial")
axes[0].set_ylabel("Magnitude (m)")
# axes[0].set_title("Drone Position from Base")
# axes[0].legend()
# axes[0].set_ylim(9.9, 10.1)
axes[0].grid(True, alpha=0.2)

# 2) Elevation angle
axes[1].plot(time[::ds], tip_angle[::ds], label="elevation angle")
# axes[1].axhline(tip_angle[0], color="gray", linestyle="--", linewidth=1, label="Initial")
axes[1].set_ylabel("Angle (deg)")
# axes[1].set_title("Elevation Angle")
# axes[1].legend()
axes[1].grid(True, alpha=0.2)

# 3) Drone tension
# axes[2].plot(time_ds, tensions_ds, lw=0.8, alpha=0.6, label="Raw")
axes[2].plot(time_smoothed, smoothed_tension, "r-", lw=2, label="Smoothed")
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Drone Tension (N)")
# axes[2].set_title("Drone Tension")
# axes[2].legend()
axes[2].grid(True, alpha=0.2)

# fig.suptitle(title_main, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# =========================
# Save and show
# =========================
out_name = (
    f"combined_response_{omega_val:.3f}.pdf"
    if omega_val is not None
    else f"combined_response_{os.path.splitext(fname)[0]}.pdf"
)

output_folder = os.path.join("results", "combined_plots")
os.makedirs(output_folder, exist_ok=True)
fig.savefig(os.path.join(output_folder, out_name), dpi=300)

# plt.show()

print(f"Saved to: {os.path.join(output_folder, out_name)}")
print(f"Max position magnitude = {tip_mag.max():.4f} m")
print(f"Min position magnitude = {tip_mag.min():.4f} m")
print(f"Max elevation angle = {tip_angle.max():.2f} deg")
print(f"Min elevation angle = {tip_angle.min():.2f} deg")
print(f"Max tension = {tensions_ds.max():.4f}")
print(f"Min tension = {tensions_ds.min():.4f}")
snapshot_times = [0.0, 5.0,10.0,15.0,20.0]
snapshot_indices = [min(int(t / h), num_steps - 1) for t in snapshot_times]
# ---------------------------------
# Plot overlay (single figure)
# ---------------------------------
plt.figure(figsize=(3.5, 4.5))  # width, height in inches

colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

for i, (idx, t_snap) in enumerate(zip(snapshot_indices, snapshot_times)):
    q = traj_nodes[idx]

    plt.plot(q[:, 0], q[:, 2],
             '-',
             color='red',
             lw=2,
             ms=3,
             label=f"t = {t_snap:.2f} s")

    # Mark final tip position
    final_q = traj_nodes[idx][-1]
    plt.scatter(final_q[0], final_q[2],
                color='green', s=80)

# Mark base (same for all)
base = traj_nodes[0][0]
plt.scatter(base[0], base[2], color='black', s=80, label="Base")

# Mark final tip position (last snapshot)
final_q = traj_nodes[snapshot_indices[-1]]
plt.scatter(final_q[-1, 0], final_q[-1, 2],
            color='green', s=80, label="Final Tip")

# Formatting
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
# plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
# plt.legend()

# Save

overlay_path = os.path.join("results", "snapshot_ree.pdf")
# plt.savefig(, dpi=300)
plt.savefig(overlay_path, bbox_inches='tight',dpi=300)
plt.show()

print(f"Saved overlay plot to: {overlay_path}")

# # =========================
# # Base and drone trajectories
# # =========================
# base_traj = traj_nodes[:, 0, :]      # base node: (T, 3)
# drone_traj = traj_nodes[:, -1, :]    # drone tip/node: (T, 3)

# base_xy = base_traj[:, :2]
# drone_xy = drone_traj[:, :2]
# drone_z = drone_traj[:, 2]

# # ---------------------------------
# # Combined trajectory plots
# # ---------------------------------
# fig2, axes2 = plt.subplots(1, 2, figsize=(15, 4.5))

# # 1) Base XY trace
# axes2[0].plot(base_xy[:, 0], base_xy[:, 1], lw=2, label="Base XY")
# axes2[0].plot(drone_xy[:, 0], drone_xy[:, 1], lw=2, label="Drone XY")

# axes2[0].scatter(base_xy[0, 0], base_xy[0, 1], s=60, label="Start")
# axes2[0].scatter(base_xy[-1, 0], base_xy[-1, 1], s=60, label="End")
# axes2[0].scatter(drone_xy[0, 0], drone_xy[0, 1], s=60, label="Drone Start")
# axes2[0].scatter(drone_xy[-1, 0], drone_xy[-1, 1], s=60, label="Drone End")
# axes2[0].set_xlabel("X [m]")
# axes2[0].set_ylabel("Y [m]")
# # axes2[0].set_title("Base XY Trajectory")
# axes2[0].grid(True)
# # axes2[0].legend()



# # 3) Drone Z vs time
# axes2[1].plot(time, drone_z, lw=2, label="Drone Z")
# axes2[1].set_xlabel("Time [s]")
# axes2[1].set_ylabel("Z [m]")
# # axes2[1].set_title("Drone Altitude")
# axes2[1].grid(True)
# # axes2[1].legend()

# fig2.tight_layout()

# traj_out = os.path.join("results", "combined_plots", "xy_z_tracesvel2.pdf")
# fig2.savefig(traj_out, dpi=300, bbox_inches="tight")
# plt.show()

# print(f"Saved trajectory plot to: {traj_out}")