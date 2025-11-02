import numpy as np
import matplotlib.pyplot as plt

# === Load saved results ===
data = np.load("results/sim_rank0_omega0.000_tf500.0s_L20.0_N50.npz")
traj_nodes = data["trajectories"]   # shape (num_steps, flat_dim)

# === Reshape ===
print(traj_nodes.shape)
# num_steps, flat_dim  = traj_nodes.shape
# num_nodes = flat_dim // 3
# traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)
num_steps, num_nodes ,flat_dim  = traj_nodes.shape
h = 0.0001

# === Time vector ===
time = np.arange(num_steps) * float(h)
dt = time[1] - time[0]
ds = 1000   # downsample factor for time plots

# === Extract tip position, magnitude, and angle ===
tip_pos = traj_nodes[:, -1, :]  # last node = tip
base_pos= traj_nodes[:, 0, :]

tip_mag = np.linalg.norm(tip_pos - base_pos, axis=1)
tip_angle = np.degrees(np.arctan2(tip_pos[:, 2], tip_pos[:, 0]))

# === Plot Tip Magnitude vs Time ===
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(time[::ds], tip_mag[::ds], label="Tip magnitude")
ax1.axhline(tip_mag[0], color="gray", linestyle="--", linewidth=1, label="Initial Value")
ax1.set_title("Drone Position Magnitude vs Time")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Magnitude [m]")
ax1.legend()
ax1.grid(True)
fig1.tight_layout()

# === Plot Tip Angle vs Time ===
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(time[::ds], tip_angle[::ds], label="Tip angle")
ax2.axhline(tip_angle[0], color="gray", linestyle="--", linewidth=1, label="Initial Value")
ax2.set_title("Elevation Angle vs Time")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Angle [deg]")
ax2.legend()
ax2.grid(True)
fig2.tight_layout()

# === FFT of Angle ===
N = len(tip_angle)
freqs = np.fft.rfftfreq(N, d=dt)
fft_vals = np.fft.rfft(tip_angle - np.mean(tip_angle))
mag = np.abs(fft_vals)
if mag.max() > 0:
    mag /= mag.max()

fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(freqs, mag, "b-")
ax3.set_xlim(0, 5)
ax3.set_title("Normalized FFT Spectrum of Angle")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_ylabel("Normalized Amplitude")
ax3.grid(True)
fig3.tight_layout()

# === Summary Stats ===
print(f"Max position magnitude = {tip_mag.max():.4f} m")
print(f"Min position magnitude = {tip_mag.min():.4f} m")
print(f"Max elevation angle = {tip_angle.max():.2f} deg")
print(f"Min elevation angle = {tip_angle.min():.2f} deg")

# === Save ===
fig1.savefig("drone_position.svg", format="svg")
fig2.savefig("drone_angle.svg", format="svg")
fig3.savefig("fft_angle.svg", format="svg")

plt.show()
