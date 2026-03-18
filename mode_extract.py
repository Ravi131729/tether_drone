import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

# ============================================================
# Load data
# ============================================================

# data = np.load("xz_extractmodes.npz")
data = np.load("x_extractmodes.npz")

traj_nodes = data["trajectories"]     # (num_steps, flat_dim)

# remove drone state if present (your code already does this)
traj_nodes = traj_nodes[:,1:]

num_steps = traj_nodes.shape[0]
num_nodes = traj_nodes.shape[1] // 3

traj_nodes = traj_nodes.reshape(num_steps, num_nodes, 3)

print("Trajectory shape:", traj_nodes.shape)

# ============================================================
# Simulation parameters
# ============================================================

h = 0.0001
time = np.arange(num_steps) * h
dt = h

# ============================================================
# Remove rigid base motion (important for cable vibration)
# ============================================================

base_motion = traj_nodes[:,0:1,:]        # base node
traj_rel = traj_nodes - base_motion      # relative positions

# ============================================================
# Remove equilibrium (mean configuration)
# ============================================================

equilibrium = np.mean(traj_rel, axis=0)
equilibrium = traj_rel[0]  # or use initial configuration as equilibrium
traj_fluct = traj_rel - equilibrium
# ============================================================
# Downsample to 100 Hz
# ============================================================

target_fs = 100
original_fs = 1/h

decimation_factor = int(original_fs / target_fs)

traj_fluct = traj_fluct[::decimation_factor]

num_steps = traj_fluct.shape[0]
dt = h * decimation_factor
time = np.arange(num_steps) * dt

print("Downsampled steps:", num_steps)
print("New dt:", dt)
print("New sampling rate:", 1/dt)
# ============================================================
# Flatten trajectory to state matrix
# ============================================================

X = traj_fluct.reshape(num_steps, num_nodes*3)

# transpose for modal analysis
# X = X.T        # shape (states, time)
# use only x displacement
X = traj_fluct[:,:,0]   # x component only

# shape should be (time, nodes)
X = X.T                 # (nodes, time)
print("State matrix shape:", X.shape)

# ============================================================
# Compute POD modes (SVD)
# ============================================================

U, S, Vt = np.linalg.svd(X, full_matrices=False)

# spatial modes
modes = U

# modal coordinates (time signals)
modal_coords = modes.T @ X

print("Modal coordinates shape:", modal_coords.shape)

# ============================================================
# Compute natural frequencies
# ============================================================

N = num_steps
freqs = rfftfreq(N, dt)

num_modes_to_extract = 6
natural_freqs = []

plt.figure(figsize=(10,6))

for i in range(num_modes_to_extract):

    signal = modal_coords[i]

    # FFT
    fft_vals = rfft(signal)

    # normalize amplitude
    spectrum = np.abs(fft_vals) / N
    spectrum[1:] *= 2   # compensate for removed negative frequencies

    idx = np.argmax(spectrum[1:]) + 1

    natural_freq = freqs[idx]
    natural_freqs.append(natural_freq)

    plt.plot(freqs, spectrum, label=f"Mode {i+1}")

plt.title("Normalized Modal Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

print("\nEstimated Natural Frequencies (Hz):")
for i,f in enumerate(natural_freqs):
    print(f"Mode {i+1}: {f:.3f} Hz")

# ============================================================
# Plot modal coordinates
# ============================================================

plt.figure(figsize=(10,6))

for i in range(num_modes_to_extract):
    plt.plot(time, modal_coords[i], label=f"Mode {i+1}")

plt.title("Modal Coordinates")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# ============================================================
# Plot mode shapes
# ============================================================

for i in range(num_modes_to_extract):

    mode_shape = modes[:, i]

    plt.figure()

    nodes = np.arange(num_nodes)

    plt.plot(mode_shape, 0.5*nodes, 'o-')

    plt.title(f"Mode Shape {i+1}")
    plt.xlabel("X")
    plt.ylabel("Z")

    # plt.gca().invert_yaxis()
    plt.grid()
plt.show()