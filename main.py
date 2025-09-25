from simulation.integrator import run_simulation
from simulation.utils import make_initial_configuration
import jax.numpy as jnp
import numpy as np
from plotter.plot import plot_xz_trajectory
from plotter.animate import animate_traj

# ----------------------------
# Problem setup
# ----------------------------
L = 20.0
N = 50
h = 1e-4
tf =20.0
steps = int(tf / h)

total_weight = 9.81 * (0.06 * L + 2.0)

params = dict(
    mu    = jnp.array(0.06),
    M     = jnp.array(2.0),
    h     = jnp.array(h),
    g     = jnp.array(-9.81),
    L     = jnp.array(L),
    N     = N,
    l_k   = L / N,
    EA    = jnp.array(1.0e5),
    gkv   = make_initial_configuration(L, N),
    g_km1v = make_initial_configuration(L, N),
    X_km1 = jnp.zeros((N+1, 3)).reshape(-1),
    force = jnp.array([0.0, 0.0, 1.1 * total_weight]),
    omega = jnp.array(3 * 2 * jnp.pi),
    delta_base_pos = jnp.array([0.0, 0.0, 0.0]),
    step = 0,
)

# ----------------------------
# Run
# ----------------------------
import time

start = time.perf_counter()
traj = run_simulation(params, num_steps=steps)
traj.block_until_ready()
end = time.perf_counter()

print(f"Simulation took {end - start:.3f} seconds")
print("Trajectory shape:", traj.shape)


print("Trajectory shape:", traj.shape)

plot_xz_trajectory(traj, N=params["N"])
animate_traj(np.array(traj), duration_sec=tf, fps=60, stl_file="models/Assembly.STL")