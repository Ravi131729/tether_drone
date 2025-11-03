from simulation.integrator import run_simulation
from simulation.utils import make_initial_configuration
import jax.numpy as jnp
import jax
import numpy as np
from plotter.plot import plot_xz_trajectory
from plotter.animate import animate_traj
from simulation.catenary import fixed_point_iteration
jax.config.update("jax_platforms", "cpu")
# ----------------------------
# Problem setup
# ----------------------------
L = 200.0
N = 20
h = 1e-4
tf =2


steps = int(tf / h)
w_base = 2
total_weight = 9.81 * (0.06 * L + 2.0)
spk =180
rho = jnp.array([0.15, 0.0, 0.3])
params = dict(
    mu    = jnp.array(0.06),
    M     = jnp.array(2.0),
    h     = jnp.array(h),
    g     = jnp.array(-9.81),
    L     = jnp.array(L),
    N     = N,
    l_k   = L / N,
    EA    = jnp.array(1.0e5),
    gkv   = make_initial_configuration(L-spk, N , rho , spk),
    g_km1v = make_initial_configuration(L-spk, N , rho , spk),
    X_km1 = jnp.zeros_like(make_initial_configuration(L, N , rho , spk)),
    force = jnp.array([10, 0.0,  1.2*total_weight]),
    omega = jnp.array(0.07),
    delta_base_pos = jnp.array([0.0, 0.0, 0.0]),
    step = 0,
    kappa = jnp.array(1),  ##need to change accordingly
    b = jnp.array(0.3),
    d = jnp.array(0.15),
    rho =rho,     #need to multiply with rotation mat of vehicle
    base_pos = jnp.array([0.0, 0.0, 0.0]),
    u_k = jnp.array(0.0)
)
print(params["g_km1v"])
# ----------------------------
# Run
# ----------------------------
import time
# params["gkv"] = fixed_point_iteration(params)
# params["g_km1v"] =params["gkv"]
# ##v_drone -= 1
# params["g_km1v"] = params["gkv"].at[-3].set(params["gkv"][-3] - params["h"])
# params["X_km1"] = params["g_km1v"] - params["gkv"]
# params["omega"] = jnp.array(w_base)
run_simulation_jit = jax.jit(run_simulation, static_argnums=(1,))
t0 = time.time()
_ = run_simulation_jit(params)
_.block_until_ready()
print("First run:", time.time() - t0, "s")
print("starting simulation")
print("simulation running ..........")
start = time.perf_counter()
traj = run_simulation_jit(params, num_steps=steps)
traj.block_until_ready()
end = time.perf_counter()

# filename = f"stsim_test_omega{w_base:.3f}_tf{tf}s_L{L}_N{N}.npz"



# np.savez(filename,
#          trajectories=np.array(traj),
#          params=params)

print(f"Simulation took {end - start:.3f} seconds")
print("Trajectory shape:", traj.shape)


print("Trajectory shape:", traj[-1])


plot_xz_trajectory(traj[:,1:], N=params["N"])
# animate_traj(np.array(traj[:,1:]), duration_sec=tf, fps=60, stl_file="models/Assembly.STL")