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
L = 15
N = 20
h = 1e-4
tf =200


steps = int(tf / h)
w_base = 1
I = jnp.array([[0.105,0,0],  #drone moment of inertia
              [0,0.105,0],
              [0,0,0.140]])
mu = 0.03
M= 2.5                       #drone mass
spk =5
rho = 0*jnp.array([0.15, 0.0, 0.3])
total_weight = 9.81 * (mu * (L-spk) + M)
uk=0.0                                   #winch torque
F = 1.1*total_weight                        # thrust force
R = jnp.eye(3)
tau = [ 0.0,0.0,0.0]                        #torque on drone
omega = 0.0
params = dict(
    mu    = jnp.array(mu),
    M     = jnp.array(M),
    h     = jnp.array(h),
    g     = jnp.array(-9.81),
    L     = jnp.array(L),
    N     = N,
    l_k   = L-spk / N,
    EA    = jnp.array(1.0e5),
    gkv   = make_initial_configuration(L-spk, N , rho , spk),
    g_km1v = make_initial_configuration(L-spk, N , rho , spk),
    X_km1 = jnp.zeros_like(make_initial_configuration(L, N , rho , spk)),
    force = F,
    omega = jnp.array(omega),
    delta_base_pos = jnp.array([0.0, 0.0, 0.0]),
    step = 0,
    kappa = jnp.array(1),  ##need to change accordingly
    b = jnp.array(0.3),
    d = jnp.array(0.15),
    rho =rho,     #need to multiply with rotation mat of vehicle
    base_pos = jnp.array([0.0, 0.0, 0.0]),
    u_k = jnp.array(uk),
    a = jnp.arange(1.0, N+1, dtype=jnp.float32),#dummy vcariable for use inside for M terms
    J = I,
    f_km1 = jnp.array([0.0,0.0,0.0]),
    fk = jnp.array([0.0,0.0,0.0]),
    tau = jnp.array(tau),
    R_v = jnp.eye(3),
    R = R
)
print(params["g_km1v"])
# ----------------------------
# Run
# ----------------------------
import time
# =-
params["g_km1v"] =params["gkv"]
# ##v_drone -= 1
params["g_km1v"] = params["gkv"].at[-3].set(params["gkv"][-3] - 1*params["h"])
params["X_km1"] = params["g_km1v"] - params["gkv"]
# params["omega"] = jnp.array(w_base)
# key = jax.random.PRNGKey(0)

# perturb = 1* jax.random.normal(key, params["gkv"].shape)
# perturb = jax.random.normal(key, params["gkv"].shape)
# perturb = perturb.at[0].set(0.0)

# perturb = perturb.at[2::3].set(0.0)  # remove y
# perturb = perturb.at[3::3].set(0.0)  # remove z
# params["g_km1v"] = params["gkv"] + perturb * params["h"]
# params["X_km1"] = params["g_km1v"] - params["gkv"]

# # optionally keep s fixed
# params["X_km1"] = params["X_km1"].at[0].set(0.0)


# # params["g_km1v"] = params["gkv"] + perturb*params["h"]

# # params["X_km1"] = params["g_km1v"] - params["gkv"]
# # params["X_km1"] = params["X_km1"].at[0].set(0.0)
# # params["X_km1"] = params["X_km1"].at[1].set(0.0)
# # params["X_km1"] = params["X_km1"].at[2].set(0.0)

# # perturb = jax.random.normal(key, params["gkv"].shape).at[1::3].set(0.0)

# # params["X_km1"] = perturb * params["h"]
# # params["X_km1"] = params["X_km1"].at[:3].set(0.0)

# # params["g_km1v"] = params["gkv"] + params["X_km1"]
# print("Initial perturbation added to cable nodes (except base):", params["X_km1"])

run_simulation_jit = jax.jit(run_simulation, static_argnums=(1,))
t0 = time.time()
_ ,_r,_= run_simulation_jit(params)
_.block_until_ready()
print("First run:", time.time() - t0, "s")
print("starting simulation")
print("simulation running ..........")

##forward Simulaion
start = time.perf_counter()


traj,traj_R,traj_fk = run_simulation_jit(params, num_steps=steps)
traj.block_until_ready()
end = time.perf_counter()

filename = f"x_extractmodes.npz"



np.savez(filename,
         trajectories=np.array(traj),
        rot_mat = np.array(traj_R),
        fk_dat  = np.array(traj_fk),
         params=params)

print(f"Simulation took {end - start:.3f} seconds")
print("Trajectory shape:", traj.shape)


# print("Trajectory shape:", traj[-1])
# traj = np.array(traj)
# traj_R = np.array(traj_R)

# print(traj_R[-1])

# plot_xz_trajectory(traj[:,1:], N=params["N"])
# animate_traj(np.array(traj[:,1:]), traj_R,duration_sec=tf, fps=60, stl_file="models/Assembly.STL")