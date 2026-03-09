import jax.random as random
import jax
import jax.numpy as jnp
from simulation.integrator import run_simulation
from simulation.utils import make_initial_configuration
import jax.numpy as jnp
import jax
import numpy as np
from plotter.plot import plot_xz_trajectory
from plotter.animate import animate_traj
from simulation.catenary import fixed_point_iteration
jax.config.update("jax_platforms", "cpu")
def hat(v):
    return jax.numpy.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])

def vee(S):
    return jax.numpy.array([S[2, 1], S[0, 2], S[1, 0]])

def cayley(c):
    I = jax.numpy.eye(3)
    c_hat = hat(c)

    # print("Inverse of I - c_hat:", jax.numpy.linalg.inv(I - c_hat))
    return (I + c_hat) @ jax.numpy.linalg.inv(I - c_hat)
def compute_Jd(J):
    trJ = jnp.trace(J)
    return 0.5 * trJ * jnp.eye(3) - J

def attitude_residual(fk,params):

    # --- unpack once ---
    J ,tau , f_km1,h,R= params['J'],params['tau'],params['f_km1'],params['h'],params['R']
    Jd = compute_Jd(J)
    F_k   = cayley(fk)
    F_km1 = cayley(f_km1)
    term = h * 9.81 * hat(jnp.array([0.0, 0.0, -1.0])) @ R.T @ jnp.array([0.0, 0.0, 1.0])
    res_att = (
        (1.0/h) * vee(F_k @ Jd - Jd @ F_k.T - Jd @ F_km1 + F_km1.T @ Jd)
      - h * R.T@tau # - term
    )
    return res_att
def newton_solve(
    residual_fn,          # function F(x, params)
    x0,                   # initial guess
    params=None,          # optional parameters to pass to residual_fn
    tol=1e-12,
    max_iter=20,
    log=False
):

    if params is not None:
        F = lambda x: residual_fn(x, params)
    else:
        F = residual_fn

    jac_fn = jax.jacfwd(F)

    def cond_fun(state):
        Xk, r, k, dx_norm = state
        return ((jnp.linalg.norm(r) > tol) | (dx_norm > tol)) & (k < max_iter)

    def body_fun(state):
        Xk, r, k, dx_norm = state
        J = jac_fn(Xk)
        delx = jnp.linalg.solve(J, -r)
        Xk_new = Xk + delx
        r_new = F(Xk_new)
        dx_norm_new = jnp.linalg.norm(delx)
        return (Xk_new, r_new, k+1, dx_norm_new)

    init_state = (x0, F(x0), 0, jnp.inf)
    Xk, r, k, dx_norm = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return Xk



# ----------------------------------------
# One simulation step
# ----------------------------------------
def step_fn(carry, _):
    params = carry

    fk0 = params['f_km1']
    del_fk = newton_solve(attitude_residual,fk0,params = params,tol = 1e-12,max_iter = 200,log = False)
    # === Time update ===
    t = params["step"] * params["h"]
    params = dict(params)  # copy before mutation
    params["step"] += 1

    # Base excitation
    omega_b = 2*jnp.pi
    tau_y = 0.01*omega_b**2*jnp.cos(2*omega_b*t)
    params["tau"] = jnp.array([0.0,tau_y,0.0])

    params['fk'] = del_fk
    params['f_km1'] = del_fk
    params["R"] = params["R"]@cayley(del_fk)

    return params, (params['R'],params['fk'],params['tau'])


# ----------------------------------------
# Simulation runner
# ----------------------------------------
def run_simulation(params, num_steps=10):

    params, traj = jax.lax.scan(step_fn, params, None, length=num_steps)

    return traj

# ----------------------------
# Problem setup
# ----------------------------

h = 1e-3
tf =10


steps = int(tf / h)

I = jnp.array([[0.105,0,0],  #drone moment of inertia
              [0,0.105,0],
              [0,0,0.140]])
                       # thrust force
R = jnp.eye(3)
tau = [ 1.0,0.0,0.0]                        #torque on drone

params = dict(
    step = 0,
    J = I,
    f_km1 = jnp.array([0.0,0.0,0.0]),
    fk = jnp.array([0.0,0.0,0.0]),
    tau = jnp.array(tau),
    R = R,
    h=h
)

import time

run_simulation_jit = jax.jit(run_simulation, static_argnums=(1,))
t0 = time.time()
_ ,_,_= run_simulation_jit(params)
_.block_until_ready()
print("First run:", time.time() - t0, "s")
print("starting simulation")
print("simulation running ..........")

##forward Simulaion
start = time.perf_counter()


traj_R,traj_fk,tau = run_simulation_jit(params, num_steps=steps)
traj_R.block_until_ready()
end = time.perf_counter()
print(f"Simulation took {end - start:.3f} seconds")
print(traj_R[-1])
import matplotlib.pyplot as plt
traj_fk = traj_fk[::100]
timesteps,_ = traj_fk.shape
omega = np.zeros_like(traj_fk)
for i in range(timesteps):
  omega[i] = (1/h)* vee((cayley(traj_fk[i]) - np.eye(3)))

plt.plot(np.arange(timesteps),omega)
plt.show()
traj_fk = tau[::100]
timesteps,_ = traj_fk.shape
plt.plot(np.arange(timesteps),traj_fk)
plt.show()