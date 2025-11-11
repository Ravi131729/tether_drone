import jax
import jax.numpy as jnp

from .residual import residual_fn,attitude_residual,cayley

from .newton_iter import newton_solve as newton_solve_py

# Mark residual_fn (arg 0) and params (arg 2) as static
newton_solve = jax.jit(newton_solve_py, static_argnums=(0,))


# ----------------------------------------
# One simulation step
# ----------------------------------------
def step_fn(carry, _):
    params = carry
    x0 = params['X_km1']
    # === Newton solve ===
    del_Xk = newton_solve(
                        residual_fn,
                        x0,
                        params=params,
                        tol=1e-12,
                        max_iter=20,
                        log=False
    )
    fk0 = params['f_km1']
    del_fk = newton_solve(attitude_residual,fk0,params = params,tol = 1e-12,max_iter = 20,log = False)
    # === Time update ===
    t = params["step"] * params["h"]
    params = dict(params)  # copy before mutation
    params["step"] += 1

    # Base excitation
    omega_b = 2*jnp.pi*params["omega"]
    z_pos = 0*0.1 * omega_b * jnp.cos(omega_b * t)
    params["delta_base_pos"] = jnp.array([0.0, 0.0, z_pos]) * params["h"]

    # === State update ===
    params["X_km1"] = del_Xk
    params["g_km1v"] = params["gkv"]
    params["gkv"] = params["gkv"] + del_Xk
    params['f_km1'] = del_fk
    params["R"] = params["R"]@cayley(del_fk)

    return params, (params["gkv"],params['R'])


# ----------------------------------------
# Simulation runner
# ----------------------------------------
def run_simulation(params, num_steps=10):

    params, traj = jax.lax.scan(step_fn, params, None, length=num_steps)

    return traj
