import jax
import jax.numpy as jnp
from functools import partial
from .residual import residual_fn,attitude_residual,cayley

from .newton_iter import newton_solve as newton_solve_py

# Mark residual_fn (arg 0) and params (arg 2) as static
newton_solve = jax.jit(newton_solve_py, static_argnums=(0,))


def base_path_velocity(t):
    """Velocity profile for a 30 s path: line, slalom, turn, sweep, eight, circle."""
    t1 = 4.0
    t2 = 10.0
    t3 = 15.0
    t4 = 20.0
    t5 = 26.0
    t6 = 30.0

    straight_vel = jnp.array([0.8, 0.0])

    slalom_t = t - t1
    slalom_vel = jnp.array([
        0.8,
        0.9 * jnp.sin(2.0 * jnp.pi * slalom_t / 3.0),
    ])

    turn_t = t - t2
    turn_radius = 2.5
    turn_omega = jnp.pi / (t3 - t2)
    turn_angle = turn_omega * turn_t
    semicircle_vel = turn_radius * turn_omega * jnp.array([
        jnp.cos(turn_angle),
        jnp.sin(turn_angle),
    ])

    sweep_t = t - t3
    sweep_vel = jnp.array([
        -0.55,
        0.55 + 0.25 * jnp.sin(2.0 * jnp.pi * sweep_t / (t4 - t3)),
    ])

    eight_t = t - t4
    eight_omega = 2.0 * jnp.pi / (t5 - t4)
    figure_eight_vel = jnp.array([
        1.4 * eight_omega * jnp.cos(eight_omega * eight_t),
        0.8 * 2.0 * eight_omega * jnp.cos(2.0 * eight_omega * eight_t),
    ])

    circle_t = t - t5
    circle_radius = 1.2
    circle_omega = 2.0 * jnp.pi / (t6 - t5)
    circle_angle = circle_omega * circle_t
    circle_vel = circle_radius * circle_omega * jnp.array([
        -jnp.sin(circle_angle),
        jnp.cos(circle_angle),
    ])

    return jnp.where(
        t < t1,
        straight_vel,
        jnp.where(
            t < t2,
            slalom_vel,
            jnp.where(
                t < t3,
                semicircle_vel,
                jnp.where(
                    t < t4,
                    sweep_vel,
                    jnp.where(t < t5, figure_eight_vel, circle_vel),
                ),
            ),
        ),
    )


def normalize_safe(v, eps=1e-8):
    return v / (jnp.linalg.norm(v) + eps)


def tracking_tau(params, base_velocity, base_pos_next):
    """PID position tracking converted into attitude torque."""
    h = params["h"]
    spk, qk_vec = jnp.split(params["gkv"], [1])
    _, del_qkm1_vec = jnp.split(params["X_km1"], [1])
    qk = qk_vec.reshape(-1, 3)
    del_qkm1 = del_qkm1_vec.reshape(-1, 3)

    drone_pos = qk[-1]
    drone_vel = del_qkm1[-1] / h
    cable_length = params["L"] - spk[0]

    desired_pos = base_pos_next + jnp.array([0.0, 0.0, cable_length])
    desired_vel = base_velocity

    pos_error = desired_pos - drone_pos
    vel_error = desired_vel - drone_vel
    integral_error = params["pid_error_integral"] + pos_error * h
    integral_limit = params["pid_integral_limit"]
    integral_error = jnp.clip(integral_error, -integral_limit, integral_limit)

    acc_cmd = (
        params["pid_kp"] * pos_error
        + params["pid_kd"] * vel_error
        + params["pid_ki"] * integral_error
    )
    acc_limit = params["pid_acc_limit"]
    acc_cmd = jnp.clip(acc_cmd, -acc_limit, acc_limit)

    gravity = jnp.array([0.0, 0.0, params["g"]])
    desired_thrust_dir = normalize_safe(acc_cmd - gravity)
    body_z = params["R"] @ jnp.array([0.0, 0.0, 1.0])
    attitude_error = jnp.cross(body_z, desired_thrust_dir)
    angular_velocity = 2.0 * params["f_km1"] / h

    tau = params["att_kp"] * attitude_error - params["att_kd"] * angular_velocity
    tau = jnp.clip(tau, -params["tau_limit"], params["tau_limit"])
    return tau, integral_error


# ----------------------------------------
# One simulation step
# ----------------------------------------
def step_fn(carry, _):
    params = dict(carry)
    t = params["step"] * params["h"]
    control_idx = jnp.floor(t / params["mpc_control_dt"]).astype(jnp.int32)
    control_idx = jnp.clip(control_idx, 0, params["mpc_controls"].shape[1] - 1)
    control = params["mpc_controls"][:, control_idx]
    params["u_k"] = -control[0]
    params["force"] = control[1]
    params["tau"] = control[2:5]

    # Base excitation and tracking controller for this step.
    omega_b = 2*jnp.pi*params["omega"]
    z_pos = 0*0.1 * omega_b * jnp.cos(omega_b * t)
    x_pos, y_pos = 0*base_path_velocity(t)
    base_velocity = jnp.array([x_pos, y_pos, z_pos])
    params["delta_base_pos"] = base_velocity * params["h"]
    base_pos_next = params["base_pos"] + params["delta_base_pos"]
    # params["tau"], pid_error_integral = tracking_tau(params, base_velocity, base_pos_next)
    # params["tau"] = 0*params["tau"]  # Use torque as a proxy for thrust in the attitude residual.

    x0 = params['X_km1']
    # === Newton solve ===
    del_Xk = newton_solve(
                        residual_fn,
                        x0,
                        params=params,
                        tol=1e-12,
                        max_iter=200,
                        log=False
    )
    fk0 = params['f_km1']
    del_fk = newton_solve(attitude_residual,fk0,params = params,tol = 1e-12,max_iter = 20,log = False)

    # params['u_k'] =0*jnp.cos(jnp.pi*t)
    # === State update ===
    params["step"] += 1
    params["X_km1"] = del_Xk
    params["g_km1v"] = params["gkv"]
    params["gkv"] = params["gkv"] + del_Xk
    params['fk'] = del_fk
    params['f_km1'] = del_fk
    params["R"] = params["R"]@cayley(del_fk)
    params["base_pos"] = base_pos_next
    # params["pid_error_integral"] = pid_error_integral

    return params, (params["gkv"],params['R'],params['fk'])


# ----------------------------------------
# Simulation runner
# ----------------------------------------
@partial(jax.jit, static_argnums=(1,))
def run_simulation(params, num_steps=10):

    params, traj = jax.lax.scan(step_fn, params, None, length=num_steps)

    return traj
