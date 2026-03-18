import jax
import jax.numpy as jnp
import numpy as np
import time, os

from mpi4py import MPI

from simulation.integrator import run_simulation
from simulation.utils import make_initial_configuration
from simulation.catenary import fixed_point_iteration

jax.config.update("jax_platforms", "cpu")

# ----------------------------
# MPI setup
# ----------------------------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------
# Problem setup
# ----------------------------

L = 15
N = 20
h = 1e-4
tf = 200
steps = int(tf / h)

mu = 0.03
M = 2.5
spk = 5

rho = jnp.array([0.15,0.0,0.3]) * 0

I = jnp.array([
    [0.105,0,0],
    [0,0.105,0],
    [0,0,0.140]
])

uk = 0.0
tau = jnp.array([0.0,0.0,0.0])

total_weight = 9.81 * (mu*(L-spk) + M)

# ----------------------------
# Param builder
# ----------------------------

def make_params():

    params = dict(

        mu=jnp.array(mu),
        M=jnp.array(M),
        h=jnp.array(h),
        g=jnp.array(-9.81),
        L=jnp.array(L),

        N=N,

        l_k=(L-spk)/N,

        EA=jnp.array(1e5),

        gkv=make_initial_configuration(L-spk, N, rho, spk),
        g_km1v=make_initial_configuration(L-spk, N, rho, spk),

        X_km1=jnp.zeros_like(make_initial_configuration(L, N, rho, spk)),

        force=1.1 * total_weight,

        omega=jnp.array(1.0),

        delta_base_pos=jnp.array([0.0,0.0,0.0]),

        step=0,

        kappa=jnp.array(1),

        b=jnp.array(0.3),
        d=jnp.array(0.15),

        rho=rho,

        base_pos=jnp.array([0.0,0.0,0.0]),

        u_k=jnp.array(uk),

        a=jnp.arange(1.0, N+1, dtype=jnp.float32),

        J=I,

        f_km1=jnp.array([0.0,0.0,0.0]),
        fk=jnp.array([0.0,0.0,0.0]),

        tau=tau,

        R_v=jnp.eye(3),
        R=jnp.eye(3)
    )

    # initial velocity
    params["g_km1v"] = params["gkv"]
    params["g_km1v"] = params["gkv"].at[-3].set(
        params["gkv"][-3] - params["h"]
    )

    params["X_km1"] = params["g_km1v"] - params["gkv"]

    return params


# ----------------------------
# Single case (same as reference)
# ----------------------------

def single_case(w_base, params):

    p = params.copy()

    p["omega"] = jnp.array(w_base)

    return run_simulation(p, num_steps=steps)


# ----------------------------
# Parallel run
# ----------------------------

def run_and_save(freqs, params):

    for w_base in freqs:

        start = time.perf_counter()

        traj, traj_R, traj_fk = single_case(w_base, params)

        traj = np.array(traj)

        filename = f"1.1kgresults1to5/sim_rank{rank}_omega{float(w_base):.3f}.npz"

        np.savez(
            filename,
            trajectories=traj,
            rot_mat=np.array(traj_R),
            fk_dat=np.array(traj_fk),
            params=params
        )

        print(
            f"[Rank {rank}] Saved {filename} in {time.perf_counter()-start:.2f}s"
        )


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":

    params = make_params()

    # frequency sweep (same style as before)
    # freqs_low = jnp.linspace(0,0.1,11)
    # freqs_high = jnp.round(jnp.linspace(0.1,5.0,20), decimals=1)

    # all_freqs = jnp.concatenate([freqs_low[:-1], freqs_high])

    # freqs = jnp.concatenate([
    # jnp.linspace(0.0,1.0,100,endpoint=False),
    # jnp.linspace(1.0,5.0,100)
    # ])
    freqs = jnp.linspace(1.0,5.0,100)

    # split across MPI ranks
    freqs_split = np.array_split(np.array(freqs), size)
    my_freqs = freqs_split[rank]

    os.makedirs("1.1kgresults1to5", exist_ok=True)

    print(f"[Rank {rank}] running {len(my_freqs)} cases")

    run_and_save(my_freqs, params)

    comm.Barrier()

    if rank == 0:
        print("All ranks finished.")