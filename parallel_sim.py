import jax
import jax.numpy as jnp
import numpy as np
import time,math

from simulation.integrator import run_simulation
from simulation.utils import make_initial_configuration
from simulation.catenary import fixed_point_iteration
import os

# Create folder if it doesn't exist
out_dir = "results_"
os.makedirs(out_dir, exist_ok=True)



jax.config.update("jax_platforms", "cpu")
# ----------------------------
# Problem setup
# ----------------------------
L = 20.0
N = 50
h = 1e-4
tf = 500.0
steps = int(tf / h)
total_weight = 9.81 * (0.06 * L + 2.0)

def make_params(L, N, h, total_weight):
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
        force = jnp.array([0, 0.0, 1.1 * total_weight]),
        delta_base_pos = jnp.array([0.0, 0.0, 0.0]),
        step = 0,
    )
    params["gkv"] = fixed_point_iteration(params)
    params["g_km1v"] = params["gkv"]
    params["g_km1v"] = params["gkv"].at[-3].set(params["gkv"][-3] - params["h"])
    params["X_km1"] = params["g_km1v"] - params["gkv"]
    return params

# ----------------------------
# Single case (for vmap)
# ----------------------------

def single_case(w_base, params):
    p = params.copy()
    p["omega"] = jnp.array(w_base)
    return run_simulation(p, num_steps=steps)

# batched_run = jax.vmap(single_case, in_axes=(0, None))
from mpi4py import MPI
import jax.numpy as jnp
import numpy as np
import os, time, math

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def run_and_save(freqs, params, steps, N, tf, out_dir):
    # each rank runs its subset
    for i, w_base in enumerate(freqs):
        start = time.perf_counter()
        traj = single_case(w_base, params)   # just one case
        traj = np.array(traj).reshape(steps, N+1, 3)
        filename = os.path.join(out_dir, f"sim_rank{rank}_omega{float(w_base):.3f}_tf{tf}s_L{L}_N{N}.npz")
        np.savez(filename, trajectories=traj, params=params)
        print(f"[Rank {rank}] Saved {filename} in {time.perf_counter()-start:.2f}s")

if __name__ == "__main__":
    params = make_params(L, N, h, total_weight)

    # define frequencies
    freqs_low = jnp.linspace(0, 0.1, 11)
    freqs_high = jnp.round(jnp.linspace(0.1, 4.0, 20),decimals = 1)
    all_freqs = jnp.concatenate([freqs_low[:-1], freqs_high])  # 50 total

    # split frequencies across ranks
    freqs_split = np.array_split(np.array(all_freqs), size)
    my_freqs = freqs_split[rank]

    out_dir = "./results"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Rank {rank}] running {len(my_freqs)} cases...")
    run_and_save(my_freqs, params, steps, N, tf, out_dir)

    comm.Barrier()
    if rank == 0:
        print("All ranks finished.")


# ----------------------------
# Run batch and save files
# ----------------------------
# if __name__ == "__main__":
#     freqs = jnp.array([0.0 ,0.01, 0.02, 0.03,0.04])  # example batch
#     params = make_params(L, N, h, total_weight)
#     print("running suimulation.....")
#     start = time.perf_counter()
#     trajs = batched_run(freqs, params)   # shape (batch, steps, (N+1)*3)
#     jax.block_until_ready(trajs)
#     end = time.perf_counter()
#     print(f"Ran {len(freqs)} sims in {end-start:.2f}s")

#     # Save each trajectory separately
#     for i, w_base in enumerate(freqs):
#         traj = np.array(trajs[i]).reshape(steps, N+1, 3)
#         filename = os.path.join(out_dir, f"sim_test_omega{float(w_base):.3f}_tf{tf}s_L{L}_N{N}.npz")
#         np.savez(filename, trajectories=traj, params=params)
#         print(f"Saved {filename}")
# def run_and_save(freqs, params, steps, N, tf, out_dir):
#     print(f"\nRunning {len(freqs)} sims for freqs {np.array(freqs)}...")
#     start = time.perf_counter()

#     # Run batch
#     trajs = batched_run(freqs, params)   # (batch, steps, (N+1)*3)
#     jax.block_until_ready(trajs)

#     end = time.perf_counter()
#     batch_time = end - start
#     print(f"Batch finished in {batch_time:.2f}s")

#     # Save results
#     for i, w_base in enumerate(freqs):
#         traj = np.array(trajs[i]).reshape(steps, N+1, 3)
#         filename = os.path.join(out_dir, f"sim_test_omega{float(w_base):.3f}_tf{tf}s_L{L}_N{N}.npz")
#         np.savez(filename, trajectories=traj, params=params)
#         print(f"  Saved {filename}")

#     return batch_time

# if __name__ == "__main__":
#     # Setup
#     params = make_params(L, N, h, total_weight)
#     freqs_low = jnp.linspace(0, 0.1, 11)

#     # High range: 0.1–4 with 50 sims
#     freqs_high = jnp.linspace(0.1, 4.0, 40)

#     # Combine them (remove duplicate 0.1)
#     all_freqs = jnp.concatenate([freqs_low[:-1], freqs_high])
#     # all_freqs = jnp.linspace(0.0, 0.5, 50)   # 50 frequency cases
#     batch_size = 5                          # run 5 at a time
#     out_dir = "./results"
#     os.makedirs(out_dir, exist_ok=True)

#     num_batches = math.ceil(len(all_freqs) / batch_size)
#     total_time = 0.0

#     for k in range(0, len(all_freqs), batch_size):
#         freqs = all_freqs[k:k+batch_size]
#         batch_idx = k // batch_size + 1

#         # Run + save
#         batch_time = run_and_save(freqs, params, steps, N, tf, out_dir)
#         total_time += batch_time

#         # Progress + ETA
#         avg_time = total_time / batch_idx
#         remaining = (num_batches - batch_idx) * avg_time
#         print(f"[Batch {batch_idx}/{num_batches}] "
#               f"Elapsed: {total_time:.1f}s | ETA: {remaining:.1f}s")