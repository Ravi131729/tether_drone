import jax.numpy as jnp

def make_initial_configuration(L, N, eps=1e-3):
    """
    Create an initial straight-line configuration for the string.

    Args:
        L   (float): total length of the string
        N   (int):   number of elements
        eps (float): small perturbation factor (not used here, but can be handy)

    Returns:
        (N+1)*3 vector flattened (shape: (3*(N+1),))
    """
    lk = L / N
    x_coords = jnp.arange(0, N + 1) * lk * (1.0 / jnp.sqrt(2.0))
    y_coords = jnp.zeros(N + 1)
    z_coords = jnp.arange(0, N + 1) * lk * (1.0 / jnp.sqrt(2.0))
    X0 = jnp.stack([x_coords, y_coords, z_coords], axis=1)
    return X0.reshape(-1)


def unpack_Xk(Xk):
    """
    Convert flattened (3*(N+1),) vector into (N+1, 3) array.
    """
    return Xk.reshape(-1, 3)
