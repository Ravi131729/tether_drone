import jax
import jax.numpy as jnp

def newton_solve(
    residual_fn,          # function F(x, params)
    x0,                   # initial guess
    params=None,          # optional parameters to pass to residual_fn
    tol=1e-12,
    max_iter=20,
    log=False
):
    # Wrap residual to include params if given
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
