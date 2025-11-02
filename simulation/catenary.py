import jax
import jax.numpy as jnp


def fgrad_Ve(EA, l_k, q_a, q_ap1):
    dq = q_ap1 - q_a
    norm_dq = jnp.linalg.norm(dq) + 1e-12
    stretch = norm_dq - l_k
    return (EA / l_k) * (stretch / norm_dq) * dq


def residual_fn_fixed_point(Xk, params):
    mu, M, h, g, l_k, EA ,N , force = (
        params['mu'], params['M'], params['h'], params['g'],
        params['l_k'], params['EA'] , params['N'], params['force']
    )
    e3 = jnp.array([0.0, 0.0, 1.0])

    gk   = Xk.reshape(-1, 3)
    residuals = jnp.zeros_like(gk)

    residuals = residuals.at[0].set(gk[0])

    def body_fn(i, residuals):
        p = i + 1

        DqVk = - mu * l_k * g * e3 \
            + fgrad_Ve(EA, l_k, gk[p-1], gk[p]) \
            - fgrad_Ve(EA, l_k, gk[p], gk[p+1])

        res =  DqVk

        return residuals.at[p].set(res)

    residuals = jax.lax.fori_loop(0, N-1, body_fn, residuals)

    # last node
    fgrad_Ve_kN = fgrad_Ve(EA, l_k, gk[-2], gk[-1])
    DqVk_last  = -(0.5 * mu * l_k + M) * g * e3 + fgrad_Ve_kN
    res_last =  DqVk_last - params['force']

    residuals = residuals.at[-1].set(res_last)

    return residuals.reshape(-1)
def fixed_point_iteration(params, tol=1e-10, max_iter=50):
    Xk = 1.5*params['g_km1v']
    F = lambda x: residual_fn_fixed_point(x, params)
    jac_fn = jax.jit(jax.jacfwd(F))


    def cond_fun(state):
        Xk, r, k = state
        return (jnp.linalg.norm(r) > tol) & (k < max_iter)

    def body_fun(state):
        Xk, r, k = state
        J = jac_fn(Xk)      # can also cache with jax.jit(jacfwd(...))
        delx = jnp.linalg.solve(J, -r)
        Xk_new = Xk + delx
        return (Xk_new, F(Xk_new), k+1)

    init_state = (Xk, F(Xk), 0)
    Xk, r, k = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return Xk

