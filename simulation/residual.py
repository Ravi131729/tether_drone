import jax.random as random
import jax
import jax.numpy as jnp


def compute_M1(mu, l_k):
    return (1.0 / 3.0) * mu * l_k

def compute_M12(mu, l_k):
    return (1.0 / 6.0) * mu * l_k

def grad_Ve(EA, l_k, q_a, q_ap1):
    dq = q_ap1 - q_a
    norm_dq = jnp.linalg.norm(dq)
    stretch = norm_dq - l_k
    return (EA / l_k) * (stretch / norm_dq) * dq

def unpack_Xk(Xk):
    return Xk.reshape(-1, 3)

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

    print("Inverse of I - c_hat:", jax.numpy.linalg.inv(I - c_hat))
    return (I + c_hat) @ jax.numpy.linalg.inv(I - c_hat)
def compute_Jd(J):
    trJ = jnp.trace(J)
    return 0.5 * trJ * jnp.eye(3) - J

def residual_fn(Xk, params):
    # --- unpack once ---
    mu, M, h, g, l_k, EA, N, force,  = (
        params['mu'], params['M'], params['h'], params['g'],
        params['l_k'], params['EA'], params['N'], params['force'],

    )

    e3 = jnp.array([0., 0., 1.])
    gk       = unpack_Xk(params['gkv'])
    g_km1    = unpack_Xk(params['g_km1v'])
    del_qkm1 = unpack_Xk(params['X_km1'])
    del_qk   = unpack_Xk(Xk)

    residuals = jnp.zeros_like(del_qk)

    # =====================================================
    # Base node residual
    # =====================================================
    residuals = residuals.at[0].set(del_qk[0] - params["delta_base_pos"])

    # =====================================================
    # Interior nodes (string elements)
    # =====================================================
    def body_fn(i, residuals):
        p = i + 1
        M1_k  = compute_M1(mu, l_k)
        M12_k = compute_M12(mu, l_k)

        DqVk = - mu * l_k * g * e3 \
             + grad_Ve(EA, l_k, gk[p-1], gk[p]) \
             - grad_Ve(EA, l_k, gk[p], gk[p+1])

        res = (
            (1/h) * M12_k * del_qk[p-1]
          + (2/h) * M1_k  * del_qk[p]
          + (1/h) * M12_k * del_qk[p+1]
          - (1/h) * M12_k * del_qkm1[p-1]
          - (2/h) * M1_k  * del_qkm1[p]
          - (1/h) * M12_k * del_qkm1[p+1]
          + h * DqVk
        )
        return residuals.at[p].set(res)

    residuals = jax.lax.fori_loop(0, N-1, body_fn, residuals)

    # =====================================================
    # Last node translational residual
    # =====================================================
    M1_k  = compute_M1(mu, l_k)
    M12_k = compute_M12(mu, l_k)
    grad_Ve_kN = grad_Ve(EA, l_k, gk[-2], gk[-1])
    DqVk_last  = -(0.5 * mu * l_k + M) * g * e3 + grad_Ve_kN

    res_last = (
        (1/h) * (M1_k + M) * del_qk[-1]
      + (1/h) * M12_k * del_qk[-2]
      - (1/h) * (M1_k + M) * del_qkm1[-1]
      - (1/h) * M12_k * del_qkm1[-2]
      + h * DqVk_last - h * force
    )
    residuals = residuals.at[-1].set(res_last)

    # # =====================================================
    # # Attitude residual
    # # =====================================================
    # F_k   = cayley(del_qk[-1])
    # F_km1 = cayley(del_qkm1[-1])

    # res_att = (
    #     (1.0/h) * vee(F_k @ Jd - Jd @ F_k.T - Jd @ F_km1 + F_km1.T @ Jd)
    #   - h * tau
    # )
    # residuals = residuals.at[-1].set(res_att)

    # =====================================================
    # Return flat vector
    # =====================================================
    return residuals.reshape(-1)



