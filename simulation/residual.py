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

    # print("Inverse of I - c_hat:", jax.numpy.linalg.inv(I - c_hat))
    return (I + c_hat) @ jax.numpy.linalg.inv(I - c_hat)
def compute_Jd(J):
    trJ = jnp.trace(J)
    return 0.5 * trJ * jnp.eye(3) - J

def compute_M_terms(mu,qk,l_k,a):

    N = 20

    dq  = qk[ :-1] - qk[1: ]

    M3 =  (1/3) * mu * l_k *(3*N**2 + 3*N + 1 - 6*N*a - 3*a + 3*a**2) / N**2

    coeff23 = (1/6) * mu * (1 + 3*N - 3*a) / N     # (N,)
    coeff31 = (1/6) * mu * (2 + 3*N - 3*a) / N     # (N,)

    M23 = coeff23[:, None] * dq   # (N,3)
    M31 = coeff31[:, None] * dq   # (N,3)

    return M3, M23, M31



def residual_fn(Xk, params):
    # --- unpack once ---
    mu, M, h, g, l_k, EA, N, force, L , kappa ,delta_base_pos , d,u_k ,base_pos , b , rho,R = (
        params['mu'], params['M'], params['h'], params['g'],
        params['l_k'], params['EA'], params['N'], params['force'],params['L'],params['kappa'] ,params["delta_base_pos"],
        params['d'] , params['u_k'],params['base_pos'] , params['b'] , params['rho'],params['R']


    )

    del_spk ,Xk_vec= xl,del_qk = jnp.split(Xk,[1,])
    del_qk   = Xk_vec.reshape(-1,3)

    del_spkm1 ,X_km1_vec = jnp.split(params['X_km1'],[1,])
    del_qkm1 = X_km1_vec.reshape(-1,3)

    spkm1 ,q_km1_vec = jnp.split(params['g_km1v'],[1,])
    q_km1 = q_km1_vec.reshape(-1,3)

    spk,qk_vec = jnp.split(params['gkv'],[1,])
    qk = qk_vec.reshape(-1,3)

    l_k = (L-spk) / N

    l_km1 = (L - spkm1)/N

    a = jnp.arange(1.0, 20+1, dtype=jnp.float32)

    e3 = jnp.array([0.0,0.0,1.0])
    spk_residual = jnp.zeros_like(del_spk)

    M0k = (1/3)*mu*l_k + mu*spk + kappa
    M0km1 = (1/3)*mu*l_km1 + mu * spkm1 + kappa

    M3_k1 , M23_k , M31_k = compute_M_terms(mu,qk,l_k,a)
    dot1 = M31_k[1:] * del_qk[1:-1] + M23_k[0:-1] * del_qk[1:-1]
    sum1 = jnp.sum(dot1)

    M3_km1 , M23_km1 , M31_km1 = compute_M_terms(mu,q_km1,l_k,a)
    dot2 = M31_km1[1:] * del_qkm1[1:-1] + M23_km1[0:-1] * del_qkm1[1:-1]
    sum2 = jnp.sum(dot2)

    dq = qk[1:] - qk[:-1]  # (N,3)
    dq_norm2 = jnp.sum(dq**2, axis=1)

    D_spk_Vk  =  (
        - mu * g * jnp.dot(base_pos,e3) + mu*g*d*jnp.sin((spk - b)/d)
        + (1/(2)) * mu* g * jnp.dot(e3,base_pos) + (1/(2*N)) * mu* g * jnp.sum(e3*qk[:-1]) + (1/(2*N)) * mu* g * jnp.sum(e3*qk[1:])
        + (1/(2*N))*(EA/(l_k**2)) *jnp.sum(dq_norm2 - l_k**2)

    )

    # =====================================================
    # winch cable residual
    # =====================================================
    spk_residual  = (
                    (1/h) * M0k * del_spk
                  + (1/h) * jnp.dot(M31_k[0] , del_qk[0])
                  + (1/h)* sum1
                  + (1/h) *jnp.dot(M23_k[-1],del_qk[-1])

                  -  (1/h) * M0km1 * del_spkm1
                  - (1/h) * jnp.dot(M31_km1[0] ,del_qkm1[0])
                  - (1/h)* sum2
                  - (1/h) *jnp.dot(M23_km1[-1],del_qkm1[-1])

                  - (0.5/h) * mu * jnp.dot(delta_base_pos,delta_base_pos)
                  - (1/3)*(1/h)*mu* jnp.dot(del_spk,del_spk)

                  + (mu/(6*N*h))*(jnp.sum(del_qk[:-1] * del_qk[:-1]) - jnp.sum(del_qk[1:] * del_qk[1:]) - jnp.sum(del_qk[:-1] * del_qk[1:]))
                  + h*D_spk_Vk - (h/d) * u_k
                  + (h/(2*l_k**2)) * ((mu/(h**2))* (del_spk**2)  + EA ) * (jnp.linalg.norm(qk[1] - qk[0]) - l_k)**2
    )
    # jax.debug.print("value = {}", h * D_spk_Vk + (h/d) * u_k)

    # spk_residual = del_spk
    string_residuals = jnp.zeros_like(del_qk)




    # =====================================================
    # Base node residual
    # =====================================================
    string_residuals = string_residuals.at[0].set(del_qk[0]  + qk[0] - base_pos - params["delta_base_pos"] - rho )

    # =====================================================
    # Interior nodes (string elements)
    # =====================================================
    def body_fn(i, string_residuals):
        p = i + 1
        M1_k  = compute_M1(mu, l_k)
        M12_k = compute_M12(mu, l_k)
        M12_km1 = compute_M12(mu, l_km1)
        M12_k = compute_M12(mu, l_km1)
        M1_km1  = compute_M1(mu, l_km1)



        DqVk = - mu * l_k * g * e3 \
             + grad_Ve(EA, l_k, qk[p-1], qk[p]) \
             - grad_Ve(EA, l_k, qk[p], qk[p+1])

        res = (
            (1/h) * M12_k * del_qk[p-1]
          + (2/h) * M1_k  * del_qk[p]
          + (1/h) * M12_k * del_qk[p+1]
          + (1/h)*del_spk * (M31_k[p] + M23_k[p-1])
          - (1/h) * M12_km1 * del_qkm1[p-1]
          - (2/h) * M1_km1  * del_qkm1[p]
          - (1/h) * M12_km1 * del_qkm1[p+1]
          - (1/h)*del_spkm1 * (M31_km1[p] + M23_km1[p-1])
          - (mu/(6*N*h)) * (1 + 3 * N -3*p) * del_spk * del_qk[p+1]
          + (mu/(3*N*h)) * del_spk * del_qk[p]
          + (mu/(6*N*h)) * (5 + 3*N- 3*p ) * del_spk * del_qk[p-1]
          + h * DqVk
        )
        return string_residuals.at[p].set(res)

    string_residuals = jax.lax.fori_loop(0, N-1, body_fn, string_residuals)

    # =====================================================
    # Last node translational residual
    # =====================================================
    M1_k  = compute_M1(mu, l_k)
    M12_k = compute_M12(mu, l_k)
    grad_Ve_kN = grad_Ve(EA, l_k, qk[-2], qk[-1])
    DqVk_last  = -(0.5 * mu * l_k + M) * g * e3 + grad_Ve_kN

    res_last = (
        (1/h) * (M1_k + M) * del_qk[-1]
      + (1/h) * M12_k * del_qk[-2]
      + (1/h) * M23_k[-1] * del_spk
      - (1/h) * (M1_k + M) * del_qkm1[-1]
      - (1/h) * M12_k * del_qkm1[-2]
      - (1/h) * M23_km1[-1] * del_spkm1
      + (mu/(6*N*h)) * del_spk *del_qk [-1]
      + (mu/(3*N*h)) * del_spk * del_qk[-2]

      + h * DqVk_last - h * force*R@e3
    )
    string_residuals = string_residuals.at[-1].set(res_last)
    residual = jnp.concatenate([jnp.atleast_1d(spk_residual), string_residuals.reshape(-1)])

    return residual

def attitude_residual(fk,params):

    # --- unpack once ---
    J ,tau , f_km1,h= params['J'],params['tau'],params['f_km1'],params['h']
    Jd = compute_Jd(J)
    F_k   = cayley(fk)
    F_km1 = cayley(f_km1)

    res_att = (
        (1.0/h) * vee(F_k @ Jd - Jd @ F_k.T - Jd @ F_km1 + F_km1.T @ Jd)
      - h * tau
    )
    return res_att