
import sympy as sp

# Variables
s, t = sp.symbols('s t', real=True)
s_p, L = sp.symbols('sp L', real=True)
m_d , mu ,kappa ,m_v,g , E , A= sp.symbols('m_d mu kappa m_v g E A ', real=True)


s_p = sp.Function('s_p')(t)
# Vector functions r0(t), r1(t) in 3D (change dimension as needed)
r0x, r0y, r0z = sp.Function('r0x')(t), sp.Function('r0y')(t), sp.Function('r0z')(t)
r1x, r1y, r1z = sp.Function('r1x')(t), sp.Function('r1y')(t), sp.Function('r1z')(t)

r0 = sp.Matrix([r0x, r0y, r0z])
r1 = sp.Matrix([r1x, r1y, r1z])
l = L - s_p
# r(s,t)
r = (1 - s/l)*r0 + (s/l)*r1

rdot = sp.diff(r, t)
norm_sq = (rdot.T * rdot)[0]

Ke_string = sp.Rational(1, 2) *mu* sp.integrate(norm_sq, (s, s_p, L))
r1_dot = sp.diff(r1,t)
Ke_drone = sp.Rational(1, 2) * m_d*(r1_dot.T * r1_dot)[0]
sp_dot = sp.diff(s_p,t)
Ke_winch = sp.Rational(1, 2) * (mu*s_p + kappa ) * sp_dot**2

r0_dot = sp.diff(r0,t)
Ke_vehicle = sp.Rational(1, 2) * m_v* (r0_dot.T * r0_dot)[0]

V_drone = -m_d * g * r1z

V_vehicle = -(m_v + mu * s_p)* g * r0z   ##included everything expect tether on winch
drds = sp.diff(r,s)

lam = sp.sqrt((drds.T * drds)[0])
# elastic energy density
integrand = sp.Rational(1,2) * E * A * (lam - 1)**2

# integrate
V_string = sp.integrate(integrand, (s, s_p, L))

sp.simplify(V_string)
KE = Ke_drone + Ke_string + Ke_vehicle + Ke_winch
PE = V_drone + V_vehicle + V_string


# --- your q vector ---
q = sp.Matrix([r0x, r0y, r0z, r1x, r1y, r1z, s_p])
qd  = q.diff(t)
qdd = qd.diff(t)

Lag = KE - PE

def EL_eq(L, qi):
    return sp.diff(sp.diff(L, sp.diff(qi, t)), t) - sp.diff(L, qi)

# Euler–Lagrange residuals: EL(q,qd,qdd)=0
EL = sp.Matrix([EL_eq(Lag, qi) for qi in q])

# ---- IMPORTANT: replace Derivative(qi(t),t) and Derivative(qi(t),t,2) by symbols
qd_syms  = sp.Matrix(sp.symbols('qd0:%d'  % len(q)))
qdd_syms = sp.Matrix(sp.symbols('qdd0:%d' % len(q)))

subs_map = {}
for i in range(len(q)):
    subs_map[sp.diff(q[i], t)]      = qd_syms[i]
    subs_map[sp.diff(q[i], t, 2)]   = qdd_syms[i]

EL_sub = sp.Matrix([sp.simplify(expr.subs(subs_map)) for expr in EL])

# Now extract: M*qdd + h = 0
M, h = sp.linear_eq_to_matrix(EL_sub, list(qdd_syms))
print(h)
