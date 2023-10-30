import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym

# Define the symbols and variables
t, l = sym.symbols('t l')
x, y, phi = sym.symbols('x y phi', cls=sym.Function)
x, y, phi = x(t), y(t), phi(t)
vx, vy, w = sym.diff(x, t), sym.diff(y, t), sym.diff(phi, t)
ax, ay, alpha = sym.diff(vx, t), sym.diff(vy, t), sym.diff(w, t)
f1 = x**2 + y**2 - l**2
f2 = x - l * sym.cos(phi)

# Define the Lagrangian function
T = sym.Rational(1, 2) * (m * (vx**2 + vy**2) + sym.Rational(1, 2) * m * l**2 * w**2)
L = T

# Use the Euler-Lagrange equation to obtain the differential equation of motion
dL_dx = sym.diff(L, x)
dL_dy = sym.diff(L, y)
dL_dphi = sym.diff(L, phi)
dL_dvx = sym.diff(L, vx)
dL_dvy = sym.diff(L, vy)
dL_dw = sym.diff(L, w)
dL_dvx_dt = sym.diff(dL_dvx, t)
dL_dvy_dt = sym.diff(dL_dvy, t)
dL_dw_dt = sym.diff(dL_dw, t)
l1, l2 = sym.symbols('lambda1 lambda2')

eq1 = dL_dvx_dt - dL_dx - l1 * sym.diff(f1, x) - l2 * sym.diff(f2, x)
eq2 = dL_dvy_dt - dL_dy - l1 * sym.diff(f1, y) - l2 * sym.diff(f2, y)
eq3 = dL_dw_dt - dL_dphi - l1 * sym.diff(f1, phi) - l2 * sym.diff(f2, phi)

print("Euler-Lagrange equation:", [eq1, eq2])

acc = sym.solve([eq1, eq2, eq3, sym.diff(sym.diff(f1, t), t), sym.diff(sym.diff(f2, t), t)], [ax, ay, alpha, l1, l2], dict=False)

print("Equilibrium points:", acc)

ax = acc[ax].subs([(m, 1), (g, 9.81), (l, 1)])
ay = acc[ay].subs([(m, 1), (g, 9.81), (l, 1)])
alpha = acc[alpha].subs([(m, 1), (g, 9.81), (l, 1)])
l1 = acc[l1].subs([(m, 1), (g, 9.81), (l, 1)])
l2 = acc[l2].subs([(m, 1), (g, 9.81), (l, 1)])

ax = sym.lambdify([t, x, vx, y, vy, phi, w], ax)
ay = sym.lambdify([t, x, vx, y, vy, phi, w], ay)
alpha = sym.lambdify([t, x, vx, y, vy, phi, w], alpha)
l1 = sym.lambdify([t, x, vx, y, vy, phi, w], l1)
l2 = sym.lambdify([t, x, vx, y, vy, phi, w], l2)

fx = sym.lambdify([t, x, vx, y, vy, phi, w], l1*sym.diff(f1, x) + l2*sym.diff(f2, x))
fy = sym.lambdify([t, x, vx, y, vy, phi, w], l1*sym.diff(f1, y) + l2*sym.diff(f2, y))
fphi = sym.lambdify([t, x, vx, y, vy, phi, w], l1*sym.diff(f1, phi) + l2*sym.diff(f2, phi))

# Define the function for the right-hand side of the differential equation
def uav2d_constrain_dynamics(t, y):
    x, vx, y, vy, phi, w = y
    dx_dt = vx
    dy_dt = vy
    dphi_dt = w
    dvx_dt = ax(t, x, vx, y, vy, phi, w)
    dvy_dt = ay(t, x, vx, y, vy, phi, w)
    dw_dt = alpha(t, x, vx, y, vy, phi, w)
    return [dx_dt, dvx_dt, dy_dt, dvy_dt, dphi_dt, dw_dt]

# Set the initial conditions and time span
y0 = [0, 1, 0, 0, 0, 0]
t_span = [0, 10]

# Solve the differential equation numerically using solve_ivp
sol = solve_ivp(uav2d_constrain_dynamics, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000))

