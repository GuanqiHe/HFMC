import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym
from dataclasses import dataclass, field

# Define the symbols and variables
t, r = sym.symbols('t r')
x, y = sym.symbols('x y', cls=sym.Function)
x, y = x(t), y(t)
vx = sym.diff(x, t)
vy = sym.diff(y, t)
ax = sym.diff(vx, t)
ay = sym.diff(vy, t)
f = x**2 + y**2 - r**2
Qx = sym.Symbol('Qx')
Qy = sym.Symbol('Qy')

# Define the Lagrangian function
L = sym.Rational(1, 2) * (vx**2 + vy**2)

# Use the Euler-Lagrange equation to obtain the differential equation of motion
dL_dx = sym.diff(L, x)
dL_dy = sym.diff(L, y)
dL_dvx = sym.diff(L, vx)
dL_dvy = sym.diff(L, vy)
dL_dvx_dt = sym.diff(dL_dvx, t)
dL_dvy_dt = sym.diff(dL_dvy, t)
l = sym.symbols('lambda')

eq1 = dL_dvx_dt - dL_dx - l * sym.diff(f, x) - Qx
eq2 = dL_dvy_dt - dL_dy - l * sym.diff(f, y) - Qy

print("Euler-Lagrange equation:", [eq1, eq2])


acc = sym.solve([eq1, eq2, sym.diff(sym.diff(f, t), t)], [ax, ay, l], dict=False)
print("Equilibrium points:", acc)

ax = acc[ax].subs([(r, 1)])
ay = acc[ay].subs([(r, 1)])
l = acc[l].subs([(r, 1)])
f = f.subs([(r, 1)])

ax = sym.lambdify([t, x, vx, y, vy, Qx, Qy], ax)
ay = sym.lambdify([t, x, vx, y, vy, Qx, Qy], ay)

fx = sym.lambdify([t, x, vx, y, vy, Qx, Qy], l*sym.diff(f, x))
fy = sym.lambdify([t, x, vx, y, vy, Qx, Qy], l*sym.diff(f, y))
lam = sym.lambdify([t, x, vx, y, vy, Qx, Qy], l)

# Define the function for the right-hand side of the differential equation

@dataclass
class ctrl_info_t:
    int_x: float = 0
    int_y: float = 0
    state: list = field(default_factory=list) # [t, x, vx, y, vy, l, fx, fy]


def box1d_dynamics(t, y, ctrl_info: ctrl_info_t):
    x, vx, y, vy = y
    dx_dt = vx
    dy_dt = vy
    # Qx = -x - vx - 0.1 * ctrl_info.int_x
    # Qy = -y - vy - 0.1 * ctrl_info.int_y
    Qx = -x - vx - 0.05 * ctrl_info.int_x
    Qy = -y - vy - 0.05 * ctrl_info.int_y
    dvx_dt = ax(t, x, vx, y, vy, Qx, Qy)
    dvy_dt = ay(t, x, vx, y, vy, Qx, Qy)
    l = lam(t, x, vx, y, vy, Qx, Qy)
    fx_ = fx(t, x, vx, y, vy, Qx, Qy)
    fy_ = fy(t, x, vx, y, vy, Qx, Qy)
    ctrl_info.int_x += 0.0 -x
    ctrl_info.int_y += 1.0 -y
    ctrl_info.state.append([t, x, vx, y, vy, l, fx_, fy_])
    return [dx_dt, dvx_dt, dy_dt, dvy_dt]

# Set the initial conditions and time span
y0 = [0, 1, 1, 0]
t_span = [0, 20]

# # Solve the differential equation numerically using solve_ivp
ctrl_info = ctrl_info_t()

sol = solve_ivp(lambda t, y: box1d_dynamics(t, y, ctrl_info), t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000))

state = np.array(ctrl_info.state)
print(state.shape)
t = state[:, 0]
fx = state[:, 5]
fy = state[:, 6]
fx = np.interp(sol.t, t, fx)
fy = np.interp(sol.t, t, fy)


# # Plot the result
plt.figure()
plt.plot(sol.t, sol.y[0], label='x')
plt.plot(sol.t, sol.y[2], label='y')
plt.plot(sol.t, np.sqrt(sol.y[0]**2 + sol.y[2]**2), label='r')
plt.plot(sol.t, sol.y[1], label='vx')
plt.plot(sol.t, sol.y[3], label='vy')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('X (m)')

plt.figure()
plt.plot(sol.t, fx, label='fx')
plt.plot(sol.t, fy, label='fy')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('N (m)')

plt.show()
