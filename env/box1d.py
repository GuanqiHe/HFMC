import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sym

# Define the symbols and variables
m, g, t = sym.symbols('m g t')
x, v = sym.symbols('x v', cls=sym.Function)
x = x(t)
v = sym.diff(x, t)
k = sym.symbols('k', real=True, positive=True)

# Define the Lagrangian function
L = sym.Rational(1, 2) * m * v**2 - m * g * x - sym.Rational(1, 2) * k * x**2

# Use the Euler-Lagrange equation to obtain the differential equation of motion
dL_dx = sym.diff(L, x)
dL_dv = sym.diff(L, v)
dL_dv_dt = sym.diff(dL_dv, t)
euler_lagrange_eq = dL_dv_dt - dL_dx
print("Euler-Lagrange equation:", euler_lagrange_eq)

x_dd = sym.solve(euler_lagrange_eq, sym.diff(v, t))[0]
print("Equilibrium points:", x_dd)

x_dd = x_dd.subs([(k, 1), (m, 1), (g, 9.81)])

print(x_dd)

x_dd = sym.lambdify([t, x, v], x_dd)

# Define the function for the right-hand side of the differential equation
def box1d_dynamics(t, y):
    x, v = y
    dx_dt = v
    dv_dt = x_dd(t, x, v)
    return [dx_dt, dv_dt]

# Set the initial conditions and time span
y0 = [0, 0]
t_span = [0, 10]

# # Solve the differential equation numerically using solve_ivp
sol = solve_ivp(box1d_dynamics, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 1000))

# # Plot the result
plt.plot(sol.t, sol.y[0])
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.show()
