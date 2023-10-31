import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import sympy as sym
import matplotlib.pyplot as plt

m,g,t = sym.symbols('m g t')
x,v = sym.symbols('x v', cls=sym.Function)
x = x(t)
v = sym.diff(x,t)
k = sym.symbols('k', real=True, positive=True)

# Define the Lagrangian function
L = sym.Rational(1,2)*m*v**2 - m*g*x - sym.Rational(1,2)*k*x**2

# Use the Euler-Lagrange equation to obtain the differential equation of motion
dL_dx = sym.diff(L,x)
dL_dv = sym.diff(L,v)
dL_dv_dt = sym.diff(dL_dv,t)
euler_lagrange_eq = dL_dv_dt - dL_dx
print("Euler-Lagrange equation:", euler_lagrange_eq)

x_dd = sym.solve(euler_lagrange_eq, sym.diff(v,t))[0]
print("Equilibrium points:", x_dd)

x_dd = x_dd.subs([(k,1), (m,1), (g,9.81)])

print(x_dd)

x_dd = sym.lambdify([t,x,v], x_dd, 'numpy')

# Define the function for the right-hand side of the differential equation
def box1d_dynamics_residual(t, y, yd):
    x, v = y
    dx, dv = yd
    res0 = dx - v
    res1 = dv - x_dd(t, x, v)
    return np.array([res0, res1])

# Set the initial conditions and time span
y0 = [0, 0]
dy0 = [0, 0]
t0 = 0.0

model = Implicit_Problem(box1d_dynamics_residual, y0, dy0, t0)
model.name = 'Box1D'

sim = IDA(model)

tfinal = 10
ncp = 500

t, y, yd = sim.simulate(tfinal, ncp)

# Plot the result
plt.plot(t, y[:,0], label='Position')
plt.plot(t, y[:,1], label='Velocity')
plt.legend()
plt.ylabel('Position (m) / Velocity (m/s)')
plt.xlabel('Time (s)')
plt.show()