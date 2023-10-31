import numpy as np
from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA
import sympy as sym
import matplotlib.pyplot as plt

# Define the symbols and variables
t, r = sym.symbols('t r')
x, y = sym.symbols('x y', cls=sym.Function)
x, y = x(t), y(t)
vx = sym.diff(x, t)
vy = sym.diff(y, t)
ax = sym.diff(vx, t)
ay = sym.diff(vy, t)
f = x**2 + y**2 - r**2

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

eq1 = dL_dvx_dt - dL_dx - l * sym.diff(f, x)
eq2 = dL_dvy_dt - dL_dy - l * sym.diff(f, y)

print("Euler-Lagrange equation:", [eq1, eq2])


acc = sym.solve([eq1, eq2, sym.diff(sym.diff(f, t), t)], [ax, ay, l], dict=False)
print("Equilibrium points:", acc)

ax = acc[ax].subs([(r, 1)])
ay = acc[ay].subs([(r, 1)])
l = acc[l].subs([(r, 1)])
f = f.subs([(r, 1)])

ax = sym.lambdify([t, x, vx, y, vy], ax, 'numpy')
ay = sym.lambdify([t, x, vx, y, vy], ay, 'numpy')

fx = sym.lambdify([t, x, vx, y, vy], l*sym.diff(f, x), 'numpy')
fy = sym.lambdify([t, x, vx, y, vy], l*sym.diff(f, y), 'numpy')

l = sym.lambdify([t, x, vx, y, vy], l, 'numpy')

f = sym.lambdify([t, x, vx, y, vy], f, 'numpy')

def box2d_circle_residual(t, y, yd):
    x, vx, y, vy = y
    dx, dvx, dy, dvy = yd
    res0 = dx - vx
    res1 = dvx - ax(t, x, vx, y, vy)
    res2 = dy - vy
    res3 = dvy - ay(t, x, vx, y, vy)
    res4 = f(t, x, vx, y, vy)
    return np.array([res0, res1, res2, res3, res4])

y0 = [1.0, 0.0, 0.0, 1.0]
dy0 = [0.0, 0.0, 1.0, 0.0]
t0 = 0.0

model = Implicit_Problem(box2d_circle_residual, y0, dy0, t0)
model.name = 'Box2D_Circle'

sim = IDA(model)

tfinal = 10
ncp = 500

t, y, yd = sim.simulate(tfinal, ncp)

# Plot the result
plt.figure()
plt.plot(t, y[:,0], label='x')
plt.plot(t, y[:,1], label='vx')
plt.plot(t, y[:,2], label='y')
plt.plot(t, y[:,3], label='vy')
plt.legend()
plt.ylabel('Position (m) / Velocity (m/s)')
plt.xlabel('Time (s)')

plt.figure()
plt.plot(t, fx(t, y[:,0], y[:,1], y[:,2], y[:,3]), label='fx')
plt.plot(t, fy(t, y[:,0], y[:,1], y[:,2], y[:,3]), label='fy')
plt.legend()
plt.ylabel('Force (N)')
plt.xlabel('Time (s)')
plt.show()
