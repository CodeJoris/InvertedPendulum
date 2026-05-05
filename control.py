import sympy as sp
import numpy as np
from sympy.physics.mechanics import dynamicsymbols
from scipy.linalg import solve_continuous_are

# --------------------------------------------------------------------------------------
# Define time and state variables
# --------------------------------------------------------------------------------------
t = sp.symbols('t')

# Define generalized coordinates that are function of time
x = sp.Function('x')(t)
theta = sp.Function('theta')(t)

# Define generalized velocities
x_dot = x.diff(t)
theta_dot = theta.diff(t)

# Define generalized accelerations
x_ddot = x_dot.diff(t)
theta_ddot = theta_dot.diff(t)

# --------------------------------------------------------------------------------------
# Define the system parameters (mass_cart, mass, length_pendulum, gravity, force_on_cart)
# --------------------------------------------------------------------------------------
M, m, l, g, u = sp.symbols('M m l g u')

# --------------------------------------------------------------------------------------
# Define the Kinematics of the system
# --------------------------------------------------------------------------------------
# Kinetic energy of the cart
T_cart = (M/2) * (x_dot**2)

# Coordinates of the mass
x_m = x - l * sp.sin(theta)
y_m = l * sp.cos(theta)

# Kinetic Energy of the mass
T_mass = (m/2) * (x_m.diff(t)**2 + y_m.diff(t)**2)

# Combine into total kinetic energy
T = T_cart + T_mass

# Potential energy
V = m*g*y_m

# Define the Lagrangian
L = T - V

# print((T+V).subs({x:200,x_dot:20,theta:0.1, theta_dot:0.02, M:1, m:0.15, l:100, g:100}))

# --------------------------------------------------------------------------------------
# Apply the Euler Lagrange Equations
# d/dt (dL/dq_dot) - dL/dq = Q (Q is the non conservative force)
# --------------------------------------------------------------------------------------
# For x coordinate, this is the coordinate on which we can apply a force `u` with motor
dL_dx_dot = sp.diff(L,x_dot) # equivalent to L.diff(x_dot)
dL_dx = sp.diff(L,x)

eq_x = sp.Eq(dL_dx_dot.diff(t) - dL_dx,u) # d/dt (dL/dx_dot) - dL/dx = u

# for theta coordinate, this is uncontrolled
dL_dtheta_dot = sp.diff(L,theta_dot)
dL_dtheta = sp.diff(L,theta)

eq_theta = sp.Eq(dL_dtheta_dot.diff(t) - dL_dtheta, 0) # d/dt (dL/dtheta_dot) - dL/dtheta = 0

# --------------------------------------------------------------------------------------
# Solve System for Accelerations
# --------------------------------------------------------------------------------------
accelerations = sp.solve([eq_x,eq_theta], (x_ddot, theta_ddot))


x_ddot_expr = accelerations[x_ddot]
theta_ddot_expr = accelerations[theta_ddot]

# Define the state vector
state = sp.Matrix([
    x,
    x_dot,
    theta,
    theta_dot
])

# Define the target vector
f = sp.Matrix([
    x_dot,              # sympy symbolic expressions: x_dot = x.diff(t)
    x_ddot_expr,        # Symboic equation from Euler-Lagrange
    theta_dot,          # sympy symbolic expressions: theta_dot = theta.diff(t)
    theta_ddot_expr     # Symboic equation from Euler-Lagrange
])

# create plain symbols
x_s, xd_s, th_s, thd_s = sp.symbols('x_s xd_s th_s thd_s')

# substitute function objects with these symbols
f_sym = f.subs({x: x_s, x_dot: xd_s, theta: th_s, theta_dot: thd_s})

state_sym = sp.Matrix([x_s, xd_s, th_s, thd_s])

A = f_sym.jacobian(state_sym)          # 4x4
B = f_sym.jacobian(sp.Matrix([u]))     # 4x1

# linearize at equilibrium
A_lin = sp.simplify(A.subs({x_s:0, xd_s:0, th_s:0, thd_s:0, u:0}))
B_lin = sp.simplify(B.subs({x_s:0, xd_s:0, th_s:0, thd_s:0, u:0}))

# Turn into numpy functions
A_func = sp.lambdify((M, m, l, g), A_lin, 'numpy')
B_func = sp.lambdify((M, m, l, g), B_lin, 'numpy')


# Define the substitutions that go into the matrices
# M = 1.0
# m = 0.15
# l = 0.5
# g = 10


def get_target_energy(m, g, l):
    # The target energy is when the pendulum is up right -> x=0, v=0, theta=0, omega=0
    return m * g * l

def get_energy(x_cart, v_cart, theta_pendulum, omega_pendulum, m_cart, m_mass, length, gravity):
    E = T + V
    # Sub the passed in values into E (it has some x.diff(t) object in there right now)
    
    # return E.subs({x:x_cart,x_dot:v_cart,theta:theta_pendulum, theta_dot:omega_pendulum, M:m_cart, m:m_mass, l:length, g:gravity})
    return 0.5 * m_mass * (length**2) * (omega_pendulum**2) + m_mass * gravity * length * np.cos(theta_pendulum)
    
# E_current = 0.5 * m * (l**2) * (theta_dot**2) + m * g * l * np.cos(theta)
# E_current = 0.5 * 0.15 * (100**2) * (0.02**2) + 0.15 * 100 * 100 * np.cos(0.1)
# print(get_energy(200,20,0.1, 0.02, 1, 0.15, 100, 100))
# print(E_current)

def get_K(M, m, l, g):
    # Evaluate the matrices with subs
    A_eval = A_func(M, m, l, g)
    B_eval = B_func(M, m, l, g)

    # Define the costs
    x_cost = 1.0            # Small cost for the cart begin away from 0
    xdot_cost = 1.0         # Small cost for the cart moving
    theta_cost = 10.0      # Big cost for the angle being far from 0
    thetadot_cost = 1.0     # Small cost for the angle changing
    u_cost = 0.1            # Very small cost for using the motor

    # Define the cost matrix for coordinates
    Q = np.diag([x_cost, xdot_cost, theta_cost, thetadot_cost])

    # Define cost matrix of the motor
    R = np.array([[u_cost]])

    # Solve the Algebraic Riccati Equation for P
    # This does the heavy lifting of: A^T*P + P*A - P*B*R^-1*B^T*P + Q = 0
    P = solve_continuous_are(A_eval,B_eval, Q, R)

    # Compute the optimal gains matrix K = R^-1 B^T P
    R_inv = np.linalg.inv(R)
    K = R_inv @ B_eval.T @ P        # all NumPy ops, shapes: (1x1)@(1x4)@(4x4) -> (1x4)

    return K

# K = get_K(M, m, l, g)
# force = - K @ state
# print(force)