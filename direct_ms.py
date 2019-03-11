#
#     This file is part of CasADi.
#
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
from casadi import *

T = 10. # Time horizon
N = 20 # number of control intervals

# Declare model variables
x1 = MX.sym('x1')
x2 = MX.sym('x2')
x = vertcat(x1, x2)
u = MX.sym('u')

# Model equations
xdot = vertcat((1-x2**2)*x1 - x2 + u, x1)

# Objective term
L = x1**2 + x2**2 + u**2

# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [xdot, L])
X0 = MX.sym('X0', 2)
U = MX.sym('U')
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + DT/2 * k1, U)
    k3, k3_q = f(X + DT/2 * k2, U)
    k4, k4_q = f(X + DT * k3, U)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [X0, U], [X, Q],['x0','p'],['x_end','cost_accum'])

# Start with an empty NLP
decision_vars = []
X0 = []
decision_vars_lower_bounds = []
decision_vars_upper_bounds = []
running_cost = 0
decision_vars_constraints_must_equal_zero = []
constraint_vars_lower_bound = []
constraint_vars_upper_bound = []

# UP TO HERE SAME SHIT AS SINGLE SHOOT

# adds init. condition to decision vars (?)
# "Lift" initial conditions
Xk = MX.sym('X0', 2)
decision_vars += [Xk]
decision_vars_lower_bounds += [0, 1]
decision_vars_upper_bounds += [0, 1]
X0 += [0, 1]

# decision_vars_lower_bounds, decision_vars_upper_bounds constraints around decision vars
# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    decision_vars   += [Uk]
    decision_vars_lower_bounds += [-1]
    decision_vars_upper_bounds += [1]
    X0  += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk_end = Fk['x_end']
    running_cost += Fk['cost_accum']

    # New NLP variable for state at end of interval
    # pick a new "initial state" every time (???)
    Xk = MX.sym('X_' + str(k+1), 2)
    decision_vars  += [Xk]
    decision_vars_lower_bounds += [-0.25, -inf]
    decision_vars_upper_bounds += [  inf,  inf]
    X0  += [0, 0]

    # Add equality constraint
    decision_vars_constraints_must_equal_zero += [Xk_end-Xk]

    # dis me wtf this do?
    # constraint_vars_lower_bound += [0, 0]
    # constraint_vars_upper_bound += [0, 0]

# Create an NLP solver
prob = {'f': running_cost, 'x': vertcat(*decision_vars), 'g': vertcat(*decision_vars_constraints_must_equal_zero)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=X0, lbx=decision_vars_lower_bounds, ubx=decision_vars_upper_bounds, lbg=constraint_vars_lower_bound, ubg=constraint_vars_upper_bound)
w_opt = sol['x'].full().flatten()

print(w_opt[2::3])

# # Plot the solution
# x1_opt = w_opt[0::3]
# x2_opt = w_opt[1::3]
# u_opt = w_opt[2::3]

# tgrid = [T/N*k for k in range(N+1)]
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x1_opt, '--')
# plt.plot(tgrid, x2_opt, '-')
# plt.step(tgrid, vertcat(DM.nan(1), u_opt), '-.')
# plt.xlabel('t')
# plt.legend(['x1','x2','u'])
# plt.grid()
# plt.show()
