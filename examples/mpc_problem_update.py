# SPDX-License-Identifier: Apache-2.0
#
# /// script
# dependencies = ["matplotlib", "daqp", "qpmpc"]
# ///

"""Solving an updated MPC problem and comparing computation times.

The system is a double integrator, i.e. a unit-mass point moving in a line,
controlled by a force (equivalently, an acceleration) with bounded magnitude.
Here the bound varies over the horizon: it is 1 m/s² during the first and last
thirds, and 2 m/s² in the middle third. There are no state constraints.

The goal is the farthest position reachable from rest to rest in the horizon,
so that the only solution is bang-bang: full acceleration up to mid-horizon,
then full deceleration.

After a first solve, we shift both the initial and goal positions and update
the QP in place, as one would do in a receding-horizon loop.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from qpsolvers import solve_problem

from qpmpc import MPCQP, MPCProblem, Plan

if __name__ == "__main__":
    horizon_duration = 3.0  # [s]
    nb_timesteps = 18
    T = horizon_duration / nb_timesteps
    A = np.array([[1.0, T], [0.0, 1.0]])
    B = np.array([T**2 / 2.0, T]).reshape((2, 1))

    # Input (acceleration) limits: -max_input[k] <= u[k] <= +max_input[k]
    third = nb_timesteps // 3
    max_input = [1.0] * third + [2.0] * third + [1.0] * third  # [m] / [s]²
    ineq_matrix = np.vstack([+np.eye(1), -np.eye(1)])
    ineq_vector = [np.array([+u_max, +u_max]) for u_max in max_input]

    # Bang-bang input profile [+1, +2, -2, -1] travels 2.5 m from rest to rest
    initial_pos = 0.0
    goal_pos = 2.5
    problem = MPCProblem(
        transition_state_matrix=A,
        transition_input_matrix=B,
        ineq_state_matrix=None,
        ineq_input_matrix=ineq_matrix,
        ineq_vector=ineq_vector,
        initial_state=np.array([initial_pos, 0.0]),
        goal_state=np.array([goal_pos, 0.0]),
        nb_timesteps=nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-8,
    )

    # Build the QP once and solve it
    t0 = perf_counter()
    mpc_qp = MPCQP(problem)
    plan = Plan(problem, solve_problem(mpc_qp.problem, solver="daqp"))
    t1 = perf_counter()
    print(f"Duration of initial solve: {1000.0 * (t1 - t0):.0} ms")

    # Shift the problem by one meter and update the QP in place
    offset = 1.0
    t0 = perf_counter()
    problem.update_initial_state(np.array([initial_pos + offset, 0.0]))
    problem.update_goal_state(np.array([goal_pos + offset, 0.0]))
    mpc_qp.update_cost_vector(problem)
    mpc_qp.update_constraint_vector(problem)
    shifted_plan = Plan(problem, solve_problem(mpc_qp.problem, solver="daqp"))
    t1 = perf_counter()
    print(f"Duration of updated solve: {1000.0 * (t1 - t0):.0} ms")

    # Input bounds do not depend on the state, so the solution is the same
    initial_inputs = plan.inputs.flatten()
    shifted_inputs = shifted_plan.inputs.flatten()
    assert np.allclose(initial_inputs, shifted_inputs)

    # Plot solution
    t = np.linspace(0.0, horizon_duration, nb_timesteps + 1)
    X = shifted_plan.states
    positions, velocities = X[:, 0], X[:, 1]
    plt.ion()
    plt.clf()
    plt.plot(t, positions)
    plt.plot(t, velocities)
    plt.step(t[:-1], shifted_plan.inputs, where="post")
    plt.step(t[:-1], max_input, "k--", where="post", linewidth=1)
    plt.step(t[:-1], [-u for u in max_input], "k--", where="post", linewidth=1)
    plt.grid(True)
    plt.legend(
        ("position", "velocity", "input (acceleration)", "input bounds")
    )
    plt.show(block=True)
