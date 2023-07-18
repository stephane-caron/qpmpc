#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triple integrator LTV system."""

import numpy as np
import pylab

from ltv_mpc import MPCProblem, solve_mpc

if __name__ == "__main__":
    horizon_duration = 1.0
    nb_timesteps = 16
    T = horizon_duration / nb_timesteps
    A = np.array([[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]])
    B = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T]).reshape((3, 1))

    # Acceleration limits
    accel_from_state = np.array([0.0, 0.0, 1.0])
    max_accel = 3.0  # [m] / [s] / [s]
    ineq_matrix = np.vstack([+accel_from_state, -accel_from_state])
    ineq_vector = np.array([+max_accel, +max_accel])

    # Complete LTV problem
    initial_pos = 0.0
    goal_pos = 1.0
    problem = MPCProblem(
        transition_state_matrix=A,
        transition_input_matrix=B,
        ineq_state_matrix=ineq_matrix,
        ineq_input_matrix=None,
        ineq_vector=ineq_vector,
        initial_state=np.array([initial_pos, 0.0, 0.0]),
        goal_state=np.array([goal_pos, 0.0, 0.0]),
        nb_timesteps=nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-6,
    )

    # Solve our LTV problem
    plan = solve_mpc(problem, solver="osqp")

    # Plot solution
    t = np.linspace(0.0, horizon_duration, nb_timesteps + 1)
    X = plan.states
    positions, velocities, accelerations = X[:, 0], X[:, 1], X[:, 2]
    pylab.ion()
    pylab.clf()
    pylab.plot(t, positions)
    pylab.plot(t, velocities)
    pylab.plot(t, accelerations)
    pylab.grid(True)
    pylab.legend(("position", "velocity", "acceleration"))
