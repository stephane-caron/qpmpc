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

"""Humanoid planning to walk a single step ahead."""

from dataclasses import dataclass

import numpy as np
import pylab

from ltv_mpc import MPCProblem, solve_mpc


@dataclass
class Parameters:
    """Parameters of the step and humanoid."""

    com_height: float = 0.8
    dsp_duration: float = 0.1  # [s]
    end_pos: float = 0.3  # [m]
    foot_length: float = 0.1  # [m]
    gravity: float = 9.81  # [m] / [s]²
    horizon_duration: float = 2.5  # [s]
    nb_timesteps: int = 16
    ssp_duration: float = 0.7  # [s]
    start_pos: float = 0.0  # [m]


def build_mpc_problem(params: Parameters):
    """Build the model predictive control problem.

    Args:
        params: Problem parameters.

    For details on this problem and how model predictive control can be used
    for humanoid stepping, see "Trajectory free linear model predictive control
    for stable walking in the presence of strong perturbations" (Wieber, 2006).
    """
    T = params.horizon_duration / params.nb_timesteps
    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))
    state_matrix = np.array(
        [[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
    )
    input_matrix = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T])
    input_matrix = input_matrix.reshape((3, 1))
    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    ineq_matrix = np.array([+zmp_from_state, -zmp_from_state])
    cur_max = params.start_pos + 0.5 * params.foot_length
    cur_min = params.start_pos - 0.5 * params.foot_length
    next_max = params.end_pos + 0.5 * params.foot_length
    next_min = params.end_pos - 0.5 * params.foot_length
    ineq_vector = [
        np.array([+1000.0, +1000.0])
        if i < nb_init_dsp_steps
        else np.array([+cur_max, -cur_min])
        if i - nb_init_dsp_steps <= nb_init_ssp_steps
        else np.array([+1000.0, +1000.0])
        if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps
        else np.array([+next_max, -next_min])
        for i in range(params.nb_timesteps)
    ]
    return MPCProblem(
        transition_state_matrix=state_matrix,
        transition_input_matrix=input_matrix,
        ineq_state_matrix=ineq_matrix,
        ineq_input_matrix=None,
        ineq_vector=ineq_vector,
        initial_state=np.array([params.start_pos, 0.0, 0.0]),
        goal_state=np.array([params.end_pos, 0.0, 0.0]),
        nb_timesteps=params.nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-3,
    )


def plot_plan(params, mpc_problem, plan):
    """Plot plan resulting from the MPC problem.

    Args:
        params: Parameters of the problem.
        mpc_problem: Model predictive control problem.
        plan: Solution to the MPC problem.
    """
    t = np.linspace(0.0, params.horizon_duration, params.nb_timesteps + 1)
    X = plan.states
    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    zmp = X.dot(zmp_from_state)
    pos = X[:, 0]
    zmp_min = [
        x[0] if abs(x[0]) < 10 else None for x in mpc_problem.ineq_vector
    ]
    zmp_max = [
        -x[1] if abs(x[1]) < 10 else None for x in mpc_problem.ineq_vector
    ]
    zmp_min.append(zmp_min[-1])
    zmp_max.append(zmp_max[-1])
    pylab.ion()
    pylab.clf()
    pylab.plot(t, pos)
    pylab.plot(t, zmp, "r-")
    pylab.plot(t, zmp_min, "g:")
    pylab.plot(t, zmp_max, "b:")
    pylab.grid(True)
    pylab.show(block=True)


if __name__ == "__main__":
    params = Parameters()
    mpc_problem = build_mpc_problem(params)
    plan = solve_mpc(mpc_problem, solver="quadprog")
    plot_plan(params, mpc_problem, plan)
