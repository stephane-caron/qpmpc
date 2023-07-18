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

"""Model predictive control part of the lipm_walking_controller.

See also: https://github.com/stephane-caron/lipm_walking_controller/
"""

from dataclasses import dataclass

import numpy as np
import pylab

from ltv_mpc import MPCProblem, solve_mpc


@dataclass
class Parameters:
    """Parameters of the step and humanoid.

    These parameters are taken from the lipm_walking_controller.
    """

    com_height: float = 0.84  # [m]
    dsp_duration: float = 0.1  # [s]
    foot_size: float = 0.065  # [m]
    gravity: float = 9.81  # [m] / [s]²
    nb_timesteps: int = 16
    sampling_period: float = 0.1  # [s]
    ssp_duration: float = 0.7  # [s]


def build_mpc_problem(params: Parameters, start_pos: float, end_pos: float):
    """Build the model predictive control problem.

    Args:
        params: Problem parameters.

    For details on this problem and how open-loop model predictive control was
    used in the lipm_walking_controller, see "Stair Climbing Stabilization of
    the HRP-4 Humanoid Robot using Whole-body Admittance Control" (Caron et
    al., 2019).
    """
    T = params.sampling_period
    state_matrix = np.array(
        [[1.0, T, T ** 2 / 2.0], [0.0, 1.0, T], [0.0, 0.0, 1.0]]
    )
    input_matrix = np.array([T ** 3 / 6.0, T ** 2 / 2.0, T]).reshape((3, 1))

    nb_init_dsp_steps = int(round(params.dsp_duration / T))
    nb_init_ssp_steps = int(round(params.ssp_duration / T))
    nb_dsp_steps = int(round(params.dsp_duration / T))

    eta = params.com_height / params.gravity
    zmp_from_state = np.array([1.0, 0.0, -eta])
    ineq_matrix = np.array([+zmp_from_state, -zmp_from_state])

    cur_max = start_pos + 0.5 * params.foot_size
    cur_min = start_pos - 0.5 * params.foot_size
    next_max = end_pos + 0.5 * params.foot_size
    next_min = end_pos - 0.5 * params.foot_size
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
        initial_state=None,
        goal_state=np.array([end_pos, 0.0, 0.0]),
        nb_timesteps=params.nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-3,
    )


def integrate(state: np.ndarray, jerk: float, dt: float) -> np.ndarray:
    """Integrate state (pos, vel, accel) with constant jerk.

    Args:
        state: Initial state.
        jerk: Constant jerk to integrate.
        dt: Duration to integrate for, in seconds.

    Returns:
        State after integration.
    """
    p_0, v_0, a_0 = state
    return np.array(
        [
            p_0 + dt * (v_0 + dt * (a_0 / 2 + dt * jerk / 6)),
            v_0 + dt * (a_0 + dt * (jerk / 2)),
            a_0 + dt * jerk,
        ]
    )

def plot_plan(params, mpc_problem, plan, state: np.ndarray) -> None:
    """Plot plan resulting from the MPC problem.

    Args:
        params: Parameters of the problem.
        mpc_problem: Model predictive control problem.
        plan: Solution to the MPC problem.
        state: Additional state to plot.
    """
    horizon_duration = params.sampling_period * params.nb_timesteps
    t = np.linspace(0.0, horizon_duration, params.nb_timesteps + 1)
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
    pylab.plot([0, 0.1], [state[0], state[0] + 0.1 * state[1]], "ro", lw=2)
    pylab.plot(t, zmp, "r-")
    pylab.plot(t, zmp_min, "g:")
    pylab.plot(t, zmp_max, "b:")
    pylab.grid(True)
    pylab.show(block=False)



if __name__ == "__main__":
    params = Parameters()
    mpc_problem = build_mpc_problem(params, start_pos=0, end_pos=1)
    state = np.array([0.0, 0.0, 0.0])
    T = params.sampling_period
    substeps: int = 20  # number of integration substeps
    dt = T / substeps
    for _ in range(10):
        mpc_problem.set_initial_state(state)
        plan = solve_mpc(mpc_problem, solver="quadprog")
        for step in range(substeps):
            state = integrate(state, plan.inputs[0], dt)
            # plot_plan(params, mpc_problem, plan, state)
            print(f"{state=}")
