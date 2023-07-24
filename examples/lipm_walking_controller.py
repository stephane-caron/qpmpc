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

"""Model predictive control part of the LIPM walking controller.

See also: https://github.com/stephane-caron/lipm_walking_controller/
"""

import argparse
from dataclasses import dataclass

import numpy as np
from loop_rate_limiters import RateLimiter

from ltv_mpc import MPCProblem, solve_mpc
from ltv_mpc.exceptions import ProblemDefinitionError
from ltv_mpc.utils import LivePlot

MAX_ZMP_DIST = 100.0  # [m]


@dataclass
class Parameters:
    """Parameters of the step and humanoid.

    These parameters are taken from the lipm_walking_controller.
    """

    com_height: float = 0.84  # [m]
    dsp_duration: float = 0.1  # [s]
    foot_size: float = 0.065  # [m]
    gravity: float = 9.81  # [m] / [s]²
    init_support_foot_pos: float = 0.09  # [m]
    nb_timesteps: int = 16
    sampling_period: float = 0.1  # [s]
    ssp_duration: float = 0.7  # [s]
    strides = [-0.18, 0.18]  # [m]

    @property
    def omega(self) -> float:
        return np.sqrt(self.gravity / self.com_height)

    @property
    def dcm_from_state(self) -> np.ndarray:
        return np.array([1.0, 1.0 / self.omega, 0.0])

    @property
    def zmp_from_state(self) -> np.ndarray:
        return np.array([1.0, 0.0, -1.0 / self.omega ** 2])


def build_mpc_problem(params: Parameters):
    """Build the model predictive control problem.

    Args:
        params: Problem parameters.

    For details on this problem and how open-loop model predictive control was
    used in the LIPM walking controller, see "Stair Climbing Stabilization of
    the HRP-4 Humanoid Robot using Whole-body Admittance Control" (Caron et
    al., 2019).
    """
    T = params.sampling_period
    state_matrix = np.array(
        [
            [1.0, T, T ** 2 / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ]
    )
    input_matrix = np.array(
        [
            T ** 3 / 6.0,
            T ** 2 / 2.0,
            T,
        ]
    ).reshape((3, 1))
    ineq_matrix = np.array([+params.zmp_from_state, -params.zmp_from_state])
    return MPCProblem(
        transition_state_matrix=state_matrix,
        transition_input_matrix=input_matrix,
        ineq_state_matrix=ineq_matrix,
        ineq_input_matrix=None,
        ineq_vector=None,
        initial_state=None,
        goal_state=None,
        nb_timesteps=params.nb_timesteps,
        terminal_cost_weight=1.0,
        stage_state_cost_weight=None,
        stage_input_cost_weight=1e-3,
    )


class PhaseStepper:
    def __init__(self, params):
        nb_dsp_steps = int(round(params.dsp_duration / T))
        nb_ssp_steps = int(round(params.ssp_duration / T))
        if 2 * (nb_dsp_steps + nb_ssp_steps) < params.nb_timesteps:
            raise ProblemDefinitionError(
                "there are more than two steps in the receding horizon"
            )

        # Skip edge cases of separate initial and final DSP durations (1/2):
        # here we set the initial index in the middle of the first SSP phase in
        # the receding horizon, thus creating a short first SSP phase.
        initial_index = 5

        self.index = initial_index
        self.nb_dsp_steps = nb_dsp_steps
        self.nb_ssp_steps = nb_ssp_steps
        self.params = params
        self.stride_index = 0

    def advance(self):
        self.index += 1
        if self.index >= self.nb_dsp_steps + self.nb_ssp_steps:
            self.index = 0

    def advance_stride(self):
        self.stride_index = (self.stride_index + 1) % len(self.params.strides)

    def get_nb_steps(self):
        offset = self.index
        nb_init_dsp_steps = max(0, self.nb_dsp_steps - offset)
        offset = max(0, offset - self.nb_dsp_steps)
        nb_init_ssp_steps = max(0, self.nb_ssp_steps - offset)
        offset = max(0, offset - self.nb_ssp_steps)

        remaining = (
            self.params.nb_timesteps - nb_init_dsp_steps - nb_init_ssp_steps
        )
        nb_next_dsp_steps = min(self.nb_dsp_steps, remaining)
        remaining = max(0, remaining - self.nb_dsp_steps)
        nb_next_ssp_steps = min(self.nb_ssp_steps, remaining)
        remaining = max(0, remaining - self.nb_ssp_steps)
        nb_last_dsp_steps = min(self.nb_dsp_steps, remaining)
        remaining = max(0, remaining - self.nb_dsp_steps)
        nb_last_ssp_steps = min(self.nb_ssp_steps, remaining)
        remaining = max(0, remaining - self.nb_ssp_steps)
        if remaining > 0:
            raise ProblemDefinitionError(
                "there are more than two steps in the receding horizon"
            )

        return (
            nb_init_dsp_steps,
            nb_init_ssp_steps,
            nb_next_dsp_steps,
            nb_next_ssp_steps,
            nb_last_dsp_steps,
            nb_last_ssp_steps,
        )

    def get_next_foot_pos(self, foot_pos: float) -> float:
        return foot_pos + self.params.strides[self.stride_index]

    def get_last_foot_pos(self, foot_pos: float) -> float:
        upcoming_stride = (self.stride_index + 1) % len(self.params.strides)
        return (
            self.get_next_foot_pos(foot_pos)
            + self.params.strides[upcoming_stride]
        )


def update_goal_and_constraints(
    mpc_problem: MPCProblem,
    phase: PhaseStepper,
    cur_foot_pos: float,
):
    (
        nb_init_dsp_steps,
        nb_init_ssp_steps,
        nb_next_dsp_steps,
        nb_next_ssp_steps,
        nb_last_dsp_steps,
        nb_last_ssp_steps,
    ) = phase.get_nb_steps()
    next_foot_pos = phase.get_next_foot_pos(cur_foot_pos)
    last_foot_pos = phase.get_last_foot_pos(cur_foot_pos)
    cur_max = cur_foot_pos + 0.5 * params.foot_size
    cur_min = cur_foot_pos - 0.5 * params.foot_size
    next_max = next_foot_pos + 0.5 * params.foot_size
    next_min = next_foot_pos - 0.5 * params.foot_size
    last_max = last_foot_pos + 0.5 * params.foot_size
    last_min = last_foot_pos - 0.5 * params.foot_size
    mpc_problem.ineq_vector = (
        [np.array([+MAX_ZMP_DIST, +MAX_ZMP_DIST])] * nb_init_dsp_steps
        + [np.array([+cur_max, -cur_min])] * nb_init_ssp_steps
        + [np.array([+MAX_ZMP_DIST, +MAX_ZMP_DIST])] * nb_next_dsp_steps
        + [np.array([+next_max, -next_min])] * nb_next_ssp_steps
        + [np.array([+MAX_ZMP_DIST, +MAX_ZMP_DIST])] * nb_last_dsp_steps
        + [np.array([+last_max, -last_min])] * nb_last_ssp_steps
    )
    goal_pos = last_foot_pos if nb_last_dsp_steps > 0 else next_foot_pos
    mpc_problem.update_goal_state(np.array([goal_pos, 0.0, 0.0]))


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
    ).flatten()


def plot_plan(t, live_plot, params, mpc_problem, plan) -> None:
    """Plot plan resulting from the MPC problem.

    Args:
        params: Parameters of the problem.
        mpc_problem: Model predictive control problem.
        plan: Solution to the MPC problem.
        state: Additional state to plot.
    """
    horizon_duration = params.sampling_period * params.nb_timesteps
    trange = np.linspace(t, t + horizon_duration, params.nb_timesteps + 1)
    X = plan.states
    zmp = X.dot(params.zmp_from_state)
    pos = X[:, 0]
    zmp_min = [
        x[0] if abs(x[0]) < MAX_ZMP_DIST else None
        for x in mpc_problem.ineq_vector
    ]
    zmp_max = [
        -x[1] if abs(x[1]) < MAX_ZMP_DIST else None
        for x in mpc_problem.ineq_vector
    ]
    zmp_min.append(zmp_min[-1])
    zmp_max.append(zmp_max[-1])
    live_plot.update_line("pos", trange, pos)
    live_plot.update_line("zmp", trange, zmp)
    live_plot.update_line("zmp_min", trange, zmp_min)
    live_plot.update_line("zmp_max", trange, zmp_max)


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slowdown",
        help="Slow time down by a multiplicative factor",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_command_line_arguments()
    params = Parameters()
    mpc_problem = build_mpc_problem(params)
    T = params.sampling_period
    substeps: int = 15  # number of integration substeps
    dt = T / substeps
    horizon_duration = params.sampling_period * params.nb_timesteps

    live_plot = LivePlot(
        xlim=(0, horizon_duration + T),
        ylim=tuple(params.strides),
    )
    live_plot.add_line("pos", "b-")
    live_plot.add_line("cur_pos", "bo", lw=2)
    live_plot.add_line("cur_dcm", "go", lw=2)
    live_plot.add_line("cur_zmp", "ro", lw=2)
    live_plot.add_line("goal_pos", "ko", lw=2)
    live_plot.add_line("zmp", "r-")
    live_plot.add_line("zmp_min", "g:")
    live_plot.add_line("zmp_max", "b:")

    rate = RateLimiter(frequency=1.0 / (args.slowdown * dt))
    t = 0.0

    phase = PhaseStepper(params)
    support_foot_pos = params.init_support_foot_pos

    # Skip edge cases of separate initial and final DSP durations (2/2): here
    # we set the initial ZMP at the center of the initial foothold, and the
    # initial DCM halfway. See the LIPM walking controller and its
    # configuration for details on initial/final DSP phases.
    init_accel = -params.omega**2 * support_foot_pos
    init_vel = 0.5 * params.omega * support_foot_pos
    state = np.array([0.0, init_vel, init_accel])

    for _ in range(300):
        mpc_problem.update_initial_state(state)
        update_goal_and_constraints(mpc_problem, phase, support_foot_pos)
        plan = solve_mpc(mpc_problem, solver="proxqp")
        plot_plan(t, live_plot, params, mpc_problem, plan)
        for step in range(substeps):
            state = integrate(state, plan.inputs[0], dt)
            t2 = t + step * dt
            if t2 >= T:
                t3 = t2 - T
                live_plot.axis.set_xlim(t3, t3 + horizon_duration + T)
            cur_pos = state[0]
            cur_dcm = params.dcm_from_state.dot(state)
            cur_zmp = params.zmp_from_state.dot(state)
            live_plot.update_line("cur_pos", [t2], [cur_pos])
            live_plot.update_line("cur_dcm", [t2], [cur_dcm])
            live_plot.update_line("cur_zmp", [t2], [cur_zmp])
            live_plot.update_line(
                "goal_pos",
                [t2 + horizon_duration],
                [mpc_problem.goal_state[0]],
            )
            live_plot.update()
            rate.sleep()
        phase.advance()
        if phase.index == 0:
            support_foot_pos = phase.get_next_foot_pos(support_foot_pos)
            phase.advance_stride()
        t += T
