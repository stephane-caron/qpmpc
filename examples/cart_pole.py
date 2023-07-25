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

"""Model predictive control of a wheeled inverted pendulum.

This is one locomotion mode for Upkie: https://github.com/tasts-robots/upkie
"""

import argparse

import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter

from ltv_mpc import solve_mpc
from ltv_mpc.live_plots import CartPolePlot
from ltv_mpc.systems import CartPole

EXAMPLE_DURATION: float = 10.0  # seconds
NB_SUBSTEPS: int = 15  # number of integration substeps


def parse_command_line_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        help="Make target ground velocity vary over time",
        choices=["positions", "velocities"],
        default="positions",
    )
    parser.add_argument(
        "--tv-vel",
        help="Make target ground velocity vary over time",
        dest="tv_vel",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--slowdown",
        help="Slow time down by a multiplicative factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--solver",
        help="QP solver called to solve MPC problems",
        choices=qpsolvers.available_solvers,
        default="proxqp",
    )
    return parser.parse_args()


def get_target_states(
    cart_pole: CartPole, state: np.ndarray, target_vel: float
):
    """Define the reference state trajectory over the receding horizon.

    Args:
        state: Cart-pole state at the beginning of the horizon.
        target_vel: Target ground velocity in m/s.

    Returns:
        Goal state at the end of the horizon.
    """
    nx = cart_pole.STATE_DIM
    T = cart_pole.sampling_period
    target_states = np.zeros((cart_pole.nb_timesteps + 1) * nx)
    for k in range(cart_pole.nb_timesteps + 1):
        target_states[k * nx] = state[0] + (k * T) * target_vel
        target_states[k * nx + 2] = target_vel
    return target_states


if __name__ == "__main__":
    args = parse_command_line_arguments()
    cart_pole = CartPole()
    live_plot = CartPolePlot(cart_pole, order=args.plot)
    mpc_problem = CartPole.build_mpc_problem(
        cart_pole,
        terminal_cost_weight=10.0,
        stage_state_cost_weight=1.0,
        stage_input_cost_weight=1e-3,
    )

    dt = cart_pole.sampling_period / NB_SUBSTEPS
    rate = RateLimiter(frequency=1.0 / (args.slowdown * dt), warn=False)
    state = np.zeros(cart_pole.STATE_DIM)
    for t in np.arange(0.0, EXAMPLE_DURATION, cart_pole.sampling_period):
        target_vel = 0.5 + (np.cos(t / 2.0) if args.tv_vel else 0.0)
        target_states = get_target_states(cart_pole, state, target_vel)
        mpc_problem.update_initial_state(state)
        mpc_problem.update_goal_state(target_states[-CartPole.STATE_DIM :])
        mpc_problem.update_target_states(target_states[: -CartPole.STATE_DIM])
        plan = solve_mpc(mpc_problem, solver=args.solver)
        for step in range(NB_SUBSTEPS):
            state = cart_pole.integrate(state, plan.first_input, dt)
            live_plot.update(
                plan=plan,
                plan_time=t,
                state=state,
                state_time=t + step * dt,
            )
            rate.sleep()

    print(f"Example ran for {EXAMPLE_DURATION} s, press Enter to quit")
    input()
