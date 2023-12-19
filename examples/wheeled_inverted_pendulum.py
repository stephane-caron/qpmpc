#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2022 Stéphane Caron
# SPDX-License-Identifier: Apache-2.0

"""Model predictive control of a wheeled inverted pendulum.

This is one locomotion mode for Upkie: https://github.com/upkie/upkie
"""

import argparse

import numpy as np

import qpsolvers

try:
    from loop_rate_limiters import RateLimiter
except ImportError:
    raise ImportError(
        "This example requires an extra dependency. "
        "You can install it by `pip install qpmpc[extras]`"
    )

from qpmpc import solve_mpc
from qpmpc.live_plots import WheeledInvertedPendulumPlot
from qpmpc.systems import WheeledInvertedPendulum

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
    pendulum: WheeledInvertedPendulum, state: np.ndarray, target_vel: float
):
    """Define the reference state trajectory over the receding horizon.

    Args:
        state: Cart-pole state at the beginning of the horizon.
        target_vel: Target ground velocity in m/s.

    Returns:
        Goal state at the end of the horizon.
    """
    nx = pendulum.STATE_DIM
    T = pendulum.sampling_period
    target_states = np.zeros((pendulum.nb_timesteps + 1) * nx)
    for k in range(pendulum.nb_timesteps + 1):
        target_states[k * nx] = state[0] + (k * T) * target_vel
        target_states[k * nx + 2] = target_vel
    return target_states


if __name__ == "__main__":
    args = parse_command_line_arguments()
    pendulum = WheeledInvertedPendulum()
    live_plot = WheeledInvertedPendulumPlot(pendulum, order=args.plot)
    mpc_problem = pendulum.build_mpc_problem(
        terminal_cost_weight=10.0,
        stage_state_cost_weight=1.0,
        stage_input_cost_weight=1e-3,
    )

    dt = pendulum.sampling_period / NB_SUBSTEPS
    rate = RateLimiter(frequency=1.0 / (args.slowdown * dt), warn=False)
    state = np.zeros(pendulum.STATE_DIM)
    for t in np.arange(0.0, EXAMPLE_DURATION, pendulum.sampling_period):
        target_vel = 0.5 + (np.cos(t / 2.0) if args.tv_vel else 0.0)
        target_states = get_target_states(pendulum, state, target_vel)
        mpc_problem.update_initial_state(state)
        mpc_problem.update_goal_state(
            target_states[-WheeledInvertedPendulum.STATE_DIM :]
        )
        mpc_problem.update_target_states(
            target_states[: -WheeledInvertedPendulum.STATE_DIM]
        )
        plan = solve_mpc(mpc_problem, solver=args.solver)
        for step in range(NB_SUBSTEPS):
            state = pendulum.integrate(state, plan.first_input, dt)
            live_plot.update(
                plan=plan,
                plan_time=t,
                state=state,
                state_time=t + step * dt,
            )
            rate.sleep()

    print(f"Example ran for {EXAMPLE_DURATION} s, press Enter to quit")
    input()
