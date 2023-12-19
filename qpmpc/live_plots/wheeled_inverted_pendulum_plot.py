#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
# SPDX-License-Identifier: Apache-2.0

"""Live plot for the cart-pole system."""

import numpy as np

from ..exceptions import PlanError
from ..plan import Plan
from ..systems import WheeledInvertedPendulum
from .live_plot import LivePlot


class WheeledInvertedPendulumPlot:
    """Live plot for the cart-pole system."""

    live_plot: LivePlot
    pendulum: WheeledInvertedPendulum
    lhs_index: int
    rhs_index: int

    def __init__(self, pendulum: WheeledInvertedPendulum, order: str) -> None:
        """Initialize live plot.

        Args:
            pendulum: Wheeled inverted pendulum system.
            order: Order of things to plot, "positions" or "velocities".
        """
        lhs_index = 0 if order == "positions" else 2
        rhs_index = 1 if order == "positions" else 3
        ps = "" if order == "positions" else "/s"
        T = pendulum.sampling_period
        live_plot = LivePlot(
            xlim=(0.0, pendulum.horizon_duration + T),
            ylim=(-0.5, 1.0),
            ylim2=(-1.0, 1.0),
        )
        live_plot.add_line("lhs", "b-")
        live_plot.axis.set_ylabel(f"Ground {order} [m{ps}]", color="b")
        live_plot.axis.tick_params(axis="y", labelcolor="b")
        live_plot.add_rhs_line("rhs", "g-")
        if live_plot.rhs_axis is not None:  # help mypy
            label = f"Angular {order} [rad{ps}]"
            live_plot.rhs_axis.set_ylabel(label, color="g")
            live_plot.rhs_axis.tick_params(axis="y", labelcolor="g")
        live_plot.add_line("lhs_cur", "bo", lw=2)
        live_plot.add_line("lhs_goal", "b--", lw=1)
        live_plot.add_rhs_line("rhs_goal", "g--", lw=1)
        live_plot.add_rhs_line("rhs_cur", "go", lw=2)
        self.pendulum = pendulum
        self.lhs_index = lhs_index
        self.live_plot = live_plot
        self.rhs_index = rhs_index

    def update_plan(self, plan: Plan, plan_time: float) -> None:
        """Update live-plot from plan.

        Args:
            plan: Solution to the MPC problem.
            plan_time: Time corresponding to the initial state.
        """
        if plan.states is None:
            raise PlanError("No state trajectory in plan")
        X = plan.states
        t = plan_time
        horizon_duration = self.pendulum.horizon_duration
        nb_timesteps = self.pendulum.nb_timesteps
        trange = np.linspace(t, t + horizon_duration, nb_timesteps + 1)
        self.live_plot.update_line("lhs", trange, X[:, self.lhs_index])
        self.live_plot.update_line("rhs", trange, X[:, self.rhs_index])
        if (
            plan.problem.target_states is None
            or plan.problem.goal_state is None
        ):
            return
        self.live_plot.update_line(
            "lhs_goal",
            trange,
            np.hstack(
                [
                    plan.problem.target_states[self.lhs_index :: 4],
                    plan.problem.goal_state[self.lhs_index],
                ]
            ),
        )
        self.live_plot.update_line(
            "rhs_goal",
            trange,
            np.hstack(
                [
                    plan.problem.target_states[self.rhs_index :: 4],
                    plan.problem.goal_state[self.rhs_index],
                ]
            ),
        )

    def update_state(self, state: np.ndarray, state_time: float):
        """Update live-plot from current state.

        Args:
            state: Current state of the system.
            state_time: Time corresponding to the state.
        """
        horizon_duration = self.pendulum.horizon_duration
        T = self.pendulum.sampling_period
        if state_time >= T:
            t2 = state_time - T
            self.live_plot.axis.set_xlim(t2, t2 + horizon_duration + T)
        self.live_plot.update_line(
            "lhs_cur", [state_time], [state[self.lhs_index]]
        )
        self.live_plot.update_line(
            "rhs_cur", [state_time], [state[self.rhs_index]]
        )

    def update(
        self,
        plan: Plan,
        plan_time: float,
        state: np.ndarray,
        state_time: float,
    ) -> None:
        """Plot plan resulting from the MPC problem.

        Args:
            plan: Solution to the MPC problem.
            plan_time: Time of the beginning of the receding horizon.
            state: Current state.
            state_time: Time of the current state.
        """
        self.update_plan(plan, plan_time)
        self.update_state(state, state_time)
        self.live_plot.update()
