#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2023 Inria
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

"""Live plot for the cart-pole system."""

import numpy as np

from ..exceptions import PlanError
from ..plan import Plan
from ..systems import CartPole
from .live_plot import LivePlot


class CartPolePlot:
    """Live plot for the cart-pole system."""

    live_plot: LivePlot
    cart_pole: CartPole
    lhs_index: int
    rhs_index: int

    def __init__(self, cart_pole: CartPole, order: str) -> None:
        """Initialize live plot.

        Args:
            cart_pole: Cart-pole system.
            order: Order of things to plot, "positions" or "velocities".
        """
        lhs_index = 0 if order == "positions" else 2
        rhs_index = 1 if order == "positions" else 3
        ps = "" if order == "positions" else "/s"
        T = cart_pole.sampling_period
        live_plot = LivePlot(
            xlim=(0.0, cart_pole.horizon_duration + T),
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
        self.cart_pole = cart_pole
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
        horizon_duration = self.cart_pole.horizon_duration
        nb_timesteps = self.cart_pole.nb_timesteps
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
        horizon_duration = self.cart_pole.horizon_duration
        T = self.cart_pole.sampling_period
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
