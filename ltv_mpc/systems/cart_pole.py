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

"""Cart-pole system."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..mpc_problem import MPCProblem
from ..plan import Plan
from ..utils import LivePlot


class CartPole:
    """Cart-pole system."""

    @dataclass
    class Parameters:
        """Parameters of the cart-pole system and receding horizon."""

        length: float = 0.6  # m
        max_ground_accel: float = 10  # m/s²
        nb_timesteps: int = 12
        sampling_period: float = 0.1  # [s]

        @property
        def omega(self) -> float:
            """Characteristic frequency of the inverted pendulum."""
            return np.sqrt(CartPole.GRAVITY / self.length)

        @property
        def horizon_duration(self) -> float:
            """Duration of the receding horizon in seconds."""
            return self.sampling_period * self.nb_timesteps

    # Constants
    GRAVITY: float = 9.81  # m/s²
    INPUT_DIM: int = 1
    STATE_DIM: int = 4

    # Attributes
    live_plot: Optional[LivePlot]
    params: Parameters
    state: np.ndarray

    @staticmethod
    def build_mpc_problem(
        params: Parameters,
        stage_input_cost_weight: Optional[float] = 1e-3,
        stage_state_cost_weight: Optional[float] = None,
        terminal_cost_weight: Optional[float] = 1.0,
    ) -> MPCProblem:
        """Build the model predictive control problem.

        Args:
            params: Problem parameters.
            stage_input_cost_weight: Weight on cumulated control costs.
            stage_state_cost_weight: Weight on cumulated state costs, or
                ``None`` to disable (default).
            terminal_cost_weight: Weight on terminal state cost, or ``None`` to
                disable.

        Returns:
            Model predictive control problem.
        """
        T = params.sampling_period
        omega = params.omega
        g = CartPole.GRAVITY

        A_disc = np.array(
            [
                [1.0, 0.0, T, 0.0],
                [0.0, np.cosh(T * omega), 0.0, np.sinh(T * omega) / omega],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, omega * np.sinh(T * omega), 0.0, np.cosh(T * omega)],
            ]
        )

        B_disc = np.array(
            [
                [T ** 2 / 2.0],
                [-np.cosh(T * omega) / g + 1.0 / g],
                [T],
                [-omega * np.sinh(T * omega) / g],
            ]
        )

        ground_accel_ineq_matrix = np.vstack([np.eye(1), -np.eye(1)])
        ground_accel_ineq_vector = np.hstack(
            [
                params.max_ground_accel,
                params.max_ground_accel,
            ]
        )

        return MPCProblem(
            transition_state_matrix=A_disc,
            transition_input_matrix=B_disc,
            ineq_state_matrix=None,
            ineq_input_matrix=ground_accel_ineq_matrix,
            ineq_vector=ground_accel_ineq_vector,
            goal_state=None,
            nb_timesteps=params.nb_timesteps,
            terminal_cost_weight=terminal_cost_weight,
            stage_state_cost_weight=stage_state_cost_weight,
            stage_input_cost_weight=stage_input_cost_weight,
        )

    @staticmethod
    def integrate(
        params: Parameters,
        state: np.ndarray,
        ground_accel: float,
        dt: float,
    ) -> np.ndarray:
        """Integrate from state with a constant control input.

        We follow a second-order Taylor expansion here, as opposed to the MPC
        problem which is based on an exact discretization.

        Args:
            params: Cart-pole parameters.
            state: Initial state.
            ground_accel: Constant control input to integrate.
            dt: Duration to integrate for, in seconds.

        Returns:
            State after integration.
        """
        r_0, theta_0, rd_0, thetad_0 = state
        rdd_0 = ground_accel
        thetadd_0 = params.omega ** 2 * (
            np.sin(theta_0) - (rdd_0 / CartPole.GRAVITY) * np.cos(theta_0)
        )

        def integrate_constant_accel(x, v, a):
            x2 = x + dt * (v + dt * (a / 2))
            v2 = v + dt * a
            return x2, v2

        r, rd = integrate_constant_accel(r_0, rd_0, rdd_0)
        theta, thetad = integrate_constant_accel(theta_0, thetad_0, thetadd_0)
        return np.array([r, theta, rd, thetad]).flatten()

    def __init__(
        self,
        params: Parameters,
        initial_state: np.ndarray,
    ):
        """Initialize a new cart-pole.

        Args:
            params: System description.
            initial_state: Initial state, a four-dimensional vector.
        """
        if initial_state is None:
            initial_state = np.zeros(CartPole.STATE_DIM)
        self.live_plot = None
        self.params = params
        self.state = initial_state

    def step(self, ground_accel: float, dt: float) -> None:
        """Integrate in-place from current state with a constant input.

        Args:
            ground_accel: Constant control input to integrate.
            dt: Duration to integrate for, in seconds.
        """
        self.state = CartPole.integrate(
            self.params, self.state, ground_accel, dt
        )

    def init_live_plot(self, order: str) -> None:
        """Initialize live plot.

        Args:
            order: Order of things to plot, "positions" or "velocities".
        """
        lhs_index = 0 if order == "positions" else 2
        rhs_index = 1 if order == "positions" else 3
        ps = "" if order == "positions" else "/s"
        T = self.params.sampling_period
        live_plot = LivePlot(
            xlim=(0.0, self.params.horizon_duration + T),
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
        self.lhs_index = lhs_index
        self.rhs_index = rhs_index
        self.live_plot = live_plot

    def update_live_plot(
        self,
        plan: Plan,
        plan_time: float,
        state_time: float,
    ) -> None:
        """Plot plan resulting from the MPC problem.

        Args:
            plan: Solution to the MPC problem.
            plan_time: Time of the beginning of the receding horizon.
            state_time: Time of the current state.
        """
        if self.live_plot is None:
            raise RuntimeError("Cannot update an uninitialized live plot")

        X = plan.states
        horizon_duration = self.params.horizon_duration
        nb_timesteps = self.params.nb_timesteps

        def update_plan_lines():
            t = plan_time
            trange = np.linspace(t, t + horizon_duration, nb_timesteps + 1)
            self.live_plot.update_line("lhs", trange, X[:, self.lhs_index])
            self.live_plot.update_line("rhs", trange, X[:, self.rhs_index])
            if plan.problem.target_states is not None:
                self.live_plot.update_line(
                    "lhs_goal",
                    trange[:-1],
                    plan.problem.target_states[self.lhs_index :: 4],
                )
                self.live_plot.update_line(
                    "rhs_goal",
                    trange[:-1],
                    plan.problem.target_states[self.rhs_index :: 4],
                )

        def update_state_lines():
            T = self.params.sampling_period
            if state_time >= T:
                t2 = state_time - T
                self.live_plot.axis.set_xlim(t2, t2 + horizon_duration + T)
            self.live_plot.update_line(
                "lhs_cur", [state_time], [self.state[self.lhs_index]]
            )
            self.live_plot.update_line(
                "rhs_cur", [state_time], [self.state[self.rhs_index]]
            )

        update_plan_lines()
        update_state_lines()
        self.live_plot.update()
