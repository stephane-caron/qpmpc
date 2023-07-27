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

"""Cart-pole system and receding horizon."""

from typing import Optional

import numpy as np

from ..mpc_problem import MPCProblem


class CartPole:
    """Cart-pole system and receding horizon."""

    GRAVITY: float = 9.81  # m/s²
    INPUT_DIM: int = 1
    STATE_DIM: int = 4

    length: float
    max_ground_accel: float
    nb_timesteps: int
    sampling_period: float

    def __init__(
        self,
        length: float = 0.6,
        max_ground_accel: float = 10,
        nb_timesteps: int = 12,
        sampling_period: float = 0.1,
    ):
        """Initialize a new cart-pole model.

        Args:
            length: Length of the pole.
            max_ground_accel: Maximum acceleration of the ground point, in m/s.
            nb_timesteps: Number of timesteps in the receding horizon.
            sampling_period: Duration of a timestep in the receding horizon, in
                seconds.
        """
        self.length = length
        self.max_ground_accel = max_ground_accel
        self.nb_timesteps = nb_timesteps
        self.sampling_period = sampling_period

    @property
    def omega(self) -> float:
        """Characteristic frequency of the inverted pendulum."""
        return np.sqrt(self.GRAVITY / self.length)

    @property
    def horizon_duration(self) -> float:
        """Duration of the receding horizon in seconds."""
        return self.sampling_period * self.nb_timesteps

    def build_mpc_problem(
        self,
        stage_input_cost_weight: float = 1e-3,
        stage_state_cost_weight: Optional[float] = None,
        terminal_cost_weight: Optional[float] = 1.0,
    ) -> MPCProblem:
        """Build the model predictive control problem.

        Args:
            stage_input_cost_weight: Weight on cumulated control costs.
            stage_state_cost_weight: Weight on cumulated state costs, or
                ``None`` to disable (default).
            terminal_cost_weight: Weight on terminal state cost, or ``None`` to
                disable.

        Returns:
            Model predictive control problem.
        """
        T = self.sampling_period
        omega = self.omega
        g = self.GRAVITY

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
                self.max_ground_accel,
                self.max_ground_accel,
            ]
        )

        return MPCProblem(
            transition_state_matrix=A_disc,
            transition_input_matrix=B_disc,
            ineq_state_matrix=None,
            ineq_input_matrix=ground_accel_ineq_matrix,
            ineq_vector=ground_accel_ineq_vector,
            nb_timesteps=self.nb_timesteps,
            terminal_cost_weight=terminal_cost_weight,
            stage_state_cost_weight=stage_state_cost_weight,
            stage_input_cost_weight=stage_input_cost_weight,
            initial_state=None,
            goal_state=None,
            target_states=None,
        )

    def integrate(
        self,
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
        thetadd_0 = self.omega ** 2 * (
            np.sin(theta_0) - (rdd_0 / self.GRAVITY) * np.cos(theta_0)
        )

        def integrate_constant_accel(x, v, a):
            x2 = x + dt * (v + dt * (a / 2))
            v2 = v + dt * a
            return x2, v2

        r, rd = integrate_constant_accel(r_0, rd_0, rdd_0)
        theta, thetad = integrate_constant_accel(theta_0, thetad_0, thetadd_0)
        return np.array([r, theta, rd, thetad]).flatten()
